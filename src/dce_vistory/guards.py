# src/dce_vistory/guards.py
# Purpose:
#   1) Prevent planner/storyboard/image prompts from inventing forbidden characters.
#   2) Remove "woodcutter/lumberjack/axe" leakage when the input JSON only contains panda.
#   3) Keep the pipeline input-grounded: story must be based only on input text, image, and simple user content.

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_FORBIDDEN_ALIASES: Dict[str, List[str]] = {
    "woodcutter": [
        "woodcutter", "lumberjack", "logger", "woodsman", "axeman",
        "man with an axe", "person with an axe", "wood cutter",
        "나무꾼", "벌목꾼", "도끼 든 남자", "도끼 든 사람"
    ],
    "axe": ["axe", "hatchet", "도끼"],
}

HUMAN_ALIASES = [
    "human", "person", "man", "woman", "boy", "girl", "old man", "farmer", "villager",
    "사람", "남자", "여자", "소년", "소녀", "노인", "농부", "마을 사람"
]


@dataclass
class StoryConstraints:
    protagonist: str
    allowed_characters: List[str] = field(default_factory=list)
    allowed_objects: List[str] = field(default_factory=list)
    allowed_locations: List[str] = field(default_factory=list)
    forbidden_characters: List[str] = field(default_factory=list)
    forbidden_objects: List[str] = field(default_factory=list)
    allow_new_characters: bool = False
    allow_humans: bool = False

    @property
    def forbidden_terms(self) -> List[str]:
        terms: List[str] = []
        for key in self.forbidden_characters + self.forbidden_objects:
            lower = key.lower().strip()
            terms.append(key)
            terms.extend(DEFAULT_FORBIDDEN_ALIASES.get(lower, []))

        # If the story is animal-only, block accidental human antagonists.
        if not self.allow_humans:
            terms.extend(HUMAN_ALIASES)

        # Always block woodcutter leakage unless explicitly allowed.
        allowed_blob = " ".join(self.allowed_characters + self.allowed_objects).lower()
        if "woodcutter" not in allowed_blob and "나무꾼" not in allowed_blob:
            terms.extend(DEFAULT_FORBIDDEN_ALIASES["woodcutter"])
            terms.extend(DEFAULT_FORBIDDEN_ALIASES["axe"])

        # Deduplicate while preserving order.
        seen = set()
        out = []
        for t in terms:
            t = str(t).strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)
        return out


class ForbiddenEntityError(ValueError):
    pass


def constraints_from_input(input_data: Dict[str, Any]) -> StoryConstraints:
    protagonist = (
        input_data.get("protagonist")
        or input_data.get("main_character")
        or input_data.get("character")
        or "main protagonist"
    )

    allowed_characters = input_data.get("allowed_characters") or [protagonist]
    allowed_objects = input_data.get("allowed_objects") or input_data.get("objects") or []
    allowed_locations = input_data.get("allowed_locations") or input_data.get("locations") or []

    # User can explicitly add forbidden terms. Woodcutter is blocked by default unless allowed.
    forbidden_characters = input_data.get("forbidden_characters") or ["woodcutter", "lumberjack", "나무꾼"]
    forbidden_objects = input_data.get("forbidden_objects") or ["axe", "hatchet", "도끼"]

    allow_new_characters = bool(input_data.get("allow_new_characters", False))
    allow_humans = bool(input_data.get("allow_humans", False))

    # If any allowed character is human-like, do not globally ban humans.
    allowed_blob = " ".join(map(str, allowed_characters)).lower()
    human_keywords = ["human", "person", "man", "woman", "boy", "girl", "사람", "남자", "여자", "소년", "소녀"]
    if any(k in allowed_blob for k in human_keywords):
        allow_humans = True

    return StoryConstraints(
        protagonist=str(protagonist),
        allowed_characters=list(map(str, allowed_characters)),
        allowed_objects=list(map(str, allowed_objects)),
        allowed_locations=list(map(str, allowed_locations)),
        forbidden_characters=list(map(str, forbidden_characters)),
        forbidden_objects=list(map(str, forbidden_objects)),
        allow_new_characters=allow_new_characters,
        allow_humans=allow_humans,
    )


def _flatten_text(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def find_forbidden_terms(obj: Any, constraints: StoryConstraints) -> List[str]:
    text = _flatten_text(obj)
    hits = []
    for term in constraints.forbidden_terms:
        pattern = re.escape(term)
        if re.search(pattern, text, flags=re.IGNORECASE):
            hits.append(term)
    # Deduplicate
    seen = set()
    return [h for h in hits if not (h.lower() in seen or seen.add(h.lower()))]


def assert_no_forbidden_terms(obj: Any, constraints: StoryConstraints, where: str = "object") -> None:
    hits = find_forbidden_terms(obj, constraints)
    if hits:
        raise ForbiddenEntityError(f"{where} contains forbidden terms: {hits}")


def hard_negative_prompt(constraints: StoryConstraints) -> str:
    terms = constraints.forbidden_terms
    generic_bad = [
        "extra character", "new character", "unrelated person", "random human",
        "duplicate protagonist", "different protagonist", "inconsistent identity",
        "text, watermark, logo, blurry, low quality, deformed"
    ]
    return ", ".join(terms + generic_bad)


def allowed_entities_text(constraints: StoryConstraints) -> str:
    return (
        f"Allowed characters ONLY: {constraints.allowed_characters}. "
        f"Allowed objects: {constraints.allowed_objects}. "
        f"Allowed locations: {constraints.allowed_locations}. "
        f"Forbidden terms: {constraints.forbidden_terms}. "
        f"allow_new_characters={constraints.allow_new_characters}."
    )
