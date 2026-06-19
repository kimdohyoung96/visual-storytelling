from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from typing import Any, Dict, List
import json
import re

from .llm import BaseLLM
from .schema import StorySeed, DCEPlan, EmotionArc, StoryboardFrame, ImageUnderstanding, CharacterProfile
from .prompts import (
    SYSTEM_NARRATIVE,
    QUALITY_SUFFIX,
    NEGATIVE_PROMPT,
    story_seed_prompt,
    story_abstract_prompt,
    dcee_branch_plan_prompt,
    dcee_candidate_selection_prompt,
    emotion_arc_prompt,
    storyboard_prompt,
    canonicalize_storyboard_prompt,
    get_emotion_rule,
    emotion_rule_text,
    emotion_delta_text,
    choose_shot_type,
    choose_camera_distance,
)
from .utils import extract_json


def _field_names(cls) -> set[str]:
    return {f.name for f in fields(cls)} if is_dataclass(cls) else set()


def _safe_make(cls, kwargs: Dict[str, Any]):
    names = _field_names(cls)
    if names:
        init_kwargs = {k: v for k, v in kwargs.items() if k in names}
        obj = cls(**init_kwargs)
        for k, v in kwargs.items():
            if k not in names:
                try:
                    setattr(obj, k, v)
                except Exception:
                    pass
        return obj
    return cls(**kwargs)


def _to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if is_dataclass(obj):
        d = asdict(obj)
        d.update({k: v for k, v in getattr(obj, "__dict__", {}).items() if k not in d})
        return d
    if isinstance(obj, dict):
        return obj
    return getattr(obj, "__dict__", {}) or {}


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    elif isinstance(value, dict):
        value = [value]
    elif not isinstance(value, list):
        value = [value]

    out = []
    for item in value:
        if item is None:
            continue
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            text = (
                item.get("name")
                or item.get("object")
                or item.get("item")
                or item.get("description")
                or item.get("title")
                or item.get("event")
                or item.get("visual_grounding")
                or str(item)
            )
        else:
            text = str(item)
        text = str(text).strip()
        if text:
            out.append(text)
    return list(dict.fromkeys(out))


def _clean_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()


def _contains_generic_text(x: Any) -> bool:
    blob = json.dumps(x, ensure_ascii=False).lower() if isinstance(x, (dict, list)) else str(x).lower()
    bad = [
        "resolve the central problem",
        "discovers the problem",
        "conflict becomes visible",
        "decisive event changes the outcome",
        "object or place that starts the story",
        "an obstacle, rival, loss, or failed attempt",
        "the protagonist faces the object",
    ]
    return any(t in blob for t in bad)


def _ending_synonym(target: str) -> str:
    t = (target or "").lower().strip()
    mapping = {
        "happy": "joy",
        "happiness": "joy",
        "joyful": "joy",
        "sad": "sadness",
        "sad ending": "sadness",
        "angry": "anger",
        "fearful": "fear",
        "scared": "fear",
        "relieved": "relief",
        "regretful": "regret",
    }
    return mapping.get(t, t or "resolution")


def _target_family(target: str) -> str:
    t = _ending_synonym(target)
    if t in {"joy", "relief", "gratitude", "hope", "happiness"}:
        return "positive"
    if t in {"sadness", "regret", "grief", "fear", "anger", "despair"}:
        return "negative"
    return "neutral"


def _ensure_identity_fields(sample: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    profile = dict(profile or {})
    protagonist = sample.get("protagonist", profile.get("name", "protagonist"))
    profile.setdefault("name", protagonist)
    profile.setdefault("role", "protagonist")
    profile.setdefault("age_group", sample.get("age_group", "adult"))
    profile.setdefault("gender", sample.get("gender", "male" if "woodcutter" in str(protagonist).lower() else "unspecified"))
    profile.setdefault("outfit", sample.get("outfit", "same main outfit, same colors, same accessories in every frame"))
    profile.setdefault("signature_items", sample.get("signature_items", []))
    profile.setdefault("face", sample.get("face", f"consistent recognizable {profile.get('age_group','adult')} {profile.get('gender','person')} face"))
    profile.setdefault("hair", sample.get("hair", "same hairstyle and head shape in every frame"))
    profile.setdefault("body", sample.get("body", "same body shape, height, and proportions in every frame"))
    profile.setdefault("color_palette", sample.get("protagonist_color_palette", "earth-tone stable protagonist palette"))
    profile.setdefault(
        "identity_anchor_prompt",
        (
            f"{profile['name']} is the SAME {profile['age_group']} {profile['gender']} protagonist in every frame; "
            f"same face shape, same age, same gender presentation, same hairstyle, same body proportions, "
            f"same outfit ({profile['outfit']}), and same signature items ({profile.get('signature_items', [])}). "
            "Only facial expression, body pose, story event, emotion, and background may change."
        ),
    )
    profile.setdefault(
        "negative_identity_prompt",
        "different person, child version, older version, gender changed, different face, different hairstyle, different body shape, different outfit",
    )
    return profile


class DCEPlanner:
    """
    Strict DCEE-CausalVerse planner.

    This class uses API calls only.
    It does not use DummyLLM, static fallback stories, or generic fallback plans.
    If the API output is empty, generic, or structurally invalid, it retries with a stronger prompt.
    If it still fails, it raises RuntimeError so the user can verify the problem immediately.
    """

    def __init__(self, llm: BaseLLM, temperature: float = 0.4, max_tokens: int = 1800):
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = min(int(max_tokens), 1800)

    def _llm_text(self, prompt: str, max_tokens: int | None = None, temperature: float | None = None) -> str:
        text = self.llm.generate(
            SYSTEM_NARRATIVE,
            prompt,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=min(max_tokens or self.max_tokens, self.max_tokens),
        )
        if not _clean_text(text):
            raise RuntimeError("LLM returned empty text.")
        return text

    def _llm_json_strict(
        self,
        prompt: str,
        stage: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        repair_hint: str = "",
    ):
        errors = []
        for attempt in range(2):
            current_prompt = prompt
            if attempt > 0:
                current_prompt = (
                    prompt
                    + "\n\nYour previous response was invalid for the DCEE-CausalVerse final code. "
                    + "Return valid JSON only. Do not use generic placeholders. "
                    + repair_hint
                )
            try:
                text = self._llm_text(current_prompt, max_tokens=max_tokens, temperature=temperature)
                data = extract_json(text)
                if data is None or data == {} or data == []:
                    raise ValueError("Parsed JSON is empty.")
                if _contains_generic_text(data):
                    raise ValueError("Parsed JSON contains generic fallback phrases.")
                return data
            except Exception as e:
                errors.append(f"attempt {attempt + 1}: {type(e).__name__}: {e}")
        raise RuntimeError(f"Strict LLM JSON generation failed at stage={stage}. " + " | ".join(errors))

    # ------------------------------------------------------------------
    # Seed
    # ------------------------------------------------------------------

    def _repair_seed_objects_with_api(self, sample: Dict[str, Any], image_summary: ImageUnderstanding | None, current_seed_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Strict API-based repair step for seed objects.
        This is not DummyLLM or static fallback. It calls the configured LLM again.
        """
        repair_payload = {
            "sample": sample,
            "image_summary": _to_dict(image_summary) if image_summary else None,
            "current_seed_json": current_seed_json,
        }
        prompt = f"""
You are repairing the `objects` and `characters` fields for a DCEE-CausalVerse visual storytelling seed.

The previous seed JSON had empty or invalid concrete objects. This is not allowed.

Input data:
{json.dumps(repair_payload, ensure_ascii=False, indent=2)}

Return JSON only with this exact schema:
{{
  "objects": ["concrete visual object 1", "concrete visual object 2", "concrete visual object 3"],
  "characters": ["concrete character 1", "concrete character 2"],
  "setting": "concrete visual setting",
  "world_context": {{
    "location": "where the story visually happens",
    "weather_prior": "weather if inferable, otherwise a visually suitable weather",
    "time_of_day": "time of day if inferable",
    "environment_prior": ["concrete background detail 1", "concrete background detail 2"]
  }},
  "visual_symbols": {{
    "object name": "causal/emotional meaning"
  }},
  "character_profiles": [
    {{
      "name": "protagonist name",
      "role": "protagonist",
      "age_group": "adult or child or unknown",
      "gender": "unspecified if unknown",
      "face": "consistent face description",
      "hair": "consistent hair/head description",
      "body": "consistent body/proportions description",
      "outfit": "consistent outfit description",
      "signature_items": ["item"],
      "color_palette": "stable color palette",
      "identity_anchor_prompt": "same identity prompt"
    }}
  ]
}}

Strict requirements:
- objects must be non-empty concrete nouns visible in images.
- For a woodcutter story, include objects such as old iron axe, river, riverbank, wooden axe handle, fairy, golden axe, silver axe if relevant to the target ending emotion.
- Do not return empty arrays.
- Do not use generic words like object, thing, central problem, obstacle.
- Return JSON only.
""".strip()

        data = self._llm_json_strict(
            prompt,
            stage="repair_seed_objects",
            max_tokens=1200,
            temperature=0.0,
            repair_hint="Return non-empty concrete objects and character_profiles.",
        )
        if not isinstance(data, dict):
            raise RuntimeError("Seed object repair did not return a dictionary.")
        if not _string_list(data.get("objects", [])):
            raise RuntimeError("Seed object repair returned empty objects.")
        if _contains_generic_text(data):
            raise RuntimeError("Seed object repair returned generic placeholder text.")
        return data


    def build_seed(self, sample: Dict[str, Any], image_summary: ImageUnderstanding | None) -> StorySeed:
        prompt = (
            story_seed_prompt(sample, _to_dict(image_summary) if image_summary else None)
            + "\n\nSTRICT REQUIREMENTS:\n"
            "- Return JSON only.\n"
            "- Include concrete setting, objects, characters, visual_symbols, and character_profiles.\n"
            "- If the story is about a woodcutter, axe, river, or fairy, explicitly include these objects.\n"
            "- Do not use placeholders such as 'central problem' or 'object'.\n"
        )
        data = self._llm_json_strict(
            prompt,
            stage="build_seed",
            max_tokens=1200,
            temperature=self.temperature,
            repair_hint="Required keys: setting, objects, characters, mood, visual_symbols, world_context, character_profiles.",
        )
        if not isinstance(data, dict):
            raise RuntimeError("Seed JSON must be a dictionary.")

        data["objects"] = _string_list(data.get("objects", []))
        data["characters"] = _string_list(data.get("characters", []))

        # Strict API repair: if the first API seed response lacks concrete objects,
        # call the API again with a dedicated object extraction schema.
        # This keeps strict mode: no DummyLLM and no static fallback.
        if not data["objects"]:
            repaired = self._repair_seed_objects_with_api(sample, image_summary, data)
            for key, value in repaired.items():
                if key in {"objects", "characters", "setting", "world_context", "visual_symbols", "character_profiles"}:
                    if key in {"objects", "characters"}:
                        data[key] = _string_list(value)
                    elif key == "world_context" and not isinstance(data.get("world_context", {}), dict):
                        data[key] = value
                    elif key == "character_profiles" and not data.get("character_profiles"):
                        data[key] = value
                    elif not data.get(key):
                        data[key] = value

        if not data["objects"]:
            raise RuntimeError("Seed JSON has no concrete objects even after strict API repair.")
        if not data["characters"]:
            raise RuntimeError("Seed JSON has no concrete characters even after strict API repair.")

        profiles = self._build_character_profiles(data, sample, image_summary)

        world_context = data.get("world_context", {})

        # Strict API repair: some models return world_context as a string/list even when the prompt asks for a dict.
        # This is a schema issue, not a DCEE failure. Repair it with the API instead of using DummyLLM/static fallback.
        if not isinstance(world_context, dict):
            repaired = self._repair_seed_objects_with_api(sample, image_summary, data)
            for key, value in repaired.items():
                if key in {"objects", "characters"}:
                    data[key] = _string_list(value)
                elif key in {"setting", "world_context", "visual_symbols", "character_profiles"}:
                    data[key] = value
            world_context = data.get("world_context", {})

        if not isinstance(world_context, dict):
            raise RuntimeError(
                f"world_context must be a dictionary in strict mode even after API repair. Got: {type(world_context).__name__}"
            )

        # Fill missing expected keys from image understanding only; this is not a semantic fallback.
        if image_summary:
            world_context.setdefault("time_of_day", getattr(image_summary, "time_of_day", ""))
            world_context.setdefault("weather_prior", getattr(image_summary, "weather", ""))
            world_context.setdefault("environment_prior", getattr(image_summary, "environment_details", []))

        seed = _safe_make(
            StorySeed,
            {
                "image_summary": image_summary,
                "text_prompt": sample.get("text_prompt", sample.get("story", sample.get("prompt", ""))),
                "protagonist": sample.get("protagonist", data.get("protagonist", "protagonist")),
                "target_ending_emotion": sample.get("target_ending_emotion", data.get("target_ending_emotion", "")),
                "genre": sample.get("genre", data.get("genre", "")),
                "style": sample.get("style", data.get("style", "")),
                "setting": data.get("setting", ""),
                "objects": data.get("objects", []),
                "characters": data.get("characters", []),
                "mood": data.get("mood", ""),
                "visual_symbols": data.get("visual_symbols", {}),
                "world_context": world_context,
                "character_profiles": profiles,
                "raw_input": sample,
            },
        )
        seed.world_context = world_context
        seed.character_profiles = profiles
        seed.raw_input = sample
        return seed

    def _build_character_profiles(self, data, sample, image_summary):
        profiles = []
        raw = data.get("character_profiles", [])
        if isinstance(raw, dict):
            raw = [raw]
        if not isinstance(raw, list):
            raise RuntimeError("character_profiles must be a list or dict in strict mode.")

        for row in raw:
            if not isinstance(row, dict):
                raise RuntimeError("Each character profile must be a dictionary.")
            row = _ensure_identity_fields(sample, row)
            profiles.append(
                _safe_make(
                    CharacterProfile,
                    {
                        "name": row.get("name", ""),
                        "role": row.get("role", ""),
                        "age_group": row.get("age_group", "adult"),
                        "gender": row.get("gender", "unspecified"),
                        "face": row.get("face", ""),
                        "hair": row.get("hair", ""),
                        "body": row.get("body", ""),
                        "outfit": row.get("outfit", ""),
                        "signature_items": _string_list(row.get("signature_items", [])),
                        "color_palette": row.get("color_palette", ""),
                        "identity_anchor_prompt": row.get("identity_anchor_prompt", ""),
                        "negative_identity_prompt": row.get("negative_identity_prompt", ""),
                    },
                )
            )

        protagonist = sample.get("protagonist", data.get("protagonist", "protagonist"))
        if not any(
            getattr(p, "role", "") == "protagonist" or getattr(p, "name", "").lower() == str(protagonist).lower()
            for p in profiles
        ):
            row = _ensure_identity_fields(sample, {"name": protagonist, "role": "protagonist"})
            profiles.insert(0, _safe_make(CharacterProfile, {
                "name": row.get("name", protagonist),
                "role": "protagonist",
                "age_group": row.get("age_group", "adult"),
                "gender": row.get("gender", "unspecified"),
                "face": row.get("face", ""),
                "hair": row.get("hair", ""),
                "body": row.get("body", ""),
                "outfit": row.get("outfit", ""),
                "signature_items": _string_list(row.get("signature_items", [])),
                "color_palette": row.get("color_palette", ""),
                "identity_anchor_prompt": row.get("identity_anchor_prompt", ""),
                "negative_identity_prompt": row.get("negative_identity_prompt", ""),
            }))
        return profiles

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------
    def generate_abstract(self, seed: StorySeed) -> str:
        prompt = (
            story_abstract_prompt(_to_dict(seed))
            + "\n\nSTRICT REQUIREMENTS:\n"
            "- Write one concrete paragraph, 80-160 words.\n"
            "- Mention the protagonist, desire, conflict, event chain, visual evidence, and target ending emotion.\n"
            "- Do not return an empty response.\n"
            "- Do not use placeholders such as 'central problem'.\n"
        )
        text = self._llm_text(prompt, max_tokens=500, temperature=self.temperature).strip()
        if not text or _contains_generic_text(text):
            raise RuntimeError(f"Invalid abstract generated in strict mode: {text[:300]}")
        return text

    # ------------------------------------------------------------------
    # DCEE Plan
    # ------------------------------------------------------------------

    def _extract_candidate_list(self, data: Any) -> List[Dict[str, Any]]:
        """
        Normalize several common API response shapes into a DCEE candidate list.

        Accepted shapes:
        - {"candidates": [...]}
        - {"candidate_plans": [...]}
        - {"routes": [...]}
        - {"branches": [...]}
        - direct single candidate dict with desire/conflict/event_chain
        - {"route_1": {...}, "route_2": {...}}
        """
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]

        if not isinstance(data, dict):
            return []

        for key in ["candidates", "candidate_plans", "plans", "routes", "branches", "dcee_candidates"]:
            value = data.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
            if isinstance(value, dict):
                # Either a single candidate or dict of candidates
                if value.get("desire") and (value.get("event_chain") or value.get("event_spine")):
                    return [value]
                return [x for x in value.values() if isinstance(x, dict)]

        # Some models return the selected plan directly instead of wrapping it.
        if data.get("desire") and data.get("conflict") and (data.get("event_chain") or data.get("event_spine")):
            return [data]

        # Some models return {"candidate_1": {...}, "candidate_2": {...}}
        dict_values = [v for v in data.values() if isinstance(v, dict)]
        candidate_values = [
            v for v in dict_values
            if v.get("desire") and v.get("conflict") and (v.get("event_chain") or v.get("event_spine"))
        ]
        return candidate_values

    def _repair_dcee_candidates_with_api(self, seed: StorySeed, abstract: str, previous_response: Any, n: int) -> List[Dict[str, Any]]:
        """
        Strict API-based DCEE candidate repair.

        This is not DummyLLM and not static fallback.
        It calls the configured API again and forces an exact DCEE-Tree schema.
        """
        repair_payload = {
            "seed": _to_dict(seed),
            "abstract": abstract,
            "previous_response": previous_response,
            "num_candidates": n,
        }
        prompt = f"""
You are repairing DCEE-Tree planning output for the final DCEE-CausalVerse code.

The previous response did not contain a valid `candidates` list.

Input:
{json.dumps(repair_payload, ensure_ascii=False, indent=2)}

Return JSON only with this exact schema:

{{
  "candidates": [
    {{
      "candidate_id": "route_1",
      "protagonist": "specific protagonist name",
      "desire": "specific desire tied to visible objects",
      "fear": "specific fear",
      "misbelief": "specific misbelief or temptation",
      "obstacle": "specific external/internal obstacle",
      "conflict": "specific conflict between desire and obstacle",
      "event_chain": [
        {{
          "event_id": "e1",
          "event": "specific drawable event",
          "causal_role": "how this event changes desire/conflict/emotion",
          "visual_grounding": "what must be visible in the image",
          "emotion_effect": "emotion caused or intensified by this event",
          "key_objects": ["concrete visible object"],
          "evidence_objects": ["visible evidence object or clue"]
        }},
        {{
          "event_id": "e2",
          "event": "specific drawable event",
          "causal_role": "how this event escalates conflict",
          "visual_grounding": "what must be visible in the image",
          "emotion_effect": "emotion caused or intensified by this event",
          "key_objects": ["concrete visible object"],
          "evidence_objects": ["visible evidence object or clue"]
        }},
        {{
          "event_id": "e3",
          "event": "specific drawable turning-point event",
          "causal_role": "turning point",
          "visual_grounding": "what must be visible in the image",
          "emotion_effect": "target ending emotion or transition toward it",
          "key_objects": ["concrete visible object"],
          "evidence_objects": ["visible evidence object or clue"]
        }}
      ],
      "turning_point": "specific causal turning point",
      "target_ending_emotion": "target ending emotion",
      "ending_state": "specific final visible state",
      "moral_or_theme": "theme"
    }}
  ]
}}

Strict requirements:
- Generate at least {max(2, n)} candidates.
- Each candidate must have at least 3 events.
- Every event must have non-empty key_objects and evidence_objects.
- Use concrete visual nouns from the input seed.
- If target ending emotion is happy/joy/relief, make the event chain lead to visible reward or relief such as honesty rewarded, old axe returned, golden/silver axe offered, grateful smile, warm riverbank light. If target ending emotion is sad/regret/grief, make the event chain lead to visible loss such as lost axe, empty hands, rain, kneeling body, deserted riverbank. Do not hard-code sadness; always use input target_ending_emotion.
- Do not use generic phrases like "central problem", "discovers the problem", "conflict becomes visible", "object or place".
- Return JSON only.
""".strip()

        data = self._llm_json_strict(
            prompt,
            stage="repair_dcee_candidates",
            max_tokens=1800,
            temperature=0.0,
            repair_hint="Return {'candidates': [...]} with concrete event_chain/key_objects/evidence_objects.",
        )
        candidates = self._extract_candidate_list(data)
        if not candidates:
            raise RuntimeError("DCEE candidate repair returned no valid candidates.")
        return candidates


    def generate_dce_plan(self, seed: StorySeed, abstract: str) -> DCEPlan:
        n = int(getattr(seed, "raw_input", {}).get("num_dcee_candidates", 4) if isinstance(getattr(seed, "raw_input", {}), dict) else 4)
        n = max(2, min(6, n))
        prompt = (
            dcee_branch_plan_prompt(_to_dict(seed), abstract, num_candidates=n)
            + "\n\nSTRICT DCEE-TREE REQUIREMENTS:\n"
            "- Return JSON only with key `candidates`.\n"
            "- Generate multiple Desire->Conflict routes before selecting events.\n"
            "- Each candidate must include desire, conflict, event_chain, turning_point, target_ending_emotion.\n"
            "- Each event must include event_id, event, causal_role, visual_grounding, emotion_effect, key_objects, evidence_objects.\n"
            "- key_objects and evidence_objects must be non-empty concrete nouns.\n"
            "- For a woodcutter story, include concrete evidence such as old iron axe, river, fairy, golden axe, empty hands, rain, riverbank.\n"
            "- Do not use generic placeholders such as 'discovers the problem'.\n"
        )
        data = self._llm_json_strict(
            prompt,
            stage="generate_dce_plan_candidates",
            max_tokens=1800,
            temperature=max(0.55, self.temperature),
            repair_hint="The JSON must be {'candidates': [candidate, ...]}. Each event needs visual_grounding, key_objects, evidence_objects.",
        )

        candidates = self._extract_candidate_list(data)

        # Strict API repair: if the model returned a direct plan, a differently named key,
        # or an invalid candidate structure, call the API again with an exact schema.
        if not candidates:
            candidates = self._repair_dcee_candidates_with_api(seed, abstract, data, n)

        candidates = [self._normalize_candidate(c, i, seed) for i, c in enumerate(candidates)]
        try:
            self._validate_candidates(candidates)
        except Exception as e:
            candidates = self._repair_dcee_candidates_with_api(seed, abstract, {"invalid_candidates": candidates, "validation_error": str(e)}, n)
            candidates = [self._normalize_candidate(c, i, seed) for i, c in enumerate(candidates)]
            self._validate_candidates(candidates)

        selected = self._select_best_candidate(seed, abstract, candidates)
        event_chain = selected.get("event_chain", selected.get("event_spine", []))
        self._validate_event_chain(event_chain, context="selected_candidate")

        dce_plan = _safe_make(
            DCEPlan,
            {
                "protagonist": selected.get("protagonist", getattr(seed, "protagonist", "")),
                "desire": selected.get("desire", ""),
                "fear": selected.get("fear", ""),
                "misbelief": selected.get("misbelief", ""),
                "obstacle": selected.get("obstacle", ""),
                "conflict": selected.get("conflict", ""),
                "event_spine": event_chain,
                "turning_point": selected.get("turning_point", ""),
                "target_ending_emotion": selected.get("target_ending_emotion", selected.get("ending_emotion", getattr(seed, "target_ending_emotion", ""))),
                "ending_state": selected.get("ending_state", ""),
                "moral_or_theme": selected.get("moral_or_theme", ""),
                "event_chain": event_chain,
                "dcee_candidates": candidates,
                "candidate_plans": candidates,
                "selected_candidate": selected,
                "planning_structure": "Strict DCEE-Tree: API-generated multiple Desire-Conflict routes -> validated Event Chain -> Ending Emotion",
            },
        )
        dce_plan.event_chain = event_chain
        dce_plan.event_spine = event_chain
        dce_plan.dcee_candidates = candidates
        dce_plan.candidate_plans = candidates
        dce_plan.selected_candidate = selected
        dce_plan.planning_structure = "Strict DCEE-Tree: API-generated multiple Desire-Conflict routes -> validated Event Chain -> Ending Emotion"
        return dce_plan

    def _normalize_candidate(self, c, idx, seed):
        if not isinstance(c, dict):
            raise RuntimeError("Each DCEE candidate must be a dictionary in strict mode.")
        c.setdefault("candidate_id", f"c{idx + 1}")
        c.setdefault("protagonist", getattr(seed, "protagonist", ""))
        c.setdefault("target_ending_emotion", c.get("ending_emotion", getattr(seed, "target_ending_emotion", "")))
        c.setdefault("ending_emotion", c.get("target_ending_emotion", getattr(seed, "target_ending_emotion", "")))

        chain = c.get("event_chain", c.get("event_spine", []))
        if isinstance(chain, dict):
            chain = chain.get("events", [chain])
        if not isinstance(chain, list):
            raise RuntimeError(f"Candidate {c.get('candidate_id')} event_chain must be a list.")

        norm = []
        for j, e in enumerate(chain):
            if not isinstance(e, dict):
                raise RuntimeError(f"Event {j} in candidate {c.get('candidate_id')} must be a dictionary.")
            e.setdefault("event_id", f"e{j + 1}")
            e["key_objects"] = _string_list(e.get("key_objects", []))
            e["evidence_objects"] = _string_list(e.get("evidence_objects", e.get("visual_evidence_objects", [])))
            norm.append(e)

        c["event_chain"] = norm
        c["event_spine"] = norm
        return c

    def _validate_candidates(self, candidates):
        if len(candidates) < 1:
            raise RuntimeError("No DCEE candidates generated.")
        for c in candidates:
            if _contains_generic_text(c):
                raise RuntimeError(f"Generic DCEE candidate detected: {c.get('candidate_id')}")
            for key in ["desire", "conflict", "event_chain", "target_ending_emotion"]:
                if not c.get(key):
                    raise RuntimeError(f"DCEE candidate {c.get('candidate_id')} missing required key: {key}")
            self._validate_event_chain(c.get("event_chain", []), context=c.get("candidate_id", "candidate"))

    def _validate_event_chain(self, chain, context="event_chain"):
        if not isinstance(chain, list) or len(chain) < 3:
            raise RuntimeError(f"{context}: event_chain must contain at least 3 events.")
        for e in chain:
            for key in ["event", "causal_role", "visual_grounding", "emotion_effect"]:
                if not _clean_text(e.get(key, "")):
                    raise RuntimeError(f"{context}: event missing required key `{key}`: {e}")
            if not _string_list(e.get("key_objects", [])):
                raise RuntimeError(f"{context}: event has empty key_objects: {e}")
            if not _string_list(e.get("evidence_objects", [])):
                raise RuntimeError(f"{context}: event has empty evidence_objects: {e}")

    def _select_best_candidate(self, seed, abstract, candidates):
        prompt = (
            dcee_candidate_selection_prompt(_to_dict(seed), abstract, candidates)
            + "\n\nSTRICT SELECTION REQUIREMENTS:\n"
            "- Return JSON only.\n"
            "- Select the candidate with strongest causal coherence, ending emotion alignment, event richness, diversity, and visual evidentiality.\n"
            "- Return selected_candidate_id and reason.\n"
        )
        data = self._llm_json_strict(
            prompt,
            stage="select_best_candidate",
            max_tokens=900,
            temperature=0.0,
            repair_hint="Required keys: selected_candidate_id, reason.",
        )
        sid = str(data.get("selected_candidate_id", ""))
        for c in candidates:
            if str(c.get("candidate_id")) == sid:
                c["selection_scores"] = data.get("scores", [])
                c["selection_reason"] = data.get("reason", "")
                return c
        raise RuntimeError(f"LLM selected unknown candidate id: {sid}")

    # ------------------------------------------------------------------
    # Emotion Arc and Storyboard
    # ------------------------------------------------------------------
    def generate_emotion_arc(self, seed: StorySeed, abstract: str, dce_plan: DCEPlan, num_frames: int) -> EmotionArc:
        prompt = (
            emotion_arc_prompt(_to_dict(seed), abstract, _to_dict(dce_plan), num_frames)
            + "\n\nSTRICT REQUIREMENTS:\n"
            f"- Return JSON only with exactly {num_frames} states and {num_frames} intensities.\n"
            "- The final state must match target_ending_emotion.\n"
            "- Include valence_curve, arousal_curve, and suspense_curve if possible.\n"
        )
        data = self._llm_json_strict(
            prompt,
            stage="generate_emotion_arc",
            max_tokens=1000,
            temperature=self.temperature,
            repair_hint=f"Required keys: states, intensities. Both must have length {num_frames}.",
        )
        states = data.get("states", [])
        intensities = data.get("intensities", [])
        if len(states) != num_frames or len(intensities) != num_frames:
            raise RuntimeError(f"Emotion arc length mismatch. states={len(states)}, intensities={len(intensities)}, expected={num_frames}")
        target = _ending_synonym(getattr(dce_plan, "target_ending_emotion", getattr(seed, "target_ending_emotion", "")))
        if _ending_synonym(states[-1]) != target and target not in str(states[-1]).lower():
            raise RuntimeError(f"Final emotion state does not match target. final={states[-1]}, target={target}")

        return _safe_make(
            EmotionArc,
            {
                "states": states,
                "intensities": [max(1, min(5, int(x))) for x in intensities],
                "rationale": data.get("rationale", ""),
                "valence_curve": data.get("valence_curve", []),
                "arousal_curve": data.get("arousal_curve", []),
                "suspense_curve": data.get("suspense_curve", []),
            },
        )


    def _extract_storyboard_list(self, data: Any) -> List[Dict[str, Any]]:
        """
        Normalize common API storyboard response shapes.

        Accepted shapes:
        - [frame, frame, ...]
        - {"storyboard": [...]}
        - {"frames": [...]}
        - {"frame_sequence": [...]}
        - {"visual_storyboard": [...]}
        - {"1": {...}, "2": {...}} or {"frame_1": {...}, ...}
        """
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]

        if not isinstance(data, dict):
            return []

        for key in ["storyboard", "frames", "frame_sequence", "visual_storyboard", "shots", "scenes"]:
            value = data.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
            if isinstance(value, dict):
                vals = [v for v in value.values() if isinstance(v, dict)]
                if vals:
                    return vals

        # Sometimes the model returns a dict keyed by frame id.
        vals = [v for v in data.values() if isinstance(v, dict)]
        if vals and any(("event" in v or "event_grounding" in v or "caption" in v) for v in vals):
            return vals

        # Sometimes a single frame dict is returned. This is invalid for the full pipeline,
        # but returning it lets validation produce a clearer length error or trigger repair.
        if data.get("event") or data.get("event_grounding") or data.get("caption"):
            return [data]

        return []

    def _repair_storyboard_with_api(self, seed: StorySeed, abstract: str, dce_plan: DCEPlan, emotion_arc: EmotionArc, previous_response: Any, num_frames: int) -> List[Dict[str, Any]]:
        """
        Strict API-based storyboard repair.

        This is not DummyLLM and not static fallback.
        It calls the configured API again and forces exact frame-level DCEE evidence schema.
        """
        payload = {
            "seed": _to_dict(seed),
            "abstract": abstract,
            "dce_plan": _to_dict(dce_plan),
            "emotion_arc": _to_dict(emotion_arc),
            "previous_response": previous_response,
            "num_frames": num_frames,
        }
        prompt = f"""
You are repairing storyboard output for the final DCEE-CausalVerse code.

The previous response did not contain a valid storyboard/frames list.

Input:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Return JSON only with this exact schema:

{{
  "storyboard": [
    {{
      "frame_id": 1,
      "caption": "one-sentence visual caption",
      "narrative_function": "DCEE role of this frame",
      "event": "specific drawable event from or derived from the selected event_chain",
      "event_causal_role": "how this event affects desire/conflict/emotion",
      "event_grounding": "exact visible evidence of the event in the image",
      "emotion": "emotion state for this frame",
      "emotion_intensity": 1,
      "key_objects": ["concrete visible object"],
      "evidence_objects": ["concrete visible clue/object that proves the event"],
      "must_show": ["mandatory visible element 1", "mandatory visible element 2"],
      "visual_focus": "main visual focus",
      "protagonist_state": "visible body/face state of the protagonist",
      "desire_link": "how this frame connects to the protagonist desire",
      "conflict_level": 1,
      "scene_location": "concrete location",
      "weather": "visual weather",
      "time_of_day": "visual time of day",
      "atmosphere": "visual atmosphere",
      "environment_details": ["background detail"],
      "supporting_cast": ["character if visible"]
    }}
  ]
}}

Strict requirements:
- Return exactly {num_frames} frames.
- Each frame must have event, event_causal_role, event_grounding, emotion, emotion_intensity, key_objects, evidence_objects, must_show.
- key_objects and evidence_objects must be non-empty concrete visible nouns.
- Use the selected DCEE event_chain as the causal backbone.
- Spread the event chain across the frames.
- The final frame must visually support the target ending emotion.
- For a woodcutter/axe/river story, use target-specific concrete visible evidence: for happy/relief use returned old axe, fairy reward, warm light, grateful smile; for sad/regret use empty hands, rain, kneeling body, lost axe absence.
- Do not use generic phrases like "central problem", "conflict becomes visible", "object or place".
- Return JSON only.
""".strip()

        data = self._llm_json_strict(
            prompt,
            stage="repair_storyboard",
            max_tokens=1800,
            temperature=0.0,
            repair_hint="Return {'storyboard': [...]} with exact frame count and concrete evidence objects.",
        )
        frames = self._extract_storyboard_list(data)
        if not frames:
            raise RuntimeError("Storyboard repair returned no valid frames.")
        return frames


    def generate_storyboard(self, seed: StorySeed, abstract: str, dce_plan: DCEPlan, emotion_arc: EmotionArc) -> List[StoryboardFrame]:
        states = getattr(emotion_arc, "states", [])
        num_frames = len(states)
        prompt = (
            storyboard_prompt(_to_dict(seed), abstract, _to_dict(dce_plan), _to_dict(emotion_arc), num_frames)
            + "\n\nSTRICT STORYBOARD REQUIREMENTS:\n"
            f"- Return JSON only with exactly {num_frames} frames.\n"
            "- Each frame must include event, event_causal_role, event_grounding, emotion, emotion_intensity, key_objects, evidence_objects, must_show.\n"
            "- Events must be drawable and concrete.\n"
            "- Evidence objects must be visible clues that explain the emotion.\n"
            "- Do not use generic placeholders.\n"
        )
        raw_storyboard = self._llm_json_strict(
            prompt,
            stage="generate_storyboard",
            max_tokens=1800,
            temperature=self.temperature,
            repair_hint="Required key: storyboard or frames list. Every frame needs event/event_grounding/evidence_objects/must_show.",
        )

        rows = self._extract_storyboard_list(raw_storyboard)

        # Strict API repair: if the storyboard shape is not usable, ask the API again
        # with an exact schema rather than using DummyLLM/static fallback.
        if not isinstance(rows, list) or not rows:
            rows = self._repair_storyboard_with_api(seed, abstract, dce_plan, emotion_arc, raw_storyboard, num_frames)

        if len(rows) != num_frames or _contains_generic_text(rows):
            rows = self._repair_storyboard_with_api(
                seed,
                abstract,
                dce_plan,
                emotion_arc,
                {"invalid_rows": rows, "reason": f"len={len(rows)}, expected={num_frames}, generic={_contains_generic_text(rows)}"},
                num_frames,
            )

        if len(rows) != num_frames:
            raise RuntimeError(f"Storyboard length mismatch after strict API repair. got={len(rows)}, expected={num_frames}")
        if _contains_generic_text(rows):
            raise RuntimeError("Storyboard contains generic placeholder text even after strict API repair.")

        # Canonicalize with API, but strict validation remains.
        canon_prompt = (
            canonicalize_storyboard_prompt(_to_dict(seed), _to_dict(dce_plan), rows)
            + "\n\nSTRICT CANONICALIZATION REQUIREMENTS:\n"
            "- Return JSON only with the same number of frames.\n"
            "- Resolve all pronouns into concrete entities.\n"
            "- Preserve event, evidence_objects, must_show, and emotion."
        )
        crows_raw = self._llm_json_strict(
            canon_prompt,
            stage="canonicalize_storyboard",
            max_tokens=1600,
            temperature=0.0,
            repair_hint="Return a storyboard/frames list with concrete nouns and no pronouns.",
        )
        crows = self._extract_storyboard_list(crows_raw)

        if not isinstance(crows, list) or len(crows) != num_frames or _contains_generic_text(crows):
            crows = self._repair_storyboard_with_api(
                seed,
                abstract,
                dce_plan,
                emotion_arc,
                {"invalid_canonicalized_storyboard": crows_raw},
                num_frames,
            )

        if not isinstance(crows, list) or len(crows) != num_frames:
            raise RuntimeError("Canonicalized storyboard invalid length or type even after strict API repair.")
        if _contains_generic_text(crows):
            raise RuntimeError("Canonicalized storyboard contains generic placeholder text even after strict API repair.")

        return self._postprocess_storyboard(crows, seed, dce_plan, emotion_arc)

    def _postprocess_storyboard(self, rows, seed, dce_plan, emotion_arc):
        protagonist_profile = self._get_protagonist_profile(seed)
        protagonist_identity = self._profile_to_prompt(protagonist_profile)
        character_reference_prompt = (
            f"Use the same protagonist identity in every frame: {protagonist_identity}. "
            "Keep identity stable while allowing emotion-specific expressions and poses."
        )
        world_context = getattr(seed, "world_context", {}) or {}
        img = getattr(seed, "image_summary", None)
        base_time = world_context.get("time_of_day", getattr(img, "time_of_day", "") if img else "")
        base_weather = world_context.get("weather_prior", getattr(img, "weather", "") if img else "")
        base_env = world_context.get("environment_prior", getattr(img, "environment_details", []) if img else [])
        chain = getattr(dce_plan, "event_chain", getattr(dce_plan, "event_spine", [])) or []

        frames = []
        prev_emotion = None
        prev_world = None
        states = getattr(emotion_arc, "states", [])
        intensities = getattr(emotion_arc, "intensities", [])

        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                raise RuntimeError("Each storyboard row must be a dictionary after canonicalization.")

            emotion = row.get("emotion", states[idx] if states else "neutral")
            intensity = max(1, min(5, int(row.get("emotion_intensity", intensities[idx] if intensities else 3))))
            rule = get_emotion_rule(emotion)

            linked = chain[min(idx, len(chain) - 1)] if chain else {}
            if not isinstance(linked, dict):
                linked = {}

            ev = row.get("event") or linked.get("event")
            evrole = row.get("event_causal_role") or linked.get("causal_role")
            evground = row.get("event_grounding") or linked.get("visual_grounding")
            if not ev or not evground:
                raise RuntimeError(f"Storyboard frame {idx+1} missing event/event_grounding.")

            evidence_objects = _string_list(row.get("evidence_objects", linked.get("evidence_objects", [])))
            key_objects = _string_list(row.get("key_objects", linked.get("key_objects", [])))
            # LLMs sometimes return `must_show` as a string instead of a list.
            # Normalize before concatenation to keep strict mode but tolerate JSON type variance.
            must_show_raw = _string_list(row.get("must_show", []))
            must_show = _string_list(
                must_show_raw
                + key_objects[:3]
                + evidence_objects[:3]
                + [
                    ev,
                    evground,
                    f"facial evidence of {emotion}",
                    f"body evidence of {emotion}",
                    "full-color emotional lighting",
                ]
            )
            if not evidence_objects or not key_objects:
                raise RuntimeError(f"Storyboard frame {idx+1} missing key_objects/evidence_objects.")

            env = _string_list(row.get("environment_details", base_env)) + [
                f"lighting style: {rule['lighting']}",
                f"composition supports {emotion}: {rule['composition']}",
                "full-color environment, not grayscale",
            ]
            loc = row.get("scene_location", getattr(seed, "setting", ""))
            weather = row.get("weather", base_weather) or rule["weather"]
            transition = row.get(
                "scene_transition",
                "Establish the initial world state."
                if not prev_world
                else f"The scene evolves from {prev_world['scene_location']} in {prev_world['weather']} weather to {loc} in {weather} weather.",
            )
            nf = row.get("narrative_function", "DCEE event progression")
            shot = choose_shot_type(idx, len(rows), nf)
            cam = choose_camera_distance(shot)
            evidence = _string_list([ev, evground] + key_objects[:3] + evidence_objects[:3])

            frame = _safe_make(
                StoryboardFrame,
                {
                    "frame_id": int(row.get("frame_id", idx + 1)),
                    "caption": row.get("caption", evground),
                    "narrative_function": nf,
                    "event": ev,
                    "protagonist_state": row.get("protagonist_state", f"The protagonist visibly experiences {emotion}."),
                    "desire_link": row.get("desire_link", getattr(dce_plan, "desire", "")),
                    # conflict_level is narrative tension, not emotion intensity.
                    "conflict_level": int(row.get("conflict_level", min(5, max(1, round(1 + idx * 4 / max(1, len(rows) - 1)))))),
                    "emotion": emotion,
                    "emotion_intensity": intensity,
                    "visual_focus": row.get("visual_focus", evground),
                    "key_objects": key_objects,
                    "evidence_objects": evidence_objects,
                    "facial_cue": row.get("facial_cue") or rule["face"],
                    "body_cue": row.get("body_cue") or rule["body"],
                    "event_cue": row.get("event_cue", ev),
                    "scene_cue": row.get("scene_cue", "The background and environment must visually support the DCEE event and emotion."),
                    "cinematic_cue": row.get("cinematic_cue", f"{shot}, cinematic storytelling composition"),
                    "scene_location": loc,
                    "time_of_day": row.get("time_of_day", base_time),
                    "weather": weather,
                    "atmosphere": row.get("atmosphere", "") or f"{emotion} atmosphere",
                    "environment_details": env,
                    "supporting_cast": _string_list(row.get("supporting_cast", [])),
                    "scene_transition": transition,
                    "character_identity": protagonist_identity,
                    "character_reference_prompt": character_reference_prompt,
                    "identity_lock": {
                        "name": getattr(protagonist_profile, "name", getattr(seed, "protagonist", "protagonist")),
                        "age_group": getattr(protagonist_profile, "age_group", ""),
                        "gender": getattr(protagonist_profile, "gender", ""),
                        "face": getattr(protagonist_profile, "face", ""),
                        "hair": getattr(protagonist_profile, "hair", ""),
                        "body": getattr(protagonist_profile, "body", ""),
                        "outfit": getattr(protagonist_profile, "outfit", ""),
                        "signature_items": getattr(protagonist_profile, "signature_items", []),
                    },
                    "emotion_delta": emotion_delta_text(prev_emotion, emotion, intensity),
                    "emotion_visual_rule": emotion_rule_text(emotion),
                    "composition_rule": f"{shot}, {cam} distance, {rule['composition']}. Show the DCEE event and visual evidence.",
                    "quality_rule": QUALITY_SUFFIX,
                    "negative_prompt": NEGATIVE_PROMPT + "; no grayscale, no missing evidence, no generic scene",
                    "dcee_stage": "Event",
                    "event_causal_role": evrole,
                    "event_grounding": evground,
                    "event_emotion_causal_consistency": f"The event '{ev}' should naturally explain or intensify '{emotion}'.",
                    "shot_type": shot,
                    "camera_distance": cam,
                    "color_palette": rule["palette"],
                    "lighting_style": rule["lighting"],
                    "must_show": must_show,
                    "emotion_evidence": evidence,
                },
            )
            frames.append(frame)
            prev_emotion = emotion
            prev_world = {"scene_location": loc, "weather": weather}
        return frames

    @staticmethod
    def _profile_to_prompt(profile: CharacterProfile) -> str:
        if hasattr(profile, "to_prompt"):
            try:
                return profile.to_prompt()
            except Exception:
                pass
        return "; ".join(
            str(getattr(profile, k, ""))
            for k in ["name", "role", "face", "hair", "body", "outfit", "signature_items", "color_palette", "identity_anchor_prompt"]
            if getattr(profile, k, "")
        )

    @staticmethod
    def _get_protagonist_profile(seed: StorySeed):
        for p in getattr(seed, "character_profiles", []) or []:
            if getattr(p, "role", "") == "protagonist" or getattr(p, "name", "").lower() == getattr(seed, "protagonist", "").lower():
                return p
        raise RuntimeError("No protagonist profile available after strict seed generation.")
