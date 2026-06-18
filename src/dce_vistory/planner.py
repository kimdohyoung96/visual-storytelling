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


def _is_generic_plan_text(text: str) -> bool:
    t = (text or "").lower()
    bad = [
        "resolve the central problem",
        "discovers the problem",
        "conflict becomes visible",
        "decisive event changes the outcome",
        "object or place that starts the story",
        "an obstacle, rival, loss, or failed attempt",
    ]
    return any(x in t for x in bad)


class DCEPlanner:
    """
    Robust DCEE-CausalVerse planner.

    This planner no longer depends solely on fragile LLM JSON.
    It first attempts LLM generation, but validates the result.
    If the output is blank, generic, or non-visual, it creates a concrete DCEE plan
    from the input premise, protagonist, target ending emotion, and detected objects.
    """

    def __init__(self, llm: BaseLLM, temperature: float = 0.4, max_tokens: int = 1800):
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = min(int(max_tokens), 1800)

    def _llm_json(self, prompt: str, max_tokens: int | None = None, temperature: float | None = None):
        txt = self.llm.generate(
            SYSTEM_NARRATIVE,
            prompt,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=min(max_tokens or self.max_tokens, self.max_tokens),
        )
        if not _clean_text(txt):
            return {}
        return extract_json(txt)

    # ---------------------------------------------------------------------
    # Seed
    # ---------------------------------------------------------------------
    def build_seed(self, sample: Dict[str, Any], image_summary: ImageUnderstanding | None) -> StorySeed:
        data: Dict[str, Any] = {}
        try:
            data = self._llm_json(story_seed_prompt(sample, _to_dict(image_summary) if image_summary else None), max_tokens=1000)
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}

        fallback = self._seed_fallback(sample, image_summary)
        for k, v in fallback.items():
            if not data.get(k):
                data[k] = v

        data["objects"] = _string_list(data.get("objects", []))
        data["characters"] = _string_list(data.get("characters", []))

        profiles = self._build_character_profiles(data, sample, image_summary)
        world_context = data.get("world_context", {})
        if not isinstance(world_context, dict):
            world_context = {"raw_world_context": str(world_context)}
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
                "target_ending_emotion": sample.get("target_ending_emotion", data.get("target_ending_emotion", "sadness")),
                "genre": sample.get("genre", data.get("genre", "folk tale")),
                "style": sample.get("style", data.get("style", "full-color cinematic storybook illustration")),
                "setting": data.get("setting", getattr(image_summary, "setting", "") if image_summary else ""),
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

    def _seed_fallback(self, sample: Dict[str, Any], image_summary: ImageUnderstanding | None) -> Dict[str, Any]:
        protagonist = sample.get("protagonist", "protagonist")
        premise = sample.get("text_prompt", sample.get("story", sample.get("prompt", "")))
        target = sample.get("target_ending_emotion", "sadness")
        style = sample.get("style", "full-color cinematic storybook illustration")
        caption = getattr(image_summary, "caption", "") if image_summary else ""
        joined = f"{premise} {caption} {protagonist}".lower()

        if "woodcutter" in joined or "axe" in joined or "river" in joined:
            return {
                "protagonist": protagonist or "woodcutter",
                "setting": "a forest riverbank near a poor woodcutter's cottage",
                "objects": ["old iron axe", "wooden axe handle", "river", "golden axe", "silver axe", "fairy", "empty hands"],
                "characters": [protagonist or "woodcutter", "river fairy"],
                "mood": f"folk-tale tension leading to {target}",
                "genre": "folk tale",
                "style": style,
                "visual_symbols": {
                    "old axe": "honesty, livelihood, and loss",
                    "golden axe": "temptation",
                    "river": "irreversible consequence",
                    "empty hands": "ending emotion made visible",
                },
                "world_context": {
                    "location": "misty forest riverbank",
                    "weather_prior": "cloudy or rainy",
                    "time_of_day": "late afternoon",
                    "environment_prior": ["wet stones", "dark river water", "dense forest", "small cottage"],
                },
            }

        return {
            "protagonist": protagonist,
            "setting": sample.get("setting", "a visually coherent story world"),
            "objects": _string_list(sample.get("objects", [])),
            "characters": [protagonist],
            "mood": f"emotionally causal story leading to {target}",
            "genre": sample.get("genre", "visual story"),
            "style": style,
            "visual_symbols": {},
            "world_context": {
                "location": sample.get("setting", ""),
                "weather_prior": "",
                "time_of_day": "",
                "environment_prior": [],
            },
        }

    def _build_character_profiles(self, data, sample, image_summary):
        profiles = []
        raw = data.get("character_profiles", [])
        if isinstance(raw, dict):
            raw = [raw]
        if not isinstance(raw, list):
            raw = []

        for row in raw:
            if not isinstance(row, dict):
                continue
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
                    },
                )
            )

        protagonist = sample.get("protagonist", data.get("protagonist", "protagonist"))
        if not any(
            getattr(p, "role", "") == "protagonist" or getattr(p, "name", "").lower() == str(protagonist).lower()
            for p in profiles
        ):
            hint = getattr(image_summary, "caption", "") if image_summary else sample.get("text_prompt", "")
            signature_items = _string_list(_string_list(sample.get("signature_items", [])) + _string_list(data.get("objects", []))[:3])
            profiles.insert(
                0,
                _safe_make(
                    CharacterProfile,
                    {
                        "name": protagonist,
                        "role": "protagonist",
                        "age_group": sample.get("age_group", "adult"),
                        "gender": sample.get("gender", "unspecified"),
                        "face": f"recognizable consistent face or character features based on: {hint}",
                        "hair": "same hairstyle or head shape in every frame",
                        "body": "same body shape and proportions in every frame",
                        "outfit": sample.get("outfit", "same main outfit, same colors, same accessories in every frame"),
                        "signature_items": signature_items,
                        "color_palette": "stable protagonist color palette across all frames",
                        "identity_anchor_prompt": (
                            f"{protagonist} must look like the same character in every frame; "
                            "same face, outfit, body shape, and signature items."
                        ),
                    },
                ),
            )
        return profiles

    # ---------------------------------------------------------------------
    # Abstract
    # ---------------------------------------------------------------------
    def generate_abstract(self, seed: StorySeed) -> str:
        try:
            text = self.llm.generate(
                SYSTEM_NARRATIVE,
                story_abstract_prompt(_to_dict(seed)),
                temperature=self.temperature,
                max_tokens=800,
            ).strip()
        except Exception:
            text = ""

        if _clean_text(text) and not _is_generic_plan_text(text):
            return text

        return self._abstract_fallback(seed)

    def _abstract_fallback(self, seed: StorySeed) -> str:
        protagonist = getattr(seed, "protagonist", "") or "the protagonist"
        setting = getattr(seed, "setting", "") or "the story world"
        target = getattr(seed, "target_ending_emotion", "") or "the target ending emotion"
        premise = getattr(seed, "text_prompt", "") or "the given premise"
        return (
            f"This visual story follows {protagonist} in {setting}. "
            f"Starting from the premise '{premise}', the protagonist pursues a concrete desire, faces a central conflict, "
            f"and moves through a causally ordered event chain. Each event is designed to provide visible evidence for the "
            f"protagonist's emotional transition through actions, objects, facial cues, body posture, weather, lighting, and color. "
            f"The story culminates in the target ending emotion of {target}, making the final frame causally justified by the "
            f"preceding visual events rather than by a superficial emotion prompt."
        )

    # ---------------------------------------------------------------------
    # DCEE plan
    # ---------------------------------------------------------------------
    def generate_dce_plan(self, seed: StorySeed, abstract: str) -> DCEPlan:
        candidates = []
        n = int(getattr(seed, "raw_input", {}).get("num_dcee_candidates", 4) if isinstance(getattr(seed, "raw_input", {}), dict) else 4)
        n = max(1, min(6, n))

        try:
            data = self._llm_json(dcee_branch_plan_prompt(_to_dict(seed), abstract, num_candidates=n), max_tokens=1600, temperature=max(0.55, self.temperature))
            candidates = data.get("candidates", data if isinstance(data, list) else [])
            if isinstance(candidates, dict):
                candidates = [candidates]
            if not isinstance(candidates, list):
                candidates = []
        except Exception:
            candidates = []

        candidates = [self._normalize_candidate(c, i, seed) for i, c in enumerate(candidates) if isinstance(c, (dict, str))]
        if not candidates or self._candidates_are_generic(candidates):
            candidates = self._semantic_fallback_candidates(seed, abstract)

        selected = self._select_best_candidate(seed, abstract, candidates)
        event_chain = selected.get("event_chain", selected.get("event_spine", []))

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
                "planning_structure": "DCEE-Tree: multiple Desire-Conflict routes -> selected Event Chain -> Ending Emotion",
            },
        )
        dce_plan.event_chain = event_chain
        dce_plan.event_spine = event_chain
        dce_plan.dcee_candidates = candidates
        dce_plan.candidate_plans = candidates
        dce_plan.selected_candidate = selected
        dce_plan.planning_structure = "DCEE-Tree: multiple Desire-Conflict routes -> selected Event Chain -> Ending Emotion"
        return dce_plan

    def _candidates_are_generic(self, candidates: List[Dict[str, Any]]) -> bool:
        blob = json.dumps(candidates, ensure_ascii=False).lower()
        if _is_generic_plan_text(blob):
            return True
        # If no concrete object/evidence appears anywhere, the plan is not useful for our paper direction.
        evidence_count = 0
        for c in candidates:
            for e in c.get("event_chain", []):
                evidence_count += len(_string_list(e.get("evidence_objects", [])))
                evidence_count += len(_string_list(e.get("key_objects", [])))
        return evidence_count == 0

    def _semantic_fallback_candidates(self, seed: StorySeed, abstract: str) -> List[Dict[str, Any]]:
        protagonist = getattr(seed, "protagonist", "protagonist") or "protagonist"
        target = _ending_synonym(getattr(seed, "target_ending_emotion", "sadness"))
        premise = f"{getattr(seed, 'text_prompt', '')} {getattr(seed, 'setting', '')} {' '.join(_string_list(getattr(seed, 'objects', [])))}".lower()

        if "woodcutter" in premise or "axe" in premise or "river" in premise:
            return [
                self._normalize_candidate(
                    {
                        "candidate_id": "route_honesty_loss",
                        "protagonist": protagonist,
                        "desire": f"{protagonist} wants to recover the old iron axe because it is his livelihood.",
                        "fear": "He fears returning home empty-handed and failing his family.",
                        "misbelief": "He briefly believes that taking a more valuable axe could solve his poverty.",
                        "obstacle": "The axe has fallen into the river and a fairy tests his honesty with golden and silver axes.",
                        "conflict": "The need to survive conflicts with the moral pressure to remain honest.",
                        "event_chain": [
                            {
                                "event_id": "e1",
                                "event": f"{protagonist} drops the old iron axe into the dark river.",
                                "causal_role": "introduces desire and loss",
                                "visual_grounding": "the old axe slips from his hands, splashes into the river, and ripples spread across the water",
                                "emotion_effect": "shock and worry",
                                "key_objects": ["old iron axe", "river", "empty hands"],
                                "evidence_objects": ["river ripples", "falling axe", "empty hands"],
                            },
                            {
                                "event_id": "e2",
                                "event": "A river fairy appears and offers a shining golden axe.",
                                "causal_role": "externalizes temptation",
                                "visual_grounding": "the fairy holds a golden axe while the woodcutter hesitates at the riverbank",
                                "emotion_effect": "temptation and anxiety",
                                "key_objects": ["river fairy", "golden axe", "riverbank"],
                                "evidence_objects": ["golden axe glow", "hesitant hands", "tense posture"],
                            },
                            {
                                "event_id": "e3",
                                "event": f"{protagonist} falsely reaches for the golden axe instead of his old axe.",
                                "causal_role": "turning point caused by moral failure",
                                "visual_grounding": "his hand reaches toward the golden axe while his face shows guilt and fear",
                                "emotion_effect": "guilt",
                                "key_objects": ["golden axe", "woodcutter's hand", "fairy's gaze"],
                                "evidence_objects": ["reaching hand", "guilty expression", "fairy's disappointed gaze"],
                            },
                            {
                                "event_id": "e4",
                                "event": "The fairy withdraws the axes and disappears into the river mist.",
                                "causal_role": "consequence of the decisive event",
                                "visual_grounding": "the fairy fades away with the golden and old axes, leaving only mist and disturbed water",
                                "emotion_effect": "panic and regret",
                                "key_objects": ["fairy", "golden axe", "old axe", "river mist"],
                                "evidence_objects": ["vanishing fairy", "empty riverbank", "mist"],
                            },
                            {
                                "event_id": "e5",
                                "event": f"{protagonist} kneels at the riverbank with empty hands and no axe.",
                                "causal_role": "ending visual evidence",
                                "visual_grounding": "he kneels alone beside the river, staring at his empty hands as rain darkens the forest",
                                "emotion_effect": target,
                                "key_objects": ["empty hands", "riverbank", "rain", "dark forest"],
                                "evidence_objects": ["empty hands", "kneeling body", "rain", "lost axe absence"],
                            },
                        ],
                        "turning_point": "The woodcutter reaches for the golden axe, making the sad ending causally inevitable.",
                        "ending_emotion": target,
                        "target_ending_emotion": target,
                        "ending_state": f"{protagonist} ends in {target}, visually explained by the lost axe, empty hands, and deserted riverbank.",
                        "moral_or_theme": "The ending emotion is caused by the visible consequence of a morally compromised choice.",
                    },
                    0,
                    seed,
                ),
                self._normalize_candidate(
                    {
                        "candidate_id": "route_desperation_accident",
                        "protagonist": protagonist,
                        "desire": f"{protagonist} wants to retrieve the axe before nightfall.",
                        "fear": "He fears the river will carry the axe away forever.",
                        "misbelief": "He believes rushing into the river will solve the problem.",
                        "obstacle": "The river current is too strong and the weather worsens.",
                        "conflict": "Desperation pushes him into dangerous action against the physical obstacle of the river.",
                        "event_chain": [
                            {
                                "event_id": "e1",
                                "event": "The axe falls from the woodcutter's grip into the river.",
                                "causal_role": "introduces loss",
                                "visual_grounding": "the axe is half-submerged in the rushing water",
                                "emotion_effect": "fear",
                                "key_objects": ["old axe", "river"],
                                "evidence_objects": ["splash", "floating axe handle"],
                            },
                            {
                                "event_id": "e2",
                                "event": "The woodcutter steps into the river and loses his balance.",
                                "causal_role": "escalates physical conflict",
                                "visual_grounding": "water rises around his boots as he reaches toward the axe",
                                "emotion_effect": "panic",
                                "key_objects": ["boots", "river current", "axe handle"],
                                "evidence_objects": ["slippery stones", "outstretched arm"],
                            },
                            {
                                "event_id": "e3",
                                "event": "The axe disappears downstream beyond his reach.",
                                "causal_role": "turning point of irreversible loss",
                                "visual_grounding": "the axe handle vanishes into dark water while he falls to his knees",
                                "emotion_effect": target,
                                "key_objects": ["dark water", "axe handle"],
                                "evidence_objects": ["distant floating axe", "collapsed posture"],
                            },
                        ],
                        "turning_point": "The axe disappears downstream.",
                        "ending_emotion": target,
                        "target_ending_emotion": target,
                        "ending_state": f"{protagonist} ends in {target} after the visual loss becomes irreversible.",
                    },
                    1,
                    seed,
                ),
            ]

        objects = _string_list(getattr(seed, "objects", []))
        central_object = objects[0] if objects else "a key object"
        return [
            self._normalize_candidate(
                {
                    "candidate_id": "route_general_causal",
                    "protagonist": protagonist,
                    "desire": f"{protagonist} wants to obtain or protect {central_object}.",
                    "fear": f"{protagonist} fears losing {central_object} or failing the people who depend on it.",
                    "misbelief": "The protagonist believes the problem can be solved without facing the deeper conflict.",
                    "obstacle": "A visible external obstacle and an internal emotional pressure block the desire.",
                    "conflict": f"The protagonist's desire for {central_object} is blocked by a concrete obstacle and a moral or emotional choice.",
                    "event_chain": [
                        {
                            "event_id": "e1",
                            "event": f"{protagonist} encounters {central_object} and realizes what is at stake.",
                            "causal_role": "introduces desire",
                            "visual_grounding": f"{central_object} is clearly visible near the protagonist",
                            "emotion_effect": "hope or concern",
                            "key_objects": [central_object],
                            "evidence_objects": [central_object, "protagonist's focused gaze"],
                        },
                        {
                            "event_id": "e2",
                            "event": f"An obstacle separates {protagonist} from {central_object}.",
                            "causal_role": "escalates conflict",
                            "visual_grounding": f"a visible barrier, loss, rival, or distance blocks access to {central_object}",
                            "emotion_effect": "anxiety",
                            "key_objects": [central_object, "obstacle"],
                            "evidence_objects": ["visible obstacle", "tense body posture"],
                        },
                        {
                            "event_id": "e3",
                            "event": f"{protagonist} makes a decisive choice that changes the fate of {central_object}.",
                            "causal_role": "turning point",
                            "visual_grounding": "the choice is visible through action, gesture, and object placement",
                            "emotion_effect": target,
                            "key_objects": [central_object],
                            "evidence_objects": ["decisive gesture", central_object, "expressive face"],
                        },
                    ],
                    "turning_point": "The protagonist makes a visible choice that determines the ending emotion.",
                    "ending_emotion": target,
                    "target_ending_emotion": target,
                    "ending_state": f"The protagonist ends in {target}, visually explained by the state of {central_object}.",
                },
                0,
                seed,
            )
        ]

    def _normalize_candidate(self, c, idx, seed):
        if not isinstance(c, dict):
            c = {"candidate_id": f"c{idx + 1}", "event_chain": [str(c)]}
        c.setdefault("candidate_id", f"c{idx + 1}")
        c.setdefault("protagonist", getattr(seed, "protagonist", ""))
        c.setdefault("target_ending_emotion", c.get("ending_emotion", getattr(seed, "target_ending_emotion", "")))
        c.setdefault("ending_emotion", c.get("target_ending_emotion", getattr(seed, "target_ending_emotion", "")))

        chain = c.get("event_chain", c.get("event_spine", []))
        if isinstance(chain, dict):
            chain = chain.get("events", [chain])
        if not isinstance(chain, list):
            chain = [str(chain)]

        norm = []
        for j, e in enumerate(chain):
            if not isinstance(e, dict):
                e = {"event": str(e)}
            e.setdefault("event_id", f"e{j + 1}")
            e.setdefault("causal_role", e.get("role", "causes or intensifies emotion"))
            e.setdefault("visual_grounding", e.get("visual_evidence", e.get("description", e.get("event", ""))))
            e.setdefault("emotion_effect", c.get("target_ending_emotion", getattr(seed, "target_ending_emotion", "")) if j == len(chain) - 1 else "emotional transition")
            e["key_objects"] = _string_list(e.get("key_objects", []))
            e["evidence_objects"] = _string_list(e.get("evidence_objects", e.get("visual_evidence_objects", [])))
            norm.append(e)

        c["event_chain"] = norm
        c["event_spine"] = norm
        return c

    def _select_best_candidate(self, seed, abstract, candidates):
        try:
            judge = self._llm_json(dcee_candidate_selection_prompt(_to_dict(seed), abstract, candidates), max_tokens=900, temperature=0.0)
            sid = str(judge.get("selected_candidate_id", ""))
            for c in candidates:
                if str(c.get("candidate_id")) == sid:
                    c["selection_scores"] = judge.get("scores", [])
                    c["selection_reason"] = judge.get("reason", "")
                    return c
        except Exception:
            pass

        def score(c):
            chain = c.get("event_chain", [])
            visual = sum(1 for e in chain if _clean_text(e.get("visual_grounding")))
            evidence = sum(len(_string_list(e.get("evidence_objects", []))) + len(_string_list(e.get("key_objects", []))) for e in chain)
            turning = 2 if c.get("turning_point") else 0
            conflict = 2 if c.get("conflict") else 0
            return len(chain) * 2 + visual + evidence + turning + conflict

        best = max(candidates, key=score)
        best["selection_reason"] = "heuristic selected for concrete causal events and visual evidence coverage"
        return best

    # ---------------------------------------------------------------------
    # Emotion arc and storyboard
    # ---------------------------------------------------------------------
    def generate_emotion_arc(self, seed: StorySeed, abstract: str, dce_plan: DCEPlan, num_frames: int) -> EmotionArc:
        try:
            data = self._llm_json(emotion_arc_prompt(_to_dict(seed), abstract, _to_dict(dce_plan), num_frames), max_tokens=1000)
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}

        states = data.get("states", [])
        intensities = data.get("intensities", [])

        target = _ending_synonym(getattr(dce_plan, "target_ending_emotion", getattr(seed, "target_ending_emotion", "sadness")))
        if len(states) != num_frames or not states:
            if target in ["sadness", "regret"]:
                base = ["concern", "temptation", "anxiety", "guilt", "regret", target]
            elif target in ["joy", "relief"]:
                base = ["hope", "concern", "determination", "tension", "relief", target]
            elif target == "fear":
                base = ["curiosity", "unease", "anxiety", "shock", "panic", "fear"]
            else:
                base = ["neutral", "concern", "tension", "decision", "consequence", target]
            states = (base + [target] * num_frames)[:num_frames]

        if len(intensities) != num_frames or not intensities:
            intensities = ([2, 3, 4, 5, 4, 5] + [5] * num_frames)[:num_frames]

        states[-1] = target
        return _safe_make(
            EmotionArc,
            {
                "states": states,
                "intensities": [max(1, min(5, int(x))) for x in intensities],
                "rationale": data.get("rationale", "DCEE fallback emotion arc from desire-conflict-event causality."),
                "valence_curve": data.get("valence_curve", []),
                "arousal_curve": data.get("arousal_curve", []),
                "suspense_curve": data.get("suspense_curve", []),
            },
        )

    def generate_storyboard(self, seed: StorySeed, abstract: str, dce_plan: DCEPlan, emotion_arc: EmotionArc) -> List[StoryboardFrame]:
        states = getattr(emotion_arc, "states", [])
        num_frames = len(states) or int(getattr(seed, "raw_input", {}).get("num_frames", 6) if isinstance(getattr(seed, "raw_input", {}), dict) else 6)

        rows = []
        try:
            rows = self._llm_json(storyboard_prompt(_to_dict(seed), abstract, _to_dict(dce_plan), _to_dict(emotion_arc), num_frames), max_tokens=1600)
            if isinstance(rows, dict):
                rows = rows.get("storyboard", rows.get("frames", rows))
            if not isinstance(rows, list):
                rows = []
        except Exception:
            rows = []

        if not rows or self._storyboard_is_generic(rows):
            rows = self._storyboard_from_event_chain(seed, dce_plan, emotion_arc, num_frames)

        try:
            crows = self._llm_json(canonicalize_storyboard_prompt(_to_dict(seed), _to_dict(dce_plan), rows), max_tokens=1400, temperature=0.0)
            if isinstance(crows, dict):
                crows = crows.get("storyboard", crows.get("frames", crows))
            if isinstance(crows, list) and not self._storyboard_is_generic(crows):
                rows = crows
        except Exception:
            pass

        return self._postprocess_storyboard(rows, seed, dce_plan, emotion_arc)

    def _storyboard_is_generic(self, rows: Any) -> bool:
        blob = json.dumps(rows, ensure_ascii=False).lower() if isinstance(rows, (list, dict)) else str(rows).lower()
        return _is_generic_plan_text(blob)

    def _storyboard_from_event_chain(self, seed, dce_plan, emotion_arc, num_frames):
        chain = getattr(dce_plan, "event_chain", getattr(dce_plan, "event_spine", [])) or []
        if not chain:
            chain = self._semantic_fallback_candidates(seed, "")[0]["event_chain"]

        rows = []
        for i in range(num_frames):
            # Spread event chain across all frames.
            ev = chain[min(int(i * len(chain) / max(1, num_frames)), len(chain) - 1)]
            if not isinstance(ev, dict):
                ev = {"event": str(ev), "visual_grounding": str(ev)}
            emotion = getattr(emotion_arc, "states", ["neutral"] * num_frames)[min(i, len(getattr(emotion_arc, "states", [])) - 1)]
            intensity = getattr(emotion_arc, "intensities", [3] * num_frames)[min(i, len(getattr(emotion_arc, "intensities", [])) - 1)]
            rows.append(
                {
                    "frame_id": i + 1,
                    "caption": ev.get("visual_grounding", ev.get("event", "")),
                    "narrative_function": ev.get("causal_role", "DCEE event progression"),
                    "event": ev.get("event", ""),
                    "event_causal_role": ev.get("causal_role", ""),
                    "event_grounding": ev.get("visual_grounding", ""),
                    "emotion": emotion,
                    "emotion_intensity": intensity,
                    "key_objects": ev.get("key_objects", []),
                    "evidence_objects": ev.get("evidence_objects", []),
                    "visual_focus": ev.get("visual_grounding", ""),
                    "protagonist_state": f"The protagonist visibly experiences {emotion}.",
                    "desire_link": getattr(dce_plan, "desire", ""),
                    "conflict_level": min(5, 1 + i),
                    "scene_location": getattr(seed, "setting", ""),
                    "must_show": _string_list(ev.get("key_objects", []) + ev.get("evidence_objects", []) + [ev.get("event", ""), ev.get("visual_grounding", "")]),
                }
            )
        return rows

    def _postprocess_storyboard(self, rows, seed, dce_plan, emotion_arc):
        protagonist_profile = self._get_protagonist_profile(seed)
        protagonist_identity = self._profile_to_prompt(protagonist_profile) if protagonist_profile else getattr(seed, "protagonist", "protagonist")
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
                row = {"event": str(row)}

            emotion = row.get("emotion", states[min(idx, len(states) - 1)] if states else "neutral")
            intensity = max(1, min(5, int(row.get("emotion_intensity", intensities[min(idx, len(intensities) - 1)] if intensities else 3))))
            rule = get_emotion_rule(emotion)

            linked = chain[min(idx, len(chain) - 1)] if chain else {}
            if not isinstance(linked, dict):
                linked = {"event": str(linked), "visual_grounding": str(linked), "causal_role": "causes emotion"}

            ev = row.get("event") or linked.get("event") or linked.get("description", "")
            evrole = row.get("event_causal_role") or linked.get("causal_role", "causes or intensifies the frame emotion")
            evground = row.get("event_grounding") or linked.get("visual_grounding", ev)

            evidence_objects = _string_list(row.get("evidence_objects", linked.get("evidence_objects", [])))
            key_objects = _string_list(row.get("key_objects", linked.get("key_objects", [])))

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
            nf = row.get("narrative_function", "event progression")
            shot = choose_shot_type(idx, len(rows), nf)
            cam = choose_camera_distance(shot)

            evidence = _string_list(
                [ev, evground]
                + key_objects[:3]
                + evidence_objects[:3]
                + ([f"evidence of desire: {row.get('desire_link')}"] if row.get("desire_link") else [])
            )
            must_show = _string_list(
                key_objects[:3]
                + evidence_objects[:3]
                + [
                    ev,
                    evground,
                    f"facial evidence of {emotion}",
                    f"body evidence of {emotion}",
                    "the current DCEE event",
                    "the visual cause of the emotion",
                    "full-color emotional lighting",
                ]
            )
            neg = NEGATIVE_PROMPT + ("; " + getattr(protagonist_profile, "negative_identity_prompt", "") if protagonist_profile else "")

            frame = _safe_make(
                StoryboardFrame,
                {
                    "frame_id": int(row.get("frame_id", idx + 1)),
                    "caption": row.get("caption", evground),
                    "narrative_function": nf,
                    "event": ev,
                    "protagonist_state": row.get("protagonist_state", f"The protagonist visibly experiences {emotion}."),
                    "desire_link": row.get("desire_link", getattr(dce_plan, "desire", "")),
                    "conflict_level": int(row.get("conflict_level", min(5, idx + 1))),
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
                    "emotion_delta": emotion_delta_text(prev_emotion, emotion, intensity),
                    "emotion_visual_rule": emotion_rule_text(emotion),
                    "composition_rule": f"{shot}, {cam} distance, {rule['composition']}. Show the DCEE event and visual evidence.",
                    "quality_rule": QUALITY_SUFFIX,
                    "negative_prompt": neg,
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
        cps = getattr(seed, "character_profiles", []) or []
        return cps[0] if cps else None
