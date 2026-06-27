from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from typing import Any, Dict, List
import json
import re

from .llm import BaseLLM
from .schema import StorySeed, DCEPlan, EmotionArc, StoryboardFrame, ImageUnderstanding, CharacterProfile
from .prompts import (
    SYSTEM_NARRATIVE,
    story_seed_prompt,
    story_abstract_prompt,
    dcee_plan_prompt,
    emotion_arc_prompt,
    next_story_sentence_prompt,
    get_emotion_rule,
    emotion_rule_text,
    choose_shot_type,
    choose_camera_distance,
)
from .utils import extract_json

_SPECIAL_STORY_TERMS = [
    "woodcutter", "lumberjack", "axe", "fairy", "golden axe", "silver axe",
    "friend", "friends", "animal friend", "animal friends", "wild animal friend", "wild animal friends",
    "helper", "helpers", "villager", "villagers", "human", "hunter", "traveler", "stranger",
    "rabbit", "fox", "deer", "bird", "squirrel", "monkey", "another panda", "other panda",
]


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


def _clean_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    elif isinstance(value, dict):
        value = list(value.values())
    elif not isinstance(value, list):
        value = [value]
    out: List[str] = []
    for item in value:
        if item is None:
            continue
        text = _clean_text(item)
        if text and text not in out:
            out.append(text)
    return out


def _unique(items: List[str]) -> List[str]:
    out: List[str] = []
    for item in items:
        t = _clean_text(item)
        if t and t not in out:
            out.append(t)
    return out


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


def _infer_default_emotion_steps(target: str, n: int) -> List[str]:
    target = _ending_synonym(target)
    if target in {"joy", "relief", "hope"}:
        base = ["curiosity", "hope", "doubt", "determination", "relief", target]
    elif target in {"sadness", "regret"}:
        base = ["curiosity", "hope", "doubt", "anxiety", "sadness", target]
    elif target in {"fear", "anger"}:
        base = ["curiosity", "doubt", "anxiety", target, target, target]
    else:
        base = ["curiosity", "doubt", "determination", "tension", "reflection", target]
    if n <= len(base):
        return base[:n]
    while len(base) < n:
        base.insert(-1, base[-2])
    return base[:n]


def _contains_forbidden(blob: Any, forbidden_terms: List[str]) -> bool:
    text = json.dumps(blob, ensure_ascii=False).lower() if isinstance(blob, (dict, list)) else str(blob).lower()
    return any(term.lower() in text for term in forbidden_terms)


def _replace_forbidden_text(text: str, forbidden_terms: List[str], protagonist: str) -> str:
    out = str(text or "")
    for term in forbidden_terms:
        if term.lower() in {"woodcutter", "lumberjack"}:
            out = re.sub(rf"\b{re.escape(term)}\b", protagonist, out, flags=re.IGNORECASE)
        else:
            out = re.sub(rf"\b{re.escape(term)}\b", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out).strip(" ,;:.")
    return out


def _sanitize_nested(value: Any, forbidden_terms: List[str], protagonist: str):
    if isinstance(value, dict):
        return {k: _sanitize_nested(v, forbidden_terms, protagonist) for k, v in value.items()}
    if isinstance(value, list):
        cleaned = []
        for v in value:
            cv = _sanitize_nested(v, forbidden_terms, protagonist)
            if isinstance(cv, str):
                if _clean_text(cv):
                    cleaned.append(_clean_text(cv))
            elif cv not in [None, {}, []]:
                cleaned.append(cv)
        return cleaned
    if isinstance(value, str):
        return _replace_forbidden_text(value, forbidden_terms, protagonist)
    return value


_AGENT_WORDS = {
    "friend", "friends", "animal friend", "animal friends", "wild animal friend", "wild animal friends",
    "helper", "helpers", "villager", "villagers", "human", "hunter", "woodcutter", "lumberjack",
    "fairy", "traveler", "stranger", "rabbit", "fox", "deer", "bird", "squirrel", "monkey",
    "another panda", "other panda",
}


def _is_agent_like(text: Any, protagonist: str) -> bool:
    s = _clean_text(text).lower()
    if not s:
        return False
    p = _clean_text(protagonist).lower()
    if p and s == p:
        return False
    return any(w in s for w in _AGENT_WORDS)


def _filter_protagonist_only_objects(items: Any, protagonist: str) -> List[str]:
    """Keep props/background objects, remove secondary characters/agents."""
    out: List[str] = []
    for item in _string_list(items):
        low = item.lower()
        if _is_agent_like(low, protagonist):
            continue
        if item not in out:
            out.append(item)
    return out


def _force_protagonist_only_step(data: Dict[str, Any], protagonist: str) -> Dict[str, Any]:
    data = dict(data or {})
    data["subject"] = protagonist
    data["supporting_cast"] = []
    data["characters"] = [protagonist]

    for key in ["sentence", "action", "visible_cause", "continuity_notes"]:
        if key in data:
            data[key] = _replace_forbidden_text(str(data.get(key, "")), list(_AGENT_WORDS), protagonist)

    data["required_objects"] = _filter_protagonist_only_objects(data.get("required_objects", []), protagonist)
    data["background_elements"] = _filter_protagonist_only_objects(data.get("background_elements", []), protagonist)
    if not data["required_objects"]:
        data["required_objects"] = _filter_protagonist_only_objects([
            protagonist,
            data.get("object", ""),
            data.get("location", ""),
            "foreground prop",
        ], protagonist)
    data["object"] = _clean_text(data.get("object", ""))
    if _is_agent_like(data["object"], protagonist):
        data["object"] = data["required_objects"][0] if data["required_objects"] else ""
    return data



def _extract_seed_visual_terms(seed: Any, limit: int = 8) -> List[str]:
    """Collect grounded props/background terms from seed without adding new agents."""
    terms: List[str] = []
    for attr in ["objects", "setting", "mood"]:
        terms.extend(_string_list(getattr(seed, attr, "")))
    wc = getattr(seed, "world_context", {}) or {}
    if isinstance(wc, dict):
        for key in ["setting", "background", "environment", "objects", "landmarks", "weather"]:
            terms.extend(_string_list(wc.get(key, [])))
    raw = getattr(seed, "raw_input", {}) or {}
    if isinstance(raw, dict):
        for key in ["objects", "setting", "background_elements", "signature_items"]:
            terms.extend(_string_list(raw.get(key, [])))
    return _unique([x for x in terms if _clean_text(x)])[:limit]


def _derive_required_objects(data: Dict[str, Any], seed: Any, protagonist: str) -> List[str]:
    """
    V20 story-locked repair:
    reconstruct only the minimum grounded set of props/background objects needed for the current sentence.
    """
    cleaned = _story_locked_visual_inventory(data, seed, protagonist)
    if not cleaned:
        cleaned = ["simple background"]
    return cleaned[:4]


def _derive_background_elements(data: Dict[str, Any], seed: Any, protagonist: str) -> List[str]:
    items: List[str] = []
    items.extend(_string_list(data.get("location", "")))
    items.extend(_string_list(data.get("background_elements", []))[:2])
    if not items:
        items.extend(_extract_seed_visual_terms(seed, limit=2))
    cleaned = _filter_protagonist_only_objects(items, protagonist)
    cleaned = [x for x in _unique(cleaned) if _clean_text(x) and x != protagonist]
    if not cleaned:
        cleaned = ["simple grounded background"]
    return cleaned[:3]


def _story_locked_visual_inventory(data: Dict[str, Any], seed: Any, protagonist: str) -> List[str]:
    """Return the smallest grounded set of visual items needed for the current sentence."""
    items: List[str] = []
    items.extend(_string_list(data.get("object", "")))
    items.extend(_string_list(data.get("location", "")))
    items.extend(_string_list(data.get("background_elements", []))[:2])
    items.extend(_string_list(data.get("visible_cause", ""))[:1])
    if len([x for x in items if _clean_text(x)]) < 2:
        items.extend(_extract_seed_visual_terms(seed, limit=2))
    cleaned = _filter_protagonist_only_objects(items, protagonist)
    cleaned = [x for x in _unique(cleaned) if _clean_text(x) and _clean_text(x).lower() != _clean_text(protagonist).lower()]
    return cleaned[:4]


def _grounded_terms(sample: Dict[str, Any], image_summary: ImageUnderstanding | None) -> List[str]:
    terms: List[str] = []
    for key in ["protagonist", "text_prompt", "style", "genre", "setting", "outfit", "age_group", "gender"]:
        terms.extend(_string_list(sample.get(key)))
    for key in ["objects", "characters", "signature_items"]:
        terms.extend(_string_list(sample.get(key, [])))
    if image_summary is not None:
        for key in ["caption", "setting", "mood", "inferred_plot_hint"]:
            terms.extend(_string_list(getattr(image_summary, key, "")))
        for key in ["objects", "characters"]:
            terms.extend(_string_list(getattr(image_summary, key, [])))
    return _unique(terms)


def _forbidden_terms(sample: Dict[str, Any], image_summary: ImageUnderstanding | None) -> List[str]:
    grounded_blob = " ".join(_grounded_terms(sample, image_summary)).lower()
    banned: List[str] = []
    for term in _SPECIAL_STORY_TERMS:
        if term.lower() not in grounded_blob:
            banned.append(term)
    return banned


def _ensure_identity_fields(sample: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    profile = dict(profile or {})
    protagonist = sample.get("protagonist", profile.get("name", "protagonist"))
    age_group = sample.get("age_group", profile.get("age_group", "unspecified"))
    gender = sample.get("gender", profile.get("gender", "unspecified"))
    outfit = sample.get("outfit", profile.get("outfit", "same main outfit and same colors in every frame"))
    profile.setdefault("name", protagonist)
    profile.setdefault("role", "protagonist")
    profile.setdefault("age_group", age_group)
    profile.setdefault("gender", gender)
    profile.setdefault("outfit", outfit)
    profile.setdefault("signature_items", sample.get("signature_items", []))
    profile.setdefault("face", sample.get("face", f"consistent recognizable {age_group} protagonist face"))
    profile.setdefault("hair", sample.get("hair", "same head shape and same hair or fur pattern in every frame"))
    profile.setdefault("body", sample.get("body", "same body proportions, same apparent age, same species in every frame"))
    profile.setdefault("color_palette", sample.get("protagonist_color_palette", "stable protagonist color palette"))
    profile.setdefault(
        "identity_anchor_prompt",
        (
            f"{profile['name']} is the SAME protagonist in every frame; same age group ({age_group}), same gender presentation ({gender}), "
            f"same face, same hair or fur pattern, same body proportions, same outfit ({outfit}), "
            f"same signature items ({profile.get('signature_items', [])}). Only facial expression, body pose, background, and action may change."
        ),
    )
    profile.setdefault(
        "negative_identity_prompt",
        "different person, different species, child version, baby version, older version, gender changed, different face, different hairstyle, different body shape, different outfit",
    )
    return profile


class DCEPlanner:
    """Grounded incremental DCEE planner."""

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

    def _llm_json_strict(self, prompt: str, stage: str, max_tokens: int | None = None, temperature: float | None = None, validate=None, repair_hint: str = ""):
        errors = []
        for attempt in range(2):
            current_prompt = prompt if attempt == 0 else (prompt + "\n\nYour previous response was invalid. Return valid JSON only. " + repair_hint)
            try:
                text = self._llm_text(current_prompt, max_tokens=max_tokens, temperature=temperature)
                data = extract_json(text)
                if data in [None, {}, []]:
                    raise ValueError("Parsed JSON is empty.")
                if validate is not None:
                    validate(data)
                return data
            except Exception as e:
                errors.append(f"attempt {attempt+1}: {type(e).__name__}: {e}")
        raise RuntimeError(f"Strict LLM JSON generation failed at stage={stage}. " + " | ".join(errors))

    def build_seed(self, sample: Dict[str, Any], image_summary: ImageUnderstanding | None) -> StorySeed:
        forbidden = _forbidden_terms(sample, image_summary)
        prompt = story_seed_prompt(sample, _to_dict(image_summary) if image_summary else None, forbidden, protagonist_only=True)

        def _validate(data):
            for key in ["setting", "objects", "characters", "mood", "visual_symbols", "world_context", "character_profiles"]:
                if key not in data:
                    raise ValueError(f"missing key: {key}")
            if not _string_list(data.get("objects")):
                raise ValueError("objects cannot be empty")
            if _contains_forbidden(data, forbidden) or _contains_forbidden(data, list(_AGENT_WORDS)):
                raise ValueError(f"contains forbidden ungrounded agents: {forbidden}")

        data = self._llm_json_strict(prompt, "build_seed", max_tokens=1200, validate=_validate, repair_hint="Use only grounded entities from the input/image summary.")
        protagonist = sample.get("protagonist", "protagonist")
        data = _sanitize_nested(data, forbidden + list(_AGENT_WORDS), protagonist)
        data["characters"] = [protagonist]
        data["objects"] = _filter_protagonist_only_objects(data.get("objects", []), protagonist)

        profiles = data.get("character_profiles", []) or []
        if not profiles:
            profiles = [{"name": protagonist, "role": "protagonist"}]
        profile_dicts = [_ensure_identity_fields(sample, p if isinstance(p, dict) else {"name": protagonist}) for p in profiles]
        try:
            character_profiles = [_safe_make(CharacterProfile, p) for p in profile_dicts]
        except Exception:
            character_profiles = profile_dicts

        kwargs = dict(
            image_summary=image_summary,
            text_prompt=sample.get("text_prompt", ""),
            protagonist=protagonist,
            target_ending_emotion=sample.get("target_ending_emotion", ""),
            genre=sample.get("genre", "visual storytelling"),
            style=sample.get("style", "full-color cinematic storybook illustration"),
            setting=data.get("setting", image_summary.setting if image_summary else ""),
            objects=_string_list(data.get("objects", [])),
            characters=_string_list(data.get("characters", [])),
            mood=data.get("mood", image_summary.mood if image_summary else ""),
            visual_symbols=data.get("visual_symbols", {}),
            raw_input=sample,
        )
        seed = _safe_make(StorySeed, kwargs)
        for extra_key, extra_val in {
            "world_context": data.get("world_context", {}),
            "character_profiles": character_profiles,
            "source_image_path": sample.get("image_path", ""),
            "forbidden_ungrounded_entities": forbidden,
        }.items():
            try:
                setattr(seed, extra_key, extra_val)
            except Exception:
                pass
        return seed

    def generate_abstract(self, seed: StorySeed) -> str:
        forbidden = getattr(seed, "forbidden_ungrounded_entities", []) or []
        text = self._llm_text(story_abstract_prompt(_to_dict(seed), forbidden, protagonist_only=True), max_tokens=500, temperature=0.35)
        text = _replace_forbidden_text(text, forbidden, getattr(seed, "protagonist", "protagonist"))
        if not _clean_text(text):
            raise RuntimeError("Abstract is empty.")
        return text

    def generate_dce_plan(self, seed: StorySeed, abstract: str) -> DCEPlan:
        forbidden = getattr(seed, "forbidden_ungrounded_entities", []) or []
        prompt = dcee_plan_prompt(_to_dict(seed), abstract, forbidden, protagonist_only=True)

        def _validate(data):
            for key in ["desire", "conflict", "target_ending_emotion"]:
                if not _clean_text(data.get(key)):
                    raise ValueError(f"missing key: {key}")
            ev = data.get("event_chain", data.get("event_spine", []))
            if not ev:
                raise ValueError("event_chain is empty")
            if _contains_forbidden(data, forbidden) or _contains_forbidden(data, list(_AGENT_WORDS)):
                raise ValueError("plan contains forbidden ungrounded agents")

        data = self._llm_json_strict(prompt, "generate_dce_plan", max_tokens=1200, validate=_validate, repair_hint="Return one grounded DCEE plan using only the seed entities.")
        protagonist = getattr(seed, "protagonist", "protagonist")
        data = _sanitize_nested(data, forbidden + list(_AGENT_WORDS), protagonist)
        event_chain = data.get("event_chain", data.get("event_spine", []))
        if isinstance(event_chain, str):
            event_chain = [event_chain]
        kwargs = {
            "protagonist": protagonist,
            "desire": data.get("desire", ""),
            "fear": data.get("fear", ""),
            "misbelief": data.get("misbelief", ""),
            "obstacle": data.get("obstacle", ""),
            "conflict": data.get("conflict", ""),
            "event_spine": [e.get("event", e) if isinstance(e, dict) else e for e in event_chain],
            "turning_point": data.get("turning_point", ""),
            "target_ending_emotion": data.get("target_ending_emotion", getattr(seed, "target_ending_emotion", "")),
            "ending_state": data.get("ending_state", ""),
            "moral_or_theme": data.get("moral_or_theme", ""),
        }
        plan = _safe_make(DCEPlan, kwargs)
        for k, v in {
            "event_chain": event_chain,
            "planning_structure": data.get("planning_structure", "DCEE: Desire-Conflict-Event-Ending Emotion"),
        }.items():
            try:
                setattr(plan, k, v)
            except Exception:
                pass
        return plan

    def generate_emotion_arc(self, seed: StorySeed, abstract: str, dce_plan: DCEPlan, num_frames: int) -> EmotionArc:
        prompt = emotion_arc_prompt(_to_dict(seed), abstract, _to_dict(dce_plan), num_frames)
        try:
            data = self._llm_json_strict(prompt, "generate_emotion_arc", max_tokens=700)
            states = _string_list(data.get("states", []))
            intensities = [int(x) for x in (data.get("intensities", []) or [])]
        except Exception:
            data = {}
            states = []
            intensities = []
        if len(states) != num_frames:
            states = _infer_default_emotion_steps(getattr(dce_plan, "target_ending_emotion", getattr(seed, "target_ending_emotion", "")), num_frames)
        if len(intensities) != num_frames:
            intensities = [1, 2, 2, 3, 4, 5][:num_frames]
            if len(intensities) < num_frames:
                intensities += [min(5, intensities[-1] + 1)] * (num_frames - len(intensities))
        states[-1] = _ending_synonym(getattr(dce_plan, "target_ending_emotion", getattr(seed, "target_ending_emotion", states[-1])))
        return _safe_make(EmotionArc, {
            "states": states,
            "intensities": intensities,
            "rationale": data.get("rationale", "Emotion grows along the visible event chain toward the target ending emotion."),
        })

    def generate_story_step(self, seed: StorySeed, abstract: str, dce_plan: DCEPlan, emotion_arc: EmotionArc, story_so_far: List[Dict[str, Any]], previous_frame: StoryboardFrame | None, frame_index: int, num_frames: int) -> Dict[str, Any]:
        forbidden = getattr(seed, "forbidden_ungrounded_entities", []) or []
        prev_dict = _to_dict(previous_frame) if previous_frame is not None else None
        prompt = next_story_sentence_prompt(_to_dict(seed), _to_dict(dce_plan), _to_dict(emotion_arc), story_so_far, prev_dict, frame_index, num_frames, forbidden, protagonist_only=True)

        def _validate(data):
            for key in ["sentence", "action", "location", "emotion", "visible_cause"]:
                if key not in data:
                    raise ValueError(f"missing key: {key}")
            if "required_objects" not in data:
                data["required_objects"] = []
            if "background_elements" not in data:
                data["background_elements"] = []
            if not _clean_text(data.get("sentence")):
                raise ValueError("sentence is empty")
            if _contains_forbidden(data, forbidden) or _contains_forbidden(data, list(_AGENT_WORDS)):
                raise ValueError("story step contains forbidden ungrounded agents")
            # V19: required_objects can be repaired from grounded fields after parsing.
            # Do not fail only because the LLM returned an empty list.
            if not _string_list(data.get("required_objects")):
                data["required_objects"] = _derive_required_objects(data, seed, getattr(seed, "protagonist", "protagonist"))

        data = self._llm_json_strict(prompt, f"generate_story_step_{frame_index+1}", max_tokens=700, validate=_validate, repair_hint="Make the sentence easy to draw, grounded, and free of ungrounded entities.")
        protagonist = getattr(seed, "protagonist", "protagonist")
        data = _sanitize_nested(data, forbidden + list(_AGENT_WORDS), protagonist)
        data = _force_protagonist_only_step(data, protagonist)
        data["required_objects"] = _derive_required_objects(data, seed, protagonist)
        data["background_elements"] = _derive_background_elements(data, seed, protagonist)
        data["frame_id"] = frame_index + 1
        data["sentence"] = _clean_text(data.get("sentence"))
        data["image_sentence"] = data["sentence"]
        states = getattr(emotion_arc, "states", []) or []
        intensities = getattr(emotion_arc, "intensities", []) or []
        data["emotion"] = _clean_text(data.get("emotion") or (states[frame_index] if frame_index < len(states) else ""))
        data["emotion_intensity"] = int(data.get("emotion_intensity") or (intensities[frame_index] if frame_index < len(intensities) else 3))
        data["required_objects"] = _derive_required_objects(data, seed, protagonist)
        data["background_elements"] = _derive_background_elements(data, seed, protagonist)
        data["supporting_cast"] = []
        data["characters"] = [protagonist]
        return data

    def story_step_to_frame(self, seed: StorySeed, dce_plan: DCEPlan, emotion_arc: EmotionArc, step: Dict[str, Any], frame_index: int, num_frames: int) -> StoryboardFrame:
        emotion = step.get("emotion", "")
        rule = get_emotion_rule(emotion)
        nf = "ending resolution" if frame_index == num_frames - 1 else ("story opening" if frame_index == 0 else "causal event progression")
        shot_type = choose_shot_type(frame_index, num_frames, nf)
        frame_dict = {
            "frame_id": frame_index + 1,
            "caption": step.get("sentence", ""),
            "narrative_function": nf,
            "event": step.get("action", ""),
            "protagonist_state": step.get("emotion", ""),
            "desire_link": getattr(dce_plan, "desire", ""),
            "conflict_level": min(5, max(1, frame_index + 1)),
            "emotion": step.get("emotion", ""),
            "emotion_intensity": int(step.get("emotion_intensity", 3)),
            "visual_focus": step.get("action", ""),
            "key_objects": _string_list(step.get("required_objects")),
            "facial_cue": rule["face"],
            "body_cue": rule["body"],
            "event_cue": step.get("visible_cause", ""),
            "scene_cue": step.get("location", ""),
            "cinematic_cue": rule["composition"],
            "prompt": "",
        }
        frame = _safe_make(StoryboardFrame, frame_dict)
        extra = {
            "story_sentence": step.get("sentence", ""),
            "image_sentence": step.get("image_sentence", step.get("sentence", "")),
            "event_causal_role": "visible story event",
            "event_grounding": step.get("visible_cause", ""),
            "emotion_evidence": _string_list(step.get("required_objects"))[:4],
            "evidence_objects": _string_list(step.get("required_objects"))[:6],
            "must_show": _unique(_story_locked_visual_inventory(step, seed, getattr(seed, "protagonist", "protagonist"))),
            "scene_location": step.get("location", ""),
            "time_of_day": step.get("time_of_day", ""),
            "weather": step.get("weather", ""),
            "atmosphere": step.get("atmosphere", ""),
            "environment_details": _string_list(step.get("background_elements"))[:3],
            "supporting_cast": [],
            "scene_transition": _clean_text(step.get("continuity_notes", "")),
            "character_identity": getattr(seed, "protagonist", "protagonist"),
            "character_reference_prompt": getattr((getattr(seed, "character_profiles", []) or [None])[0], "identity_anchor_prompt", ""),
            "emotion_visual_rule": emotion_rule_text(emotion),
            "shot_type": shot_type,
            "camera_shot": shot_type,
            "camera_distance": choose_camera_distance(shot_type),
            "lighting_style": rule["lighting"],
            "color_palette": rule["palette"],
            "event_grounding_text": step.get("visible_cause", ""),
            "full_story_sentence": step.get("sentence", ""),
        }
        for k, v in extra.items():
            try:
                setattr(frame, k, v)
            except Exception:
                pass
        return frame
