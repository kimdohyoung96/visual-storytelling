from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from typing import Any, Dict, List

from .llm import BaseLLM
from .schema import StorySeed, DCEPlan, EmotionArc, StoryboardFrame, ImageUnderstanding, CharacterProfile
from .prompts import (
    SYSTEM_NARRATIVE,
    QUALITY_SUFFIX,
    NEGATIVE_PROMPT,
    story_seed_prompt,
    story_abstract_prompt,
    dce_plan_prompt,
    emotion_arc_prompt,
    storyboard_prompt,
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
    """
    Convert arbitrary LLM/sample output into a deduplicated list[str].

    Fixes:
        TypeError: unhashable type: 'dict'

    Cause:
        data["objects"] or sample["signature_items"] can include dictionaries.
        list(dict.fromkeys(raw_list)) fails when raw_list contains dicts.
    """
    if value is None:
        return []

    if isinstance(value, str):
        value = [value]
    elif isinstance(value, dict):
        value = [value]
    elif not isinstance(value, list):
        value = [value]

    items: List[str] = []
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
                or item.get("id")
                or str(item)
            )
        else:
            text = str(item)

        text = str(text).strip()
        if text:
            items.append(text)

    return list(dict.fromkeys(items))


def _ending_synonym(target: str) -> str:
    target = (target or "").lower().strip()
    return {
        "happy": "joy",
        "happiness": "joy",
        "sad": "sadness",
        "angry": "anger",
        "fearful": "fear",
        "sad ending": "sadness",
        "happy ending": "joy",
    }.get(target, target or "resolution")


class DCEPlanner:
    """Schema-compatible DCEE planner: Desire -> Conflict -> Event Chain -> Ending Emotion."""

    def __init__(self, llm: BaseLLM, temperature: float = 0.4, max_tokens: int = 2500):
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens

    def build_seed(self, sample: Dict[str, Any], image_summary: ImageUnderstanding | None) -> StorySeed:
        text = self.llm.generate(
            SYSTEM_NARRATIVE,
            story_seed_prompt(sample, _to_dict(image_summary) if image_summary else None),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        data = extract_json(text)

        # LLM이 objects/characters를 dict list로 뱉는 경우를 방어
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

        seed_kwargs = {
            "image_summary": image_summary,
            "text_prompt": sample.get("text_prompt", ""),
            "protagonist": sample.get("protagonist", ""),
            "target_ending_emotion": sample.get("target_ending_emotion", ""),
            "genre": sample.get("genre", ""),
            "style": sample.get("style", ""),
            "setting": data.get("setting", getattr(image_summary, "setting", "") if image_summary else ""),
            "objects": data.get("objects", getattr(image_summary, "objects", []) if image_summary else []),
            "characters": data.get("characters", getattr(image_summary, "characters", []) if image_summary else []),
            "mood": data.get("mood", getattr(image_summary, "mood", "") if image_summary else ""),
            "visual_symbols": data.get("visual_symbols", {}),
            "world_context": world_context,
            "character_profiles": profiles,
            "raw_input": sample,
        }

        seed = _safe_make(StorySeed, seed_kwargs)
        seed.world_context = world_context
        seed.character_profiles = profiles
        seed.raw_input = sample
        return seed

    def _build_character_profiles(
        self,
        data: Dict[str, Any],
        sample: Dict[str, Any],
        image_summary: ImageUnderstanding | None,
    ) -> List[CharacterProfile]:
        profiles: List[CharacterProfile] = []

        raw_profiles = data.get("character_profiles", [])
        if isinstance(raw_profiles, dict):
            raw_profiles = [raw_profiles]
        if not isinstance(raw_profiles, list):
            raw_profiles = []

        for row in raw_profiles:
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

        protagonist = sample.get("protagonist", "protagonist")
        if not any(
            getattr(p, "name", "").lower() == str(protagonist).lower()
            or getattr(p, "role", "") == "protagonist"
            for p in profiles
        ):
            image_hint = getattr(image_summary, "caption", "") if image_summary else sample.get("text_prompt", "")

            # ✅ 핵심 수정: raw list에 dict가 섞여도 안전하게 문자열 리스트로 변환 후 중복 제거
            signature_items = _string_list(
                _string_list(sample.get("signature_items", []))
                + _string_list(data.get("objects", []))[:2]
            )

            profiles.insert(
                0,
                _safe_make(
                    CharacterProfile,
                    {
                        "name": protagonist,
                        "role": "protagonist",
                        "age_group": sample.get("age_group", "adult"),
                        "gender": sample.get("gender", "unspecified"),
                        "face": f"recognizable consistent face or character features based on: {image_hint}",
                        "hair": "same hairstyle or head shape in every frame",
                        "body": "same body shape and proportions in every frame",
                        "outfit": sample.get(
                            "outfit",
                            "same main outfit, same colors, same accessories in every frame",
                        ),
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

    def generate_abstract(self, seed: StorySeed) -> str:
        return self.llm.generate(
            SYSTEM_NARRATIVE,
            story_abstract_prompt(_to_dict(seed)),
            temperature=self.temperature,
            max_tokens=900,
        ).strip()

    def generate_dce_plan(self, seed: StorySeed, abstract: str) -> DCEPlan:
        data = extract_json(
            self.llm.generate(
                SYSTEM_NARRATIVE,
                dce_plan_prompt(_to_dict(seed), abstract),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        )

        event_spine = data.get("event_spine", data.get("event_chain", []))
        if isinstance(event_spine, dict):
            event_spine = event_spine.get("events", [event_spine])
        if not isinstance(event_spine, list):
            event_spine = [str(event_spine)]

        normalized_event_spine = []
        for idx, ev in enumerate(event_spine):
            if isinstance(ev, dict):
                ev.setdefault("event_id", f"e{idx + 1}")
                ev.setdefault("causal_role", ev.get("role", "causes or intensifies the protagonist emotion"))
                ev.setdefault("visual_grounding", ev.get("visual_evidence", ev.get("description", ev.get("event", ""))))
                normalized_event_spine.append(ev)
            else:
                normalized_event_spine.append(
                    {
                        "event_id": f"e{idx + 1}",
                        "event": str(ev),
                        "causal_role": "causes or intensifies the protagonist emotion",
                        "visual_grounding": str(ev),
                    }
                )

        dce_plan = _safe_make(
            DCEPlan,
            {
                "protagonist": data.get("protagonist", getattr(seed, "protagonist", "")),
                "desire": data.get("desire", ""),
                "fear": data.get("fear", ""),
                "misbelief": data.get("misbelief", ""),
                "obstacle": data.get("obstacle", ""),
                "conflict": data.get("conflict", ""),
                "event_spine": normalized_event_spine,
                "turning_point": data.get("turning_point", ""),
                "target_ending_emotion": data.get("target_ending_emotion", getattr(seed, "target_ending_emotion", "")),
                "ending_state": data.get("ending_state", ""),
                "moral_or_theme": data.get("moral_or_theme", ""),
                "event_chain": normalized_event_spine,
                "planning_structure": "DCEE: Desire-Conflict-Event-Ending Emotion",
            },
        )
        dce_plan.event_chain = normalized_event_spine
        dce_plan.planning_structure = "DCEE: Desire-Conflict-Event-Ending Emotion"
        return dce_plan

    def generate_emotion_arc(
        self,
        seed: StorySeed,
        abstract: str,
        dce_plan: DCEPlan,
        num_frames: int,
    ) -> EmotionArc:
        data = extract_json(
            self.llm.generate(
                SYSTEM_NARRATIVE,
                emotion_arc_prompt(_to_dict(seed), abstract, _to_dict(dce_plan), num_frames),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        )

        states = data.get("states", [])
        intensities = data.get("intensities", [])
        if len(states) != num_frames:
            states = (states + [_ending_synonym(getattr(dce_plan, "target_ending_emotion", ""))] * num_frames)[:num_frames]
        if len(intensities) != num_frames:
            intensities = (intensities + [3] * num_frames)[:num_frames]
        if states:
            states[-1] = _ending_synonym(getattr(dce_plan, "target_ending_emotion", ""))

        return _safe_make(
            EmotionArc,
            {
                "states": states,
                "intensities": [max(1, min(5, int(x))) for x in intensities],
                "rationale": data.get("rationale", ""),
            },
        )

    def generate_storyboard(
        self,
        seed: StorySeed,
        abstract: str,
        dce_plan: DCEPlan,
        emotion_arc: EmotionArc,
    ) -> List[StoryboardFrame]:
        states = getattr(emotion_arc, "states", [])
        intensities = getattr(emotion_arc, "intensities", [])
        num_frames = len(states)

        rows = extract_json(
            self.llm.generate(
                SYSTEM_NARRATIVE,
                storyboard_prompt(_to_dict(seed), abstract, _to_dict(dce_plan), _to_dict(emotion_arc), num_frames),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        )

        if isinstance(rows, dict):
            rows = rows.get("storyboard", rows.get("frames", rows))
        if not isinstance(rows, list):
            raise ValueError(f"Storyboard must be a list, got {type(rows)}: {rows}")

        protagonist_profile = self._get_protagonist_profile(seed)
        protagonist_identity = self._profile_to_prompt(protagonist_profile) if protagonist_profile else getattr(seed, "protagonist", "protagonist")
        character_reference_prompt = (
            f"Use the same protagonist identity in every frame: {protagonist_identity}. "
            "Do not change face, hairstyle/head shape, outfit, body proportions, or signature items."
        )

        world_context = getattr(seed, "world_context", {}) or {}
        image_summary = getattr(seed, "image_summary", None)
        base_time = world_context.get("time_of_day", getattr(image_summary, "time_of_day", "") if image_summary else "")
        base_weather = world_context.get("weather_prior", getattr(image_summary, "weather", "") if image_summary else "")
        base_env = world_context.get("environment_prior", getattr(image_summary, "environment_details", []) if image_summary else [])

        event_chain = getattr(dce_plan, "event_chain", getattr(dce_plan, "event_spine", [])) or []
        if isinstance(event_chain, dict):
            event_chain = [event_chain]

        frames: List[StoryboardFrame] = []
        prev_emotion = None
        prev_world = None

        for idx, row in enumerate(rows):
            emotion = row.get("emotion", states[min(idx, len(states) - 1)] if states else "neutral")
            intensity = max(1, min(5, int(row.get("emotion_intensity", intensities[min(idx, len(intensities) - 1)] if intensities else 3))))
            rule = get_emotion_rule(emotion)

            linked_event = event_chain[min(idx, len(event_chain) - 1)] if event_chain else {}
            if not isinstance(linked_event, dict):
                linked_event = {
                    "event": str(linked_event),
                    "causal_role": "causes emotion",
                    "visual_grounding": str(linked_event),
                }

            row_event = row.get("event") or linked_event.get("event") or linked_event.get("description") or ""
            event_causal_role = row.get("event_causal_role") or linked_event.get("causal_role", "causes or intensifies the frame emotion")
            event_grounding = row.get("event_grounding") or linked_event.get("visual_grounding", row_event)

            scene_location = row.get("scene_location", getattr(seed, "setting", ""))
            weather = row.get("weather", base_weather) or rule["weather"]
            time_of_day = row.get("time_of_day", base_time)

            env = _string_list(row.get("environment_details", base_env if isinstance(base_env, list) else []))
            env = env + [
                f"lighting style: {rule['lighting']}",
                f"composition supports {emotion}: {rule['composition']}",
                "full-color environment, not grayscale",
            ]

            transition = row.get(
                "scene_transition",
                "Establish the initial world state clearly in the first frame."
                if not prev_world
                else f"The scene evolves from {prev_world['scene_location']} in {prev_world['weather']} weather to {scene_location} in {weather} weather.",
            )

            narrative_function = row.get("narrative_function", "")
            shot_type = choose_shot_type(idx, len(rows), narrative_function)
            camera_distance = choose_camera_distance(shot_type)

            key_objects = _string_list(row.get("key_objects", []))
            evidence = _string_list(
                [row_event, event_grounding]
                + key_objects[:3]
                + ([f"evidence of desire: {row.get('desire_link')}"] if row.get("desire_link", "") else [])
            )

            must_show = _string_list(
                key_objects[:3]
                + [
                    row_event,
                    event_grounding,
                    f"facial evidence of {emotion}",
                    f"body evidence of {emotion}",
                    f"environment evidence of {emotion}",
                    "the current DCEE event",
                    "the visual cause of the protagonist emotion",
                    "full-color emotional lighting",
                ]
            )

            neg = NEGATIVE_PROMPT
            if protagonist_profile:
                neg += "; " + getattr(protagonist_profile, "negative_identity_prompt", "")

            frame = _safe_make(
                StoryboardFrame,
                {
                    "frame_id": int(row.get("frame_id", idx + 1)),
                    "caption": row.get("caption", ""),
                    "narrative_function": narrative_function,
                    "event": row_event,
                    "protagonist_state": row.get("protagonist_state", ""),
                    "desire_link": row.get("desire_link", ""),
                    "conflict_level": int(row.get("conflict_level", 1)),
                    "emotion": emotion,
                    "emotion_intensity": intensity,
                    "visual_focus": row.get("visual_focus", ""),
                    "key_objects": key_objects,
                    "facial_cue": row.get("facial_cue") or rule["face"],
                    "body_cue": row.get("body_cue") or rule["body"],
                    "event_cue": row.get("event_cue", row_event),
                    "scene_cue": row.get("scene_cue", "The background and environment must visually support the DCEE event and emotion."),
                    "cinematic_cue": row.get("cinematic_cue", f"{shot_type}, cinematic storytelling composition"),
                    "scene_location": scene_location,
                    "time_of_day": time_of_day,
                    "weather": weather,
                    "atmosphere": row.get("atmosphere", "") or f"{emotion} atmosphere",
                    "environment_details": env,
                    "supporting_cast": _string_list(row.get("supporting_cast", [])),
                    "scene_transition": transition,
                    "character_identity": protagonist_identity,
                    "character_reference_prompt": character_reference_prompt,
                    "emotion_delta": emotion_delta_text(prev_emotion, emotion, intensity),
                    "emotion_visual_rule": emotion_rule_text(emotion),
                    "composition_rule": (
                        f"{shot_type}, {camera_distance} distance, {rule['composition']}. "
                        "The image must show the DCEE event and the visual cause of the protagonist emotion."
                    ),
                    "quality_rule": QUALITY_SUFFIX,
                    "negative_prompt": neg,
                    "dcee_stage": "Event",
                    "event_causal_role": event_causal_role,
                    "event_grounding": event_grounding,
                    "event_emotion_causal_consistency": (
                        f"The event '{row_event}' should naturally explain or intensify the emotion '{emotion}'."
                    ),
                    "shot_type": shot_type,
                    "camera_distance": camera_distance,
                    "color_palette": rule["palette"],
                    "lighting_style": rule["lighting"],
                    "must_show": must_show,
                    "emotion_evidence": evidence,
                },
            )

            frames.append(frame)
            prev_emotion = emotion
            prev_world = {"scene_location": scene_location, "weather": weather}

        return frames

    @staticmethod
    def _profile_to_prompt(profile: CharacterProfile) -> str:
        if hasattr(profile, "to_prompt"):
            try:
                return profile.to_prompt()
            except Exception:
                pass
        return ", ".join(
            str(getattr(profile, k, ""))
            for k in [
                "name",
                "role",
                "face",
                "hair",
                "body",
                "outfit",
                "signature_items",
                "color_palette",
                "identity_anchor_prompt",
            ]
            if getattr(profile, k, "")
        )

    @staticmethod
    def _get_protagonist_profile(seed: StorySeed):
        for p in getattr(seed, "character_profiles", []):
            if getattr(p, "role", "") == "protagonist" or getattr(p, "name", "").lower() == getattr(seed, "protagonist", "").lower():
                return p
        cps = getattr(seed, "character_profiles", [])
        return cps[0] if cps else None
