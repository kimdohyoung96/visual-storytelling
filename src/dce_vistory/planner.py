from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from .llm import BaseLLM
from .schema import StorySeed, DCEPlan, EmotionArc, StoryboardFrame, ImageUnderstanding, CharacterProfile
from .prompts import (
    SYSTEM_NARRATIVE, QUALITY_SUFFIX, NEGATIVE_PROMPT,
    story_seed_prompt, story_abstract_prompt, dce_plan_prompt, emotion_arc_prompt, storyboard_prompt,
    get_emotion_rule, emotion_rule_text, emotion_delta_text, choose_shot_type, choose_camera_distance,
)
from .utils import extract_json


def _ending_synonym(target: str) -> str:
    target = (target or "").lower()
    return {"happy": "joy", "sad": "sadness", "angry": "anger", "fearful": "fear"}.get(target, target or "resolution")


class DCEPlanner:
    def __init__(self, llm: BaseLLM, temperature: float = 0.4, max_tokens: int = 2500):
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = max_tokens

    def build_seed(self, sample: Dict[str, Any], image_summary: ImageUnderstanding | None) -> StorySeed:
        text = self.llm.generate(SYSTEM_NARRATIVE, story_seed_prompt(sample, asdict(image_summary) if image_summary else None), temperature=self.temperature, max_tokens=self.max_tokens)
        data = extract_json(text)
        profiles = self._build_character_profiles(data, sample, image_summary)
        world_context = data.get("world_context", {})
        if image_summary:
            world_context.setdefault("time_of_day", getattr(image_summary, "time_of_day", ""))
            world_context.setdefault("weather_prior", getattr(image_summary, "weather", ""))
            world_context.setdefault("environment_prior", getattr(image_summary, "environment_details", []))
        return StorySeed(
            image_summary=image_summary,
            text_prompt=sample.get("text_prompt", ""),
            protagonist=sample.get("protagonist", ""),
            target_ending_emotion=sample.get("target_ending_emotion", ""),
            genre=sample.get("genre", ""),
            style=sample.get("style", ""),
            setting=data.get("setting", getattr(image_summary, "setting", "") if image_summary else ""),
            objects=data.get("objects", getattr(image_summary, "objects", []) if image_summary else []),
            characters=data.get("characters", getattr(image_summary, "characters", []) if image_summary else []),
            mood=data.get("mood", getattr(image_summary, "mood", "") if image_summary else ""),
            visual_symbols=data.get("visual_symbols", {}),
            world_context=world_context,
            character_profiles=profiles,
            raw_input=sample,
        )

    def _build_character_profiles(self, data: Dict[str, Any], sample: Dict[str, Any], image_summary: ImageUnderstanding | None) -> List[CharacterProfile]:
        profiles: List[CharacterProfile] = []
        for row in data.get("character_profiles", []):
            if isinstance(row, dict):
                profiles.append(CharacterProfile(
                    name=row.get("name", ""), role=row.get("role", ""), age_group=row.get("age_group", "adult"),
                    gender=row.get("gender", "unspecified"), face=row.get("face", ""), hair=row.get("hair", ""),
                    body=row.get("body", ""), outfit=row.get("outfit", ""), signature_items=row.get("signature_items", []),
                    color_palette=row.get("color_palette", ""), identity_anchor_prompt=row.get("identity_anchor_prompt", ""),
                ))
        protagonist = sample.get("protagonist", "protagonist")
        if not any(p.name.lower() == protagonist.lower() or p.role == "protagonist" for p in profiles):
            image_hint = getattr(image_summary, "caption", "") if image_summary else sample.get("text_prompt", "")
            profiles.insert(0, CharacterProfile(
                name=protagonist, role="protagonist", age_group=sample.get("age_group", "adult"),
                gender=sample.get("gender", "unspecified"),
                face=f"recognizable consistent face or character features based on: {image_hint}",
                hair="same hairstyle or head shape in every frame",
                body="same body shape and proportions in every frame",
                outfit=sample.get("outfit", "same main outfit, same colors, same accessories in every frame"),
                signature_items=list(dict.fromkeys(sample.get("signature_items", []) + data.get("objects", [])[:2])),
                color_palette="stable protagonist color palette across all frames",
                identity_anchor_prompt=f"{protagonist} must look like the same character in every frame; same face, outfit, body shape, and signature items.",
            ))
        return profiles

    def generate_abstract(self, seed: StorySeed) -> str:
        return self.llm.generate(SYSTEM_NARRATIVE, story_abstract_prompt(asdict(seed)), temperature=self.temperature, max_tokens=900).strip()

    def generate_dce_plan(self, seed: StorySeed, abstract: str) -> DCEPlan:
        data = extract_json(self.llm.generate(SYSTEM_NARRATIVE, dce_plan_prompt(asdict(seed), abstract), temperature=self.temperature, max_tokens=self.max_tokens))
        return DCEPlan(
            protagonist=data.get("protagonist", seed.protagonist), desire=data.get("desire", ""), fear=data.get("fear", ""),
            misbelief=data.get("misbelief", ""), obstacle=data.get("obstacle", ""), conflict=data.get("conflict", ""),
            event_spine=data.get("event_spine", []), turning_point=data.get("turning_point", ""),
            target_ending_emotion=data.get("target_ending_emotion", seed.target_ending_emotion),
            ending_state=data.get("ending_state", ""), moral_or_theme=data.get("moral_or_theme", ""),
        )

    def generate_emotion_arc(self, seed: StorySeed, abstract: str, dce_plan: DCEPlan, num_frames: int) -> EmotionArc:
        data = extract_json(self.llm.generate(SYSTEM_NARRATIVE, emotion_arc_prompt(asdict(seed), abstract, asdict(dce_plan), num_frames), temperature=self.temperature, max_tokens=self.max_tokens))
        states = data.get("states", [])
        intensities = data.get("intensities", [])
        if len(states) != num_frames:
            states = (states + [_ending_synonym(dce_plan.target_ending_emotion)] * num_frames)[:num_frames]
        if len(intensities) != num_frames:
            intensities = (intensities + [3] * num_frames)[:num_frames]
        if states:
            states[-1] = _ending_synonym(dce_plan.target_ending_emotion)
        return EmotionArc(states=states, intensities=[max(1, min(5, int(x))) for x in intensities], rationale=data.get("rationale", ""))

    def generate_storyboard(self, seed: StorySeed, abstract: str, dce_plan: DCEPlan, emotion_arc: EmotionArc) -> List[StoryboardFrame]:
        num_frames = len(emotion_arc.states)
        rows = extract_json(self.llm.generate(SYSTEM_NARRATIVE, storyboard_prompt(asdict(seed), abstract, asdict(dce_plan), asdict(emotion_arc), num_frames), temperature=self.temperature, max_tokens=self.max_tokens))
        if isinstance(rows, dict):
            rows = rows.get("storyboard", rows.get("frames", rows))
        if not isinstance(rows, list):
            raise ValueError(f"Storyboard must be a list, got {type(rows)}: {rows}")

        protagonist_profile = self._get_protagonist_profile(seed)
        protagonist_identity = protagonist_profile.to_prompt() if protagonist_profile else seed.protagonist
        character_reference_prompt = f"Use the same protagonist identity in every frame: {protagonist_identity}. Do not change face, hairstyle/head shape, outfit, body proportions, or signature items."

        base_time = seed.world_context.get("time_of_day", getattr(seed.image_summary, "time_of_day", "") if seed.image_summary else "")
        base_weather = seed.world_context.get("weather_prior", getattr(seed.image_summary, "weather", "") if seed.image_summary else "")
        base_env = seed.world_context.get("environment_prior", getattr(seed.image_summary, "environment_details", []) if seed.image_summary else [])

        frames: List[StoryboardFrame] = []
        prev_emotion = None
        prev_world = None
        for idx, row in enumerate(rows):
            emotion = row.get("emotion", emotion_arc.states[min(idx, len(emotion_arc.states) - 1)])
            intensity = max(1, min(5, int(row.get("emotion_intensity", emotion_arc.intensities[min(idx, len(emotion_arc.intensities) - 1)]))))
            rule = get_emotion_rule(emotion)
            scene_location = row.get("scene_location", seed.setting)
            weather = row.get("weather", base_weather) or rule["weather"]
            time_of_day = row.get("time_of_day", base_time)
            env = row.get("environment_details", base_env if isinstance(base_env, list) else [])
            if not isinstance(env, list):
                env = [str(env)]
            env = env + [f"lighting style: {rule['lighting']}", f"composition supports {emotion}: {rule['composition']}", "full-color environment, not grayscale"]
            transition = row.get("scene_transition", "Establish the initial world state clearly in the first frame." if not prev_world else f"The scene evolves from {prev_world['scene_location']} in {prev_world['weather']} weather to {scene_location} in {weather} weather.")
            narrative_function = row.get("narrative_function", "")
            shot_type = choose_shot_type(idx, len(rows), narrative_function)
            camera_distance = choose_camera_distance(shot_type)
            event = row.get("event", "")
            key_objects = row.get("key_objects", [])
            if not isinstance(key_objects, list):
                key_objects = [str(key_objects)]
            evidence = ([event] if event else []) + key_objects[:3]
            if row.get("desire_link", ""):
                evidence.append(f"evidence of desire: {row.get('desire_link')}")
            must_show = key_objects[:3] + [f"facial evidence of {emotion}", f"body evidence of {emotion}", f"environment evidence of {emotion}", "the current story event", "full-color emotional lighting"]

            neg = NEGATIVE_PROMPT + ("; " + protagonist_profile.negative_identity_prompt if protagonist_profile else "")
            frame = StoryboardFrame(
                frame_id=int(row.get("frame_id", idx + 1)), caption=row.get("caption", ""),
                narrative_function=narrative_function, event=event, protagonist_state=row.get("protagonist_state", ""),
                desire_link=row.get("desire_link", ""), conflict_level=int(row.get("conflict_level", 1)),
                emotion=emotion, emotion_intensity=intensity, visual_focus=row.get("visual_focus", ""),
                key_objects=key_objects, facial_cue=row.get("facial_cue") or rule["face"], body_cue=row.get("body_cue") or rule["body"],
                event_cue=row.get("event_cue", event), scene_cue=row.get("scene_cue", "The background and environment must visually support the event and emotion."),
                cinematic_cue=row.get("cinematic_cue", f"{shot_type}, cinematic storytelling composition"),
                scene_location=scene_location, time_of_day=time_of_day, weather=weather,
                atmosphere=row.get("atmosphere", "") or f"{emotion} atmosphere: {rule['environment']}",
                environment_details=env, supporting_cast=row.get("supporting_cast", []), scene_transition=transition,
                character_identity=protagonist_identity, character_reference_prompt=character_reference_prompt,
                emotion_delta=emotion_delta_text(prev_emotion, emotion, intensity), emotion_visual_rule=emotion_rule_text(emotion),
                composition_rule=f"{shot_type}, {camera_distance} distance, {rule['composition']}. The scene must show both the event and the visual cause of the protagonist emotion.",
                quality_rule=QUALITY_SUFFIX, negative_prompt=neg,
            )
            frame.shot_type = shot_type
            frame.camera_distance = camera_distance
            frame.color_palette = rule["palette"]
            frame.lighting_style = rule["lighting"]
            frame.must_show = must_show
            frame.emotion_evidence = evidence
            frames.append(frame)
            prev_emotion = emotion
            prev_world = {"scene_location": scene_location, "weather": weather}
        return frames

    @staticmethod
    def _get_protagonist_profile(seed: StorySeed):
        for p in seed.character_profiles:
            if p.role == "protagonist" or p.name.lower() == seed.protagonist.lower():
                return p
        return seed.character_profiles[0] if seed.character_profiles else None
