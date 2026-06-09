from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from .emotion_world_rules import get_world_rule
from .latent_schema import CharacterLatent, WorldLatent, EmotionLatent, VisualControlPacket


class ButterflyController:
    """
    Decoder-Encoder-Decoder controller.

    It builds a VisualControlPacket that is later converted into SDXL cross-attention adapter tokens.
    """

    def __init__(self, quality_suffix: str, negative_prompt: str, num_hypotheses: int = 3):
        self.quality_suffix = quality_suffix
        self.negative_prompt = negative_prompt
        self.num_hypotheses = num_hypotheses

    def build_character_latent(self, seed: Any) -> CharacterLatent:
        profile = None
        for p in getattr(seed, "character_profiles", []) or []:
            if getattr(p, "role", "") == "protagonist" or getattr(p, "name", "").lower() == getattr(seed, "protagonist", "").lower():
                profile = p
                break
        if profile is None and getattr(seed, "character_profiles", None):
            profile = seed.character_profiles[0]

        if profile is None:
            return CharacterLatent(
                name=getattr(seed, "protagonist", "protagonist"),
                role="protagonist",
                identity_prompt=f"same protagonist identity: {getattr(seed, 'protagonist', 'protagonist')}",
            )

        return CharacterLatent(
            name=profile.name,
            role=profile.role,
            identity_prompt=profile.to_prompt(),
            outfit_prompt=profile.outfit,
            signature_items=profile.signature_items,
            reference_image=None,
            negative_prompt=profile.negative_identity_prompt,
        )

    def create_packet(self, frame: Any, seed: Any, dce_plan: Any, memory: Dict[str, Any], style: str, previous_frame: Any = None) -> VisualControlPacket:
        character = self.build_character_latent(seed)
        world_rule = get_world_rule(getattr(frame, "emotion", ""))

        world = WorldLatent(
            scene_location=getattr(frame, "scene_location", "") or getattr(seed, "setting", ""),
            time_of_day=getattr(frame, "time_of_day", "") or "cinematic golden hour",
            weather=getattr(frame, "weather", "") or world_rule["weather"],
            atmosphere=getattr(frame, "atmosphere", "") or world_rule["environment"],
            environment_details=list(getattr(frame, "environment_details", []) or []) + [
                world_rule["environment"],
                f"lighting: {world_rule['lighting']}",
                f"color palette: {world_rule['color']}",
                f"composition: {world_rule['composition']}",
            ],
            scene_transition=getattr(frame, "scene_transition", ""),
            symbolic_objects=getattr(seed, "visual_symbols", {}) if hasattr(seed, "visual_symbols") else {},
        )

        emotion = EmotionLatent(
            emotion=getattr(frame, "emotion", ""),
            intensity=int(getattr(frame, "emotion_intensity", 3)),
            delta_from_previous=getattr(frame, "emotion_delta", ""),
            facial_rule=getattr(frame, "facial_cue", ""),
            body_rule=getattr(frame, "body_cue", ""),
            lighting_rule=world_rule["lighting"],
            color_rule=world_rule["color"],
            composition_rule=getattr(frame, "composition_rule", "") or world_rule["composition"],
        )

        intensity = max(1, min(5, emotion.intensity))
        emotion_w = 0.20 + 0.08 * intensity
        adapter_weights = {
            "character_adapter": 0.35,
            "world_adapter": 0.25,
            "emotion_adapter": emotion_w,
            "event_adapter": max(0.10, 1.0 - (0.35 + 0.25 + emotion_w)),
        }

        character_text = character.identity_prompt
        if character.outfit_prompt:
            character_text += f"; outfit: {character.outfit_prompt}"
        if character.signature_items:
            character_text += "; signature items: " + ", ".join(character.signature_items)

        event_text = (
            f"event: {getattr(frame, 'event', '')}; "
            f"visual focus: {getattr(frame, 'visual_focus', '')}; "
            f"key objects: {', '.join(getattr(frame, 'key_objects', []) or [])}"
        )
        world_text = world.to_prompt()
        emotion_text = emotion.to_prompt()

        positive_prompt = f"""
[STYLE]
{style}

[CHARACTER]
{character_text}
Preserve the same identity, face, body type, outfit, and signature items across frames.

[EVENT]
{event_text}

[WORLD / BACKGROUND / SITUATION]
{world_text}

[EMOTION]
{emotion_text}

[MEMORY]
{memory}

[QUALITY]
{self.quality_suffix}

Create a coherent cinematic storybook illustration. The protagonist emotion must be visible,
but the surrounding world must also clearly show the story situation, weather, time, atmosphere,
and environmental context.
""".strip()

        return VisualControlPacket(
            frame_id=int(getattr(frame, "frame_id", 0)),
            positive_prompt=positive_prompt,
            negative_prompt=self.negative_prompt + "; " + character.negative_prompt,
            adapter_weights=adapter_weights,
            control_metadata={
                "character_text": character_text,
                "world_text": world_text,
                "emotion_text": emotion_text,
                "event_text": event_text,
                "world": asdict(world),
                "emotion": asdict(emotion),
                "character": asdict(character),
            },
            reference_images={},
        )
