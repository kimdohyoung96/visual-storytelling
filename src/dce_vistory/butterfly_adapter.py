from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from .emotion_world_rules import get_world_rule
from .latent_schema import CharacterLatent, WorldLatent, EmotionLatent, VisualControlPacket


class ButterflyController:
    """
    DCEE-aware Decoder-Encoder-Decoder visual controller.

    This controller converts a frame-level DCEE storyboard state into a VisualControlPacket.
    It keeps the original class name for backward compatibility, but the packet now explicitly
    separates Character, World, Emotion, and Event controls, where Event is the causal visual
    evidence that links Conflict to Ending Emotion.
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

        if hasattr(profile, "to_prompt"):
            try:
                identity_prompt = profile.to_prompt()
            except Exception:
                identity_prompt = str(getattr(profile, "identity_anchor_prompt", getattr(profile, "name", "protagonist")))
        else:
            identity_prompt = "; ".join(
                str(getattr(profile, k, "")) for k in ["name", "role", "face", "hair", "body", "outfit", "identity_anchor_prompt"]
                if getattr(profile, k, "")
            )

        return CharacterLatent(
            name=getattr(profile, "name", getattr(seed, "protagonist", "protagonist")),
            role=getattr(profile, "role", "protagonist"),
            identity_prompt=identity_prompt,
            outfit_prompt=getattr(profile, "outfit", ""),
            signature_items=getattr(profile, "signature_items", []) or [],
            reference_image=None,
            negative_prompt=getattr(profile, "negative_identity_prompt", "different identity, changed face, changed outfit, inconsistent character"),
        )

    def create_packet(self, frame: Any, seed: Any, dce_plan: Any, memory: Dict[str, Any], style: str, previous_frame: Any = None) -> VisualControlPacket:
        character = self.build_character_latent(seed)
        world_rule = get_world_rule(getattr(frame, "emotion", ""))

        world = WorldLatent(
            scene_location=getattr(frame, "scene_location", "") or getattr(seed, "setting", ""),
            time_of_day=getattr(frame, "time_of_day", "") or "cinematic story time",
            weather=getattr(frame, "weather", "") or world_rule.get("weather", "cinematic weather"),
            atmosphere=getattr(frame, "atmosphere", "") or world_rule.get("environment", "emotionally meaningful atmosphere"),
            environment_details=list(getattr(frame, "environment_details", []) or []) + [
                world_rule.get("environment", "story-relevant environment"),
                f"lighting: {getattr(frame, 'lighting_style', '') or world_rule.get('lighting', '')}",
                f"color palette: {getattr(frame, 'color_palette', '') or world_rule.get('color', '')}",
                f"composition: {getattr(frame, 'composition_rule', '') or world_rule.get('composition', '')}",
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
            lighting_rule=getattr(frame, "lighting_style", "") or world_rule.get("lighting", ""),
            color_rule=getattr(frame, "color_palette", "") or world_rule.get("color", ""),
            composition_rule=getattr(frame, "composition_rule", "") or world_rule.get("composition", ""),
        )

        intensity = max(1, min(5, emotion.intensity))
        emotion_w = 0.20 + 0.08 * intensity
        adapter_weights = {
            "character_adapter": 0.32,
            "world_adapter": 0.23,
            "emotion_adapter": emotion_w,
            "event_adapter": max(0.15, 1.0 - (0.32 + 0.23 + emotion_w)),
        }

        character_text = character.identity_prompt
        if character.outfit_prompt:
            character_text += f"; outfit: {character.outfit_prompt}"
        if character.signature_items:
            character_text += "; signature items: " + ", ".join(character.signature_items)

        event_text = (
            f"DCEE visible event: {getattr(frame, 'event', '')}; "
            f"causal role: {getattr(frame, 'event_causal_role', '')}; "
            f"event grounding: {getattr(frame, 'event_grounding', '')}; "
            f"emotion evidence: {getattr(frame, 'emotion_evidence', [])}; "
            f"must show: {getattr(frame, 'must_show', [])}; "
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

[DCEE EVENT - MUST BE VISIBLE]
{event_text}

[WORLD / BACKGROUND / SITUATION]
{world_text}

[EMOTION - MUST BE CAUSED BY THE EVENT]
{emotion_text}
The image must show not only what the protagonist feels, but also why the event makes the protagonist feel it.

[SALIENT CAUSAL MEMORY]
{memory}

[QUALITY]
{self.quality_suffix}

Create a coherent full-color cinematic storybook illustration. The planned event must be visible,
the emotion must be readable, and the visual evidence connecting event to emotion must be present.
""".strip()

        return VisualControlPacket(
            frame_id=int(getattr(frame, "frame_id", 0)),
            positive_prompt=positive_prompt,
            negative_prompt=self.negative_prompt + "; " + character.negative_prompt + "; missing event, missing emotion cause, portrait only",
            adapter_weights=adapter_weights,
            control_metadata={
                "character_text": character_text,
                "world_text": world_text,
                "emotion_text": emotion_text,
                "event_text": event_text,
                "dcee_event_text": event_text,
                "event_causal_role": getattr(frame, "event_causal_role", ""),
                "event_grounding": getattr(frame, "event_grounding", ""),
                "emotion_evidence": getattr(frame, "emotion_evidence", []),
                "must_show": getattr(frame, "must_show", []),
                "world": asdict(world),
                "emotion": asdict(emotion),
                "character": asdict(character),
            },
            reference_images={},
        )
