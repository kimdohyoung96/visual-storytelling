
from __future__ import annotations
from dataclasses import asdict
from typing import Any, Dict
from .emotion_world_rules import get_world_rule
from .latent_schema import CharacterLatent, WorldLatent, EmotionLatent, VisualControlPacket
from .frame_director import build_frame_visual_spec


def _normalize_symbolic_objects(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items() if str(k).strip()}
    if isinstance(value, str):
        text = value.strip()
        return {text: 'visual symbol'} if text else {}
    if isinstance(value, (list, tuple)):
        out = {}
        for item in value:
            if item is None:
                continue
            if isinstance(item, dict):
                key = item.get('object') or item.get('name') or item.get('symbol') or item.get('item') or item.get('key')
                val = item.get('meaning') or item.get('symbolic_meaning') or item.get('description') or item.get('value') or 'visual symbol'
                if key:
                    out[str(key)] = str(val)
            else:
                text = str(item).strip()
                if text:
                    out[text] = 'visual symbol'
        return out
    return {str(value): 'visual symbol'}


def _safe_join(items):
    return ', '.join([str(x) for x in (items or []) if str(x).strip()])


class ButterflyController:
    def __init__(self, quality_suffix: str, negative_prompt: str, num_hypotheses: int = 3):
        self.quality_suffix = quality_suffix
        self.negative_prompt = negative_prompt
        self.num_hypotheses = num_hypotheses

    def build_character_latent(self, seed: Any) -> CharacterLatent:
        profile = None
        for p in getattr(seed, 'character_profiles', []) or []:
            if getattr(p, 'role', '') == 'protagonist' or getattr(p, 'name', '').lower() == getattr(seed, 'protagonist', '').lower():
                profile = p
                break
        if profile is None and getattr(seed, 'character_profiles', None):
            profile = seed.character_profiles[0]
        if profile is None:
            return CharacterLatent(name=getattr(seed, 'protagonist', 'protagonist'), role='protagonist', identity_prompt=f"same protagonist identity: {getattr(seed,'protagonist','protagonist')}")
        if hasattr(profile, 'to_prompt'):
            try:
                ident = profile.to_prompt()
            except Exception:
                ident = str(getattr(profile, 'identity_anchor_prompt', getattr(profile, 'name', 'protagonist')))
        else:
            ident = '; '.join(str(getattr(profile, k, '')) for k in ['name', 'role', 'age', 'gender', 'face', 'hair', 'body', 'outfit', 'identity_anchor_prompt'] if getattr(profile, k, ''))
        return CharacterLatent(
            name=getattr(profile, 'name', getattr(seed, 'protagonist', 'protagonist')),
            role=getattr(profile, 'role', 'protagonist'),
            identity_prompt=ident,
            outfit_prompt=getattr(profile, 'outfit', ''),
            signature_items=getattr(profile, 'signature_items', []) or [],
            reference_image=None,
            negative_prompt=getattr(profile, 'negative_identity_prompt', 'different identity, changed face, changed outfit, inconsistent character, child version, older version, gender changed'),
        )

    def create_packet(self, frame: Any, seed: Any, dce_plan: Any, memory: Dict[str, Any], style: str, previous_frame: Any = None, anchors: Dict[str, Any] | None = None) -> VisualControlPacket:
        character = self.build_character_latent(seed)
        world_rule = get_world_rule(getattr(frame, 'emotion', ''))
        world = WorldLatent(
            scene_location=getattr(frame, 'scene_location', '') or getattr(seed, 'setting', ''),
            time_of_day=getattr(frame, 'time_of_day', '') or 'cinematic story time',
            weather=getattr(frame, 'weather', '') or world_rule.get('weather', 'cinematic weather'),
            atmosphere=getattr(frame, 'atmosphere', '') or world_rule.get('environment', 'emotionally meaningful atmosphere'),
            environment_details=list(getattr(frame, 'environment_details', []) or []) + [world_rule.get('environment', 'story-relevant environment'), f"lighting: {getattr(frame,'lighting_style','') or world_rule.get('lighting','')}", f"color palette: {getattr(frame,'color_palette','') or world_rule.get('color','')}", f"composition: {getattr(frame,'composition_rule','') or world_rule.get('composition','')}"],
            scene_transition=getattr(frame, 'scene_transition', ''),
            symbolic_objects=_normalize_symbolic_objects(getattr(seed, 'visual_symbols', {}) if hasattr(seed, 'visual_symbols') else {}),
        )
        emotion = EmotionLatent(
            emotion=getattr(frame, 'emotion', ''),
            intensity=int(getattr(frame, 'emotion_intensity', 3)),
            delta_from_previous=getattr(frame, 'emotion_delta', ''),
            facial_rule=getattr(frame, 'facial_cue', ''),
            body_rule=getattr(frame, 'body_cue', ''),
            lighting_rule=getattr(frame, 'lighting_style', '') or world_rule.get('lighting', ''),
            color_rule=getattr(frame, 'color_palette', '') or world_rule.get('color', ''),
            composition_rule=getattr(frame, 'composition_rule', '') or world_rule.get('composition', ''),
        )
        intensity = max(1, min(5, emotion.intensity))
        adapter_weights = {'character_adapter': 0.24, 'world_adapter': 0.16, 'emotion_adapter': 0.18 + 0.04 * intensity, 'event_adapter': 0.24, 'evidence_adapter': 0.24}

        identity_lock = getattr(frame, 'identity_lock', {}) or {}
        identity_lock_text = '; '.join(f"{k}: {v}" for k, v in identity_lock.items() if v)
        char_text = (
            character.identity_prompt
            + '; STRICT IDENTITY LOCK: same age, gender, face shape, hairstyle, body proportions, skin tone, outfit base, and signature items across all frames.'
            + ' Only expression, pose, camera shot, event action, weather, and background may change.'
            + (' ' + identity_lock_text if identity_lock_text else '')
            + (f"; outfit: {character.outfit_prompt}" if character.outfit_prompt else '')
            + (('; signature items: ' + _safe_join(character.signature_items)) if character.signature_items else '')
        )

        salient_history = (memory or {}).get('salient_history', '')
        continuity = (memory or {}).get('continuity_constraints', {}) or {}
        continuity_text = '; '.join(f"{k}: {v}" for k, v in continuity.items() if v)
        anchor_text = str(anchors or {})
        source_reference_image_path = getattr(seed, 'source_image_path', '') or getattr(seed, 'image_path', '')
        frame_visual_spec = build_frame_visual_spec(frame, seed, getattr(seed, '_current_full_story', None), int(getattr(frame, 'frame_id', 1)) - 1, int(getattr(seed, '_total_frames', 6)), source_reference_image_path)
        prev_text = ''
        if previous_frame is not None:
            prev_text = f"previous frame event={getattr(previous_frame,'event','')}; previous emotion={getattr(previous_frame,'emotion','')}; progress the story forward instead of repeating the same pose or action."

        story_text = f"story sentence: {getattr(frame,'story_sentence','')}; alignment reason: {getattr(frame,'story_alignment_reason','')}; caption: {getattr(frame,'caption','')}"
        frame_goal_text = f"Frame goal: visualize exactly this narrative beat -> {getattr(frame,'story_sentence','')}. The visible scene must clearly show the event `{getattr(frame,'event','')}` and why the protagonist feels `{getattr(frame,'emotion','')}`."
        event_text = f"DCEE visible event: {getattr(frame,'event','')}; causal role: {getattr(frame,'event_causal_role','')}; event grounding: {getattr(frame,'event_grounding','')}; narrative function: {getattr(frame,'narrative_function','')}"
        evidence_text = f"key objects: {getattr(frame,'key_objects',[])}; visual evidence objects: {getattr(frame,'evidence_objects',[])}; emotion evidence: {getattr(frame,'emotion_evidence',[])}; must show: {getattr(frame,'must_show',[])}; visual focus: {getattr(frame,'visual_focus','')}; scene must show both the event and the visible cause of the protagonist emotion"
        world_text = world.to_prompt()
        emotion_text = emotion.to_prompt()

        positive = f"""
[STYLE] {style}
[FRAME GOAL] {frame_goal_text}
[CHARACTER] {char_text}
[STORY SENTENCE] {story_text}
[DCEE EVENT] {event_text}
[EVIDENCE] {evidence_text}
[WORLD] {world_text}
[EMOTION CAUSED BY EVENT] {emotion_text}
[SALIENT HISTORY] {salient_history}
[CONTINUITY CONSTRAINTS] {continuity_text}
[PREVIOUS FRAME CONTEXT] {prev_text}
[ANCHORS] {anchor_text}
[QUALITY] {self.quality_suffix}
Create a coherent full-color cinematic storybook illustration.
Requirements:
1. The selected frame must match the story sentence, not just a generic portrait.
2. The protagonist identity must remain consistent with prior frames.
3. The scene must visibly show the event, the key evidence objects, and the emotional cause.
4. The background, weather, lighting, and camera framing must support the specific narrative beat.
5. Advance the story from the previous frame; do not repeat the same static composition.
""".strip()
        negative = self.negative_prompt + '; ' + character.negative_prompt + '; generic portrait only, static pose repetition, missing event, missing evidence, weak emotion, missing visual cause, empty background, inconsistent protagonist, child version, older version, gender changed, grayscale, monochrome'
        return VisualControlPacket(
            frame_id=int(getattr(frame, 'frame_id', 0)),
            positive_prompt=positive,
            negative_prompt=negative,
            adapter_weights=adapter_weights,
            control_metadata={
                'character_text': char_text,
                'world_text': world_text,
                'emotion_text': emotion_text,
                'story_text': story_text,
                'frame_goal_text': frame_goal_text,
                'event_text': event_text,
                'evidence_text': evidence_text,
                'history_text': salient_history,
                'continuity_text': continuity_text,
                'anchor_text': anchor_text,
                'dcee_event_text': event_text,
                'event_causal_role': getattr(frame, 'event_causal_role', ''),
                'event_grounding': getattr(frame, 'event_grounding', ''),
                'evidence_objects': getattr(frame, 'evidence_objects', []),
                'emotion_evidence': getattr(frame, 'emotion_evidence', []),
                'must_show': getattr(frame, 'must_show', []),
                'story_sentence': getattr(frame, 'story_sentence', ''),
                'visual_focus': getattr(frame, 'visual_focus', ''),
                'camera_shot': getattr(frame, 'camera_shot', ''),
                'key_objects': getattr(frame, 'key_objects', []),
                'frame': {'frame_id': getattr(frame, 'frame_id', 0), 'narrative_function': getattr(frame, 'narrative_function', ''), 'event': getattr(frame, 'event', ''), 'event_grounding': getattr(frame, 'event_grounding', ''), 'emotion': getattr(frame, 'emotion', ''), 'emotion_intensity': getattr(frame, 'emotion_intensity', 3), 'conflict_level': getattr(frame, 'conflict_level', 3), 'story_sentence': getattr(frame, 'story_sentence', ''), 'must_show': getattr(frame, 'must_show', []), 'camera_shot': getattr(frame, 'camera_shot', ''), 'visual_focus': getattr(frame, 'visual_focus', ''), 'scene_location': getattr(frame, 'scene_location', ''), 'weather': getattr(frame, 'weather', ''), 'atmosphere': getattr(frame, 'atmosphere', '')},
                'world': asdict(world),
                'emotion': asdict(emotion),
                'character': asdict(character),
                'source_reference_image_path': source_reference_image_path,
                'frame_visual_spec': frame_visual_spec.to_dict(),
            },
            reference_images={'subject': source_reference_image_path} if source_reference_image_path else {},
        )
