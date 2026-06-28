from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List
import re
from .story_bible import emotion_cue_from_bible, world_for_frame
from .story_graph import get_frame_graph_hints, as_list as graph_as_list, unique as graph_unique, extract_entity_candidates


def clean(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "").replace("\n", " ")).strip()


def as_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x if clean(v)]
    if isinstance(x, dict):
        return [clean(v) for v in x.values() if clean(v)]
    s = clean(x)
    return [s] if s else []


def unique(items: List[str], limit: int | None = None) -> List[str]:
    out = []
    for item in items:
        item = clean(item)
        if item and item not in out:
            out.append(item)
    return out if limit is None else out[:limit]


def shorten(text: str, max_words: int = 18) -> str:
    words = clean(text).split()
    return " ".join(words[:max_words])


_AGENT_TERMS = {"person", "man", "woman", "boy", "girl", "child", "human", "friend", "helper", "companion", "sidekick", "crowd", "another bear", "other bear", "another panda", "other panda"}


def _is_forbidden_entity_term(text: Any, protagonist: str) -> bool:
    s = clean(text).lower()
    if not s:
        return False
    p = clean(protagonist).lower()
    if s == p or s in {"protagonist", "main character", "subject"}:
        return True
    return any(t in s for t in _AGENT_TERMS)


def _filter_visual_inventory(items: List[str], protagonist: str, limit: int = 10) -> List[str]:
    out: List[str] = []
    for item in items:
        s = clean(item)
        if not s:
            continue
        if _is_forbidden_entity_term(s, protagonist):
            continue
        if s not in out:
            out.append(s)
    return out[:limit]


def _mood_visibility_hint(emotion: str, atmosphere: str, weather: str) -> str:
    mood = clean(' '.join([emotion, atmosphere, weather])).lower()
    if any(k in mood for k in ['sad', 'sadness', 'empty', 'emptiness', 'lonely', 'loneliness', 'void', 'melancholy', 'dark', 'night', 'rain', 'gloom']):
        return 'moody dark environment, but keep the protagonist clearly visible with readable face, clean silhouette, soft rim light, and enough front light to see expression and action'
    return 'balanced readable lighting, clear silhouette, and visible face, limbs, and props'


@dataclass
class FrameVisualSpec:
    frame_id: int
    total_frames: int
    story_sentence: str
    protagonist: str
    subject_identity: str
    subject_reference_policy: str
    primary_action: str
    visible_event: str
    visible_cause: str
    required_objects: List[str]
    carry_over_entities: List[str]
    recurring_entities: List[str]
    forbidden_objects: List[str]
    location: str
    weather: str
    atmosphere: str
    emotion: str
    facial_expression: str
    body_pose: str
    camera: str
    continuity: str
    negative: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _infer_action(sentence: str, event: str) -> str:
    event = clean(event)
    if event:
        return event
    return clean(sentence)


def _forbidden_for_subject(protagonist: str) -> List[str]:
    p = protagonist.lower()
    base = [
        'text', 'watermark', 'logo', 'duplicate protagonist', 'unrelated extra people',
        'wrong age', 'wrong gender', 'child version', 'baby version', 'juvenile version',
        'completely different outfit', 'generic portrait only', 'empty background', 'missing props', 'cropped feet', 'cropped face', 'cropped paws', 'cropped hands', 'cropped full body', 'extra character', 'second subject', 'tiny extra animal', 'tiny extra human', 'unreadable dark subject', 'underexposed protagonist'
    ]
    if 'panda' in p or 'bear' in p:
        base += ['human protagonist', 'human face replacing panda', 'panda turning into human', 'human instead of panda', 'another bear', 'two bears', 'bear with person']
    if 'woodcutter' in p:
        base += ['panda protagonist', 'animal protagonist', 'different man']
    return base


def build_frame_visual_spec(frame: Any, seed: Any, full_story: Dict[str, Any] | None, frame_index: int, total_frames: int, reference_image_path: str = '') -> FrameVisualSpec:
    rows = (full_story or {}).get('sentences', []) if isinstance(full_story, dict) else []
    story_row = rows[frame_index] if frame_index < len(rows) and isinstance(rows[frame_index], dict) else {}
    story_sentence = (
        clean(getattr(frame, 'image_caption_en', ''))
        or clean(story_row.get('image_caption_en') or story_row.get('image_sentence') or story_row.get('sentence'))
        or clean(getattr(frame, 'image_sentence', ''))
        or clean(getattr(frame, 'story_sentence', ''))
        or clean(getattr(frame, 'caption', ''))
    )

    story_bible = getattr(seed, '_story_bible', {}) or {}
    if not isinstance(story_bible, dict):
        story_bible = {}
    bible_world = world_for_frame(story_bible, frame) if story_bible else {}

    story_graph = getattr(seed, '_story_graph', {}) or {}
    graph_hints = get_frame_graph_hints(story_graph, frame_index, story_sentence)

    protagonist = clean(getattr(seed, 'protagonist', '')) or clean(getattr(frame, 'protagonist', '')) or 'protagonist'

    identity_parts = []
    profiles = getattr(seed, 'character_profiles', []) or []
    if profiles:
        prof = profiles[0]
        for key in ['name', 'role', 'age_group', 'gender', 'face', 'hair', 'body', 'outfit', 'identity_anchor_prompt']:
            val = clean(getattr(prof, key, ''))
            if val:
                identity_parts.append(f'{key}: {val}')
        sig = as_list(getattr(prof, 'signature_items', []))
        if sig:
            identity_parts.append('signature items: ' + ', '.join(sig))
    subject_identity = '; '.join(identity_parts) or protagonist
    if isinstance(story_bible, dict) and story_bible.get('subject_identity_prompt'):
        subject_identity = story_bible.get('subject_identity_prompt') + '; ' + subject_identity

    required_objects = _filter_visual_inventory(unique(
        as_list(story_row.get('required_objects', []))
        + as_list(story_row.get('required_objects_en', []))
        + as_list(story_row.get('object', ''))
        + as_list(getattr(frame, 'key_objects', []))
        + as_list(getattr(frame, 'evidence_objects', []))
        + as_list(getattr(frame, 'emotion_evidence', []))
        + as_list(getattr(frame, 'must_show', []))
        + as_list((bible_world or {}).get('stable_background', [])),
        14,
    ), protagonist, 10)

    emotion = clean(story_row.get('emotion')) or clean(getattr(frame, 'emotion', ''))
    bible_emotion = emotion_cue_from_bible(story_bible, emotion) if story_bible else {}
    event = clean(story_row.get('action') or story_row.get('event')) or clean(getattr(frame, 'event', ''))
    event_grounding = clean(story_row.get('visible_cause') or story_row.get('action')) or clean(getattr(frame, 'event_grounding', ''))
    visual_focus = clean(getattr(frame, 'visual_focus', ''))

    camera = (
        clean(getattr(frame, 'camera_shot', ''))
        or clean(getattr(frame, 'shot_type', ''))
        or 'medium story shot showing face, action, key objects, and background'
    )

    carry_over = unique(as_list(graph_hints.get('carry_over_entities', [])), 6)
    recurring = unique(as_list(graph_hints.get('recurring_entities', [])), 6)
    location = (
        clean((bible_world or {}).get('fixed_setting'))
        or clean(story_row.get('location'))
        or clean(graph_hints.get('location'))
        or clean(getattr(frame, 'scene_location', ''))
        or clean(getattr(seed, 'setting', ''))
    )
    weather = clean((bible_world or {}).get('weather')) or clean(graph_hints.get('weather')) or clean(getattr(frame, 'weather', ''))
    atmosphere = clean((bible_world or {}).get('atmosphere')) or clean(graph_hints.get('atmosphere')) or clean(getattr(frame, 'atmosphere', ''))

    continuity_parts = [
        f'frame {frame_index + 1}/{total_frames}',
        'visualize only this story sentence',
        'same protagonist identity across all frames',
        'advance the story instead of repeating the previous pose',
        'the current frame must visually realize the current caption more strongly than any earlier frame',
    ]
    if carry_over:
        continuity_parts.append('carry over entities: ' + ', '.join(carry_over))
    if recurring:
        continuity_parts.append('recurring story entities: ' + ', '.join(recurring[:3]))
    continuity = '; '.join(continuity_parts)

    return FrameVisualSpec(
        frame_id=int(getattr(frame, 'frame_id', frame_index + 1)),
        total_frames=total_frames,
        story_sentence=story_sentence,
        protagonist=protagonist,
        subject_identity=subject_identity,
        subject_reference_policy=(
            f'use the input reference image only to keep protagonist identity stable: {reference_image_path}'
            if reference_image_path else
            'no external reference image available; use subject identity text strictly'
        ),
        primary_action=_infer_action(story_sentence, event),
        visible_event=event or story_sentence,
        visible_cause=event_grounding or story_sentence,
        required_objects=required_objects,
        carry_over_entities=carry_over,
        recurring_entities=recurring,
        forbidden_objects=_forbidden_for_subject(protagonist),
        location=location,
        weather=weather,
        atmosphere=atmosphere,
        emotion=emotion,
        facial_expression=bible_emotion.get('face') or clean(story_row.get('facial_cue')) or clean(getattr(frame, 'facial_cue', '')) or f'clear facial expression of {emotion}',
        body_pose=bible_emotion.get('body') or clean(story_row.get('body_cue')) or clean(getattr(frame, 'body_cue', '')) or f'body pose clearly showing {emotion} while doing the action',
        camera=camera + (f'; visual focus: {visual_focus}' if visual_focus else ''),
        continuity=continuity,
        negative=', '.join(unique(_forbidden_for_subject(protagonist) + as_list((story_bible or {}).get('global_negative', [])), 60)),
    )


def prompt_from_spec(spec: FrameVisualSpec, mode: str = 'caption_locked') -> str:
    obj = ', '.join(unique(spec.required_objects, 10)) or 'only grounded story objects'
    carry = ', '.join(unique(spec.carry_over_entities, 5))
    recurring = ', '.join(unique(spec.recurring_entities, 5))
    identity = shorten(spec.subject_identity, 42)
    sentence = shorten(spec.story_sentence, 48)
    action = shorten(spec.primary_action or spec.visible_event, 24)
    cause = shorten(spec.visible_cause, 24)
    loc = shorten(spec.location, 18)
    weather = shorten(spec.weather, 10) or 'story-appropriate weather'
    atmosphere = shorten(spec.atmosphere, 12) or 'story-appropriate atmosphere'
    face = shorten(spec.facial_expression, 16)
    body = shorten(spec.body_pose, 18)
    camera = shorten(spec.camera, 20)
    visibility = _mood_visibility_hint(spec.emotion, spec.atmosphere, spec.weather)

    base = (
        f'professional full-color cinematic storybook illustration. '
        f'visual storytelling frame {spec.frame_id} of {spec.total_frames}. '
        f'caption-locked rendering: {sentence}. '
        f'exactly one protagonist only: {spec.protagonist}. '
        f'do not show any second person, second animal, duplicate protagonist, crowd, or helper. '
        f'keep protagonist identity consistent across frames: {identity}. '
        f'show a readable single scene with exactly one main moment from the current caption; not a generic portrait, not a collage, and not a repeated idle pose. '
        f'the protagonist must be the central subject and visually dominant in the frame. '
        f'show the protagonist full body or medium-wide when possible; face, limbs, and main prop must remain visible and uncropped. '
        f'current visible action: {action}. visible cause or evidence: {cause}. '
        f'only grounded visual elements may appear: {obj}. '
        f'place/background: {loc}. weather: {weather}. atmosphere: {atmosphere}. '
        f'emotion: {spec.emotion}. facial expression: {face}. body pose: {body}. '
        f'lighting/readability: {visibility}. '
        f'camera/composition: {camera}. '
        f'render the current story event clearly and literally. '
    )

    if mode == 'evidence_locked':
        extra = f'all listed required objects and visual evidence must be clearly visible in the same frame. show why the protagonist is doing the action. keep only these carry-over entities if still relevant: {carry}.'
    elif mode == 'continuity_locked':
        extra = f'preserve protagonist identity and world continuity, but the current caption and current action are more important than previous frames. recurring story entities allowed only if explicitly listed: {recurring}.'
    elif mode == 'emotion_locked':
        extra = f'the viewer must immediately understand why the protagonist feels {spec.emotion}; emotional cause and required evidence must be visible in the same scene.'
    elif mode == 'visibility_locked':
        extra = 'the protagonist must remain clearly readable even if the background is dark or moody; avoid underexposure, hide no limbs or face, and separate the protagonist from the background with clean lighting and silhouette.'
    else:
        extra = 'caption locking is mandatory: visualize this exact caption, not a generic portrait, not an unrelated moment, and not an earlier or later event.'
    return clean(base + ' ' + extra)


def negative_from_spec(spec: FrameVisualSpec) -> str:

    return clean(
        'split screen, multi panel, collage, comic page, multiple scenes, generic portrait, repeated static pose, unrelated poster image, '
        'missing protagonist, wrong protagonist, baby animal, cub, juvenile, childlike body, '
        'different identity, inconsistent character, missing action, missing event, missing evidence, missing required object, cropped out props, cropped feet, cropped paws, cropped face, '
        'wrong background, wrong weather, unrelated humans, person, man, woman, child, girl, boy, extra animal, duplicate characters, second protagonist, companion character, sidekick, tiny extra figure, '
        'unrelated object, unrelated prop, text, watermark, low quality, blurry, underexposed subject, unreadable dark face, invisible eyes, severe crop, clipped body, '
        + spec.negative
    )
