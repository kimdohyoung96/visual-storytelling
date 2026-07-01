from __future__ import annotations

from dataclasses import dataclass, asdict, field
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


_AGENT_TERMS = {"person", "man", "woman", "boy", "girl", "child", "human", "friend", "helper", "companion", "sidekick", "crowd", "another bear", "other bear", "another panda", "other panda", "second bear", "mirror character"}

_ACTION_TERMS = {
    'enters','enter','searches','search','looks','look','climbs','climb','follows','follow','hears','hear','arrives','arrive','retrieves','retrieve','picks up','pick up','holds','holding',
    'determined','frustrated','hopeful','curious','excited','joyful','accomplished','uncertain'
}
_GENERIC_VISUAL_TERMS = {'evidence','event','emotion','foreground','midground','background','visible evidence','scene','story action'}

def _looks_like_action_or_generic(text: Any) -> bool:
    s = clean(text).lower()
    if not s:
        return True
    if s in _ACTION_TERMS or s in _GENERIC_VISUAL_TERMS:
        return True
    if re.fullmatch(r'(is|are|was|were|be|being|been|to)\b.*', s):
        return True
    if len(s.split()) == 1 and s.endswith('ing'):
        return True
    return False


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
        if _looks_like_action_or_generic(s):
            continue
        if s not in out:
            out.append(s)
    return out[:limit]


def _compact_subject_identity(protagonist: str, subject_identity: str) -> str:
    text = clean(f"{protagonist} {subject_identity}").lower()
    parts: List[str] = []
    if 'white bear' in text or ('bear' in text and 'white' in text):
        parts += ['adult white bear', 'white fur']
    elif 'brown bear' in text or ('bear' in text and 'brown' in text):
        parts += ['adult brown bear', 'brown fur']
    elif 'polar bear' in text:
        parts += ['adult polar bear', 'white fur']
    elif 'panda' in text:
        parts += ['adult panda', 'black and white fur']
    elif protagonist:
        parts.append(clean(protagonist))
    for key in ['friendly', 'gentle', 'large', 'fluffy']:
        if key in text:
            parts.append(key)
    out = []
    for p in parts:
        p = clean(p)
        if p and p not in out:
            out.append(p)
    return '; '.join(out[:4]) or clean(protagonist) or 'protagonist'




def _identity_lock_from_sources(protagonist: str, raw_subject_identity: str, story_bible: Dict[str, Any] | None, reference_image_path: str = '') -> str:
    """Build a stable identity contract that is stronger than the short SDXL anchor.

    V33 compressed identity into species/color only. That helped prompt length, but it also
    removed the exact face/body/signature cues that keep the protagonist consistent. V34 keeps
    a short anchor for SDXL while also storing a stricter identity lock used by the generator.
    """
    bible = story_bible if isinstance(story_bible, dict) else {}
    src = ' ; '.join(unique([
        clean(protagonist),
        clean(bible.get('subject_identity_prompt', '')),
        clean(raw_subject_identity),
        clean(bible.get('character_identity', '')),
        clean(bible.get('identity_lock', '')),
    ], 8))
    compact = _compact_subject_identity(protagonist, src)
    clauses = [
        f"exactly one {clean(protagonist) or 'protagonist'}",
        compact,
        "same species, age, body size, silhouette, face shape, fur/color pattern, and signature items in every frame",
        "hands/paws, feet, face, ears, and body proportions must remain consistent and visible",
        "only expression, gaze, pose, small scene-caused dirt/wetness, and lighting may change",
    ]
    if reference_image_path:
        clauses.append("use the input reference image as the protagonist identity anchor, not as a background template")
    return '; '.join(unique(clauses, 7))


def _infer_scene_location_from_text(text: str) -> str:
    t = clean(text).lower()
    rules = [
        (("river", "stream", "waterfall", "flowing water", "riverbank"), "riverbank with visible flowing water"),
        (("lake", "pond"), "lakeshore with visible water"),
        (("forest", "woods", "tree", "underbrush", "trail"), "forest path with trees and underbrush"),
        (("cave", "den"), "cave entrance with surrounding rocks"),
        (("mountain", "cliff", "hill"), "mountain path with distant landscape"),
        (("village", "town", "market"), "small village street with story props"),
        (("home", "house", "room", "cabin"), "cozy interior room with contextual objects"),
        (("field", "meadow", "grass"), "open meadow with grass and sky"),
        (("snow", "ice", "winter"), "snowy outdoor landscape"),
        (("rain", "storm"), "rainy outdoor scene with wet ground"),
    ]
    for keys, loc in rules:
        if any(k in t for k in keys):
            return loc
    return "story-specific environment visible behind the action"



def _strong_story_required_objects(story_sentence: str, protagonist: str, signature_items: List[str] | None = None) -> List[str]:
    text = clean(story_sentence).lower()
    out: List[str] = []
    signature_items = signature_items or []
    for s in signature_items:
        ss = clean(s)
        if ss and ss.lower() not in {clean(protagonist).lower(), 'protagonist'} and ss not in out:
            out.append(ss)

    # Story object / environment nouns to keep visible in the frame.
    noun_rules = [
        (['honey jar', 'lost honey jar', 'jar'], 'honey jar'),
        (['bush', 'bushes', 'underbrush'], 'bushes or underbrush'),
        (['root', 'roots'], 'tangled roots'),
        (['hill', 'small hill', 'slope', 'elevation'], 'small hill or slope'),
        (['path', 'trail'], 'forest path'),
        (['lake', 'water', 'shore'], 'serene lake'),
        (['forest', 'woods', 'tree', 'deep forest'], 'deep forest trees'),
        (['glimmer'], 'distant glimmer or shining clue'),
    ]
    for keys, label in noun_rules:
        if any(k in text for k in keys) and label not in out:
            out.append(label)
    return out[:8]

def _extract_scene_elements(sentence: str, story_row: Dict[str, Any], frame: Any, bible_world: Dict[str, Any] | None) -> List[str]:
    text = ' '.join([
        clean(sentence),
        clean(story_row.get('location', '')),
        clean(story_row.get('weather', '')),
        clean(story_row.get('atmosphere', '')),
        clean(getattr(frame, 'scene_location', '')),
        clean(getattr(frame, 'weather', '')),
        clean(getattr(frame, 'atmosphere', '')),
    ]).lower()
    out: List[str] = []
    keyword_map = {
        'river': 'visible river water', 'stream': 'visible flowing stream', 'water': 'visible water surface',
        'forest': 'trees and underbrush', 'tree': 'trees', 'woods': 'wooded background',
        'rain': 'rain and wet ground', 'storm': 'storm clouds', 'snow': 'snowy ground',
        'lake': 'lakeshore', 'pond': 'pond edge', 'cave': 'rocky cave entrance',
        'path': 'clear walking path', 'bridge': 'small bridge', 'home': 'home interior details',
        'house': 'house exterior/interior context', 'market': 'market stalls', 'village': 'village buildings',
        'mountain': 'distant mountains', 'field': 'open field and sky', 'meadow': 'meadow flowers and grass',
    }
    for k, v in keyword_map.items():
        if k in text:
            out.append(v)
    if bible_world:
        out += as_list(bible_world.get('stable_background', []))[:2]
    return unique(out, 8)


def _dcee_stage(frame_index: int, total_frames: int) -> str:
    if total_frames <= 1:
        return 'single ending state'
    r = frame_index / max(1, total_frames - 1)
    if r <= 0.20:
        return 'Desire setup'
    if r <= 0.55:
        return 'Conflict escalation'
    if r <= 0.82:
        return 'Event turning point'
    return 'Ending resolution'


def _dcee_appearance_delta(frame_index: int, total_frames: int, emotion: str, facial_expression: str, body_pose: str, weather: str) -> str:
    stage = _dcee_stage(frame_index, total_frames)
    emotion = clean(emotion) or 'current emotion'
    face = clean(facial_expression) or f'face clearly shows {emotion}'
    pose = clean(body_pose) or f'body pose clearly shows {emotion}'
    weather = clean(weather)
    if stage.startswith('Desire'):
        base = 'clean stable identity; hopeful/curious posture; no damage or redesign'
    elif stage.startswith('Conflict'):
        base = 'same identity under pressure; tense posture; small scene-caused dirt, wet fur/clothes, or scratches allowed only if the story/weather requires it'
    elif stage.startswith('Event'):
        base = 'same identity during decisive action; active pose and clear object interaction; do not redesign the face/body'
    else:
        base = 'same identity at resolution; calmer posture and softened expression; accumulated scene traces may remain but species/body/face stay unchanged'
    if weather:
        base += f'; weather effect may affect surface only: {weather}'
    return f'{stage}: {base}; expression cue: {face}; pose cue: {pose}'


def _scene_contract(sentence: str, action: str, evidence: str, location: str, weather: str, atmosphere: str, required_objects: List[str], scene_elements: List[str]) -> str:
    objs = ', '.join(unique(required_objects, 5)) or 'caption-grounded objects'
    bg = ', '.join(unique(scene_elements, 5)) or clean(location) or _infer_scene_location_from_text(sentence)
    parts = [
        f'foreground: protagonist performing {shorten(action or sentence, 12)}',
        f'midground/evidence: {shorten(evidence or objs, 12)}',
        f'background: {shorten(location or bg, 12)} with {shorten(bg, 12)}',
    ]
    if weather:
        parts.append(f'weather: {shorten(weather, 6)}')
    if atmosphere:
        parts.append(f'atmosphere: {shorten(atmosphere, 6)}')
    parts.append('compose as a complete story scene, not a cropped portrait')
    return '; '.join(unique(parts, 7))



def _background_contract(sentence: str, location: str, weather: str, atmosphere: str, scene_elements: List[str], required_objects: List[str]) -> str:
    bg = ', '.join(unique(scene_elements, 6)) or clean(location) or _infer_scene_location_from_text(sentence)
    extra = ', '.join(unique(required_objects, 4)) or 'story props'
    parts = [
        f'background must be clearly visible and match the current sentence: {shorten(sentence, 14)}',
        f'environment: {shorten(location or bg, 12)}',
        f'background elements: {shorten(bg, 12)}',
        f'context props in the environment: {shorten(extra, 10)}',
        'show a layered environment with visible ground plus distant or surrounding setting details',
        'do not use a blank, plain, gray, white, or empty background',
    ]
    if weather:
        parts.append(f'weather must affect the background: {shorten(weather, 6)}')
    if atmosphere:
        parts.append(f'atmosphere should be visible in the scene: {shorten(atmosphere, 6)}')
    return '; '.join(unique(parts, 8))


def _storytelling_contract(frame_index: int, total_frames: int, sentence: str, action: str, previous_hint: str = '', next_hint: str = '') -> str:
    stage = _dcee_stage(frame_index, total_frames)
    parts = [
        f'DCEE stage: {stage}',
        f'this frame must visualize the exact current story step: {shorten(sentence, 14)}',
        f'make the protagonist clearly do this story action: {shorten(action or sentence, 12)}',
        'the frame must read like one panel of a progressing visual story, not an isolated portrait',
        'the image should show a meaningful change or progression from previous frames while preserving protagonist identity',
        'foreground action, evidence, and background should work together to tell what is happening now',
        'this frame must represent the current step distinctly, not repeat the previous frame and not jump ahead to the next frame',
    ]
    if previous_hint:
        parts.append(f'continue after the previous moment: {shorten(previous_hint, 10)}')
    if next_hint:
        parts.append(f'prepare naturally for the next moment without skipping to it: {shorten(next_hint, 10)}')
    return '; '.join(unique(parts, 9))



def _critical_visual_nouns(sentence: str, location: str, required_objects: List[str], scene_elements: List[str]) -> List[str]:
    text = ' '.join([
        clean(sentence),
        clean(location),
        ', '.join(as_list(required_objects)),
        ', '.join(as_list(scene_elements)),
    ]).lower()
    out: List[str] = []
    rules = [
        (['lost jar', 'honey jar', 'jar'], 'lost honey jar'),
        (['honey'], 'honey'),
        (['tangled roots', 'tree roots', 'roots', 'root'], 'tangled roots'),
        (['steep slope', 'slope', 'incline', 'hill', 'hillside'], 'steep slope'),
        (['serene lake', 'lake', 'lakeshore', 'shore', 'shoreline'], 'serene lake'),
        (['deep forest', 'dense forest', 'forest', 'woods', 'trees', 'underbrush'], 'dense forest'),
        (['trail', 'path'], 'forest path'),
        (['branch', 'branches', 'fallen branches'], 'fallen branches'),
        (['bush'], 'bush'),
        (['water', 'water edge', "water's edge"], 'water edge'),
    ]
    for keys, label in rules:
        if any(k in text for k in keys):
            out.append(label)
    for item in as_list(required_objects) + as_list(scene_elements) + [location]:
        s = clean(item).lower()
        if not s:
            continue
        if s in {'protagonist', 'subject', 'bear', 'white bear', 'panda'}:
            continue
        if _looks_like_action_or_generic(s):
            continue
        if s not in out:
            out.append(s)
    return unique(out, 8)
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
    identity_lock: str = ''
    dcee_appearance_delta: str = ''
    scene_contract: str = ''
    storytelling_contract: str = ''
    background_contract: str = ''
    critical_visual_nouns: List[str] = field(default_factory=list)
    previous_story_hint: str = ''
    next_story_hint: str = ''
    scene_elements: List[str] = field(default_factory=list)
    foreground_elements: List[str] = field(default_factory=list)
    scene_summary: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _visualize_event_action(sentence: str, event: str, evidence: Any = None) -> str:
    ev = clean(event).lower()
    sent = clean(sentence).lower()
    evid = ' '.join(as_list(evidence)).lower()
    text = f"{ev} {sent} {evid}"
    if any(k in text for k in ['hear', 'hears', 'listen', 'sound of water']):
        return 'pauses and reacts toward nearby visible flowing water'
    if any(k in text for k in ['follow', 'follows']):
        return 'walks along and follows the visible flowing water'
    if any(k in text for k in ['search', 'looking for', 'look for']):
        return 'searches the surroundings for the lost object'
    if any(k in text for k in ['arrive', 'arrives']):
        return 'reaches the place and notices the important object'
    if any(k in text for k in ['retrieve', 'retrieves', 'pick up', 'recover']):
        return 'picks up and holds the recovered object'
    if any(k in text for k in ['enter', 'enters']):
        return 'walks into the scene and begins the search'
    if clean(event):
        return clean(event)
    return clean(sentence)


def _infer_action(sentence: str, event: str) -> str:
    return _visualize_event_action(sentence, event)


def _forbidden_for_subject(protagonist: str) -> List[str]:
    p = protagonist.lower()
    base = [
        'text', 'watermark', 'logo', 'duplicate protagonist', 'unrelated extra people',
        'wrong age', 'wrong gender', 'child version', 'baby version', 'juvenile version',
        'completely different outfit', 'generic portrait only', 'empty background', 'missing props', 'cropped feet', 'cropped face', 'cropped paws', 'extra character', 'second subject'
    ]
    if 'panda' in p or 'bear' in p:
        base += ['human protagonist', 'human face replacing panda', 'panda turning into human', 'human instead of panda', 'another bear', 'two bears', 'bear with person', 'mirror duplicate', 'reflection as second bear']
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
        anchor = clean(getattr(prof, 'identity_anchor_prompt', ''))
        if anchor:
            identity_parts.append(anchor)
        face = clean(getattr(prof, 'face', ''))
        body = clean(getattr(prof, 'body', ''))
        if face:
            identity_parts.append(face)
        if body:
            identity_parts.append(body)
        sig = as_list(getattr(prof, 'signature_items', []))
        if sig:
            identity_parts.extend(sig[:2])
    raw_subject_identity = '; '.join(identity_parts) or protagonist
    subject_identity = _compact_subject_identity(protagonist, raw_subject_identity)
    if isinstance(story_bible, dict) and story_bible.get('subject_identity_prompt'):
        subject_identity = _compact_subject_identity(protagonist, story_bible.get('subject_identity_prompt') + '; ' + subject_identity)

    required_objects = _filter_visual_inventory(unique(
        as_list(story_row.get('required_objects', []))
        + as_list(story_row.get('required_objects_en', []))
        + as_list(story_row.get('object', ''))
        + as_list(getattr(frame, 'key_objects', []))
        + as_list(getattr(frame, 'evidence_objects', []))
        + as_list(getattr(frame, 'emotion_evidence', []))
        + as_list(getattr(frame, 'must_show', []))
        + _strong_story_required_objects(story_sentence, protagonist, as_list(getattr(seed, 'signature_items', [])) or as_list(getattr(getattr(seed, 'character_profiles', [None])[0], 'signature_items', []))),
        16,
    ), protagonist, 12)

    emotion = clean(story_row.get('emotion')) or clean(getattr(frame, 'emotion', ''))
    bible_emotion = emotion_cue_from_bible(story_bible, emotion) if story_bible else {}
    event = clean(story_row.get('action') or story_row.get('event')) or clean(getattr(frame, 'event', ''))
    event_grounding = clean(story_row.get('visible_cause') or story_row.get('action')) or clean(getattr(frame, 'event_grounding', ''))
    visual_action = _visualize_event_action(story_sentence, event or story_sentence, as_list(getattr(frame, 'must_show', [])) + as_list(getattr(frame, 'evidence_objects', [])))
    visual_focus = clean(getattr(frame, 'visual_focus', ''))

    camera = (
        clean(getattr(frame, 'camera_shot', ''))
        or clean(getattr(frame, 'shot_type', ''))
        or 'medium story shot showing face, action, key objects, and background'
    )

    carry_over = unique(as_list(graph_hints.get('carry_over_entities', [])), 6)
    recurring = unique(as_list(graph_hints.get('recurring_entities', [])), 6)
    location = (
        clean(story_row.get('location'))
        or clean(getattr(frame, 'scene_location', ''))
        or clean(graph_hints.get('location'))
        or _infer_scene_location_from_text(story_sentence)
        or clean((bible_world or {}).get('fixed_setting'))
        or clean(getattr(seed, 'setting', ''))
    )
    weather = clean(story_row.get('weather')) or clean(getattr(frame, 'weather', '')) or clean(graph_hints.get('weather')) or clean((bible_world or {}).get('weather'))
    atmosphere = clean(story_row.get('atmosphere')) or clean(getattr(frame, 'atmosphere', '')) or clean(graph_hints.get('atmosphere')) or clean((bible_world or {}).get('atmosphere'))

    previous_hint = clean(story_row.get('previous_sentence')) or (clean(rows[frame_index - 1].get('sentence', '')) if frame_index > 0 and frame_index - 1 < len(rows) and isinstance(rows[frame_index - 1], dict) else '')
    next_hint = clean(story_row.get('next_sentence')) or (clean(rows[frame_index + 1].get('sentence', '')) if frame_index + 1 < len(rows) and isinstance(rows[frame_index + 1], dict) else '')

    continuity_parts = [
        f'frame {frame_index + 1}/{total_frames}',
        'visualize only this story sentence',
        'same protagonist identity across all frames',
        'advance the story instead of repeating the previous pose',
        'the current frame must visually realize the current caption more strongly than any earlier frame',
        'show a complete single scene rather than a generic portrait',
        'make invisible or abstract events visually explicit through pose, object, and scene evidence',
        'preserve a full colorful environment that matches the caption, not a gray empty background',
        'show the current step clearly between the previous and next moments',
    ]
    if carry_over:
        continuity_parts.append('carry over entities: ' + ', '.join(carry_over))
    if recurring:
        continuity_parts.append('recurring story entities: ' + ', '.join(recurring[:3]))
    continuity = '; '.join(continuity_parts)

    scene_elements = _extract_scene_elements(story_sentence, story_row, frame, bible_world)
    critical_visual_nouns = _critical_visual_nouns(story_sentence, location, required_objects, scene_elements)
    identity_lock = _identity_lock_from_sources(protagonist, raw_subject_identity, story_bible, reference_image_path)
    facial_expression_value = bible_emotion.get('face') or clean(story_row.get('facial_cue')) or clean(getattr(frame, 'facial_cue', '')) or f'clear facial expression of {emotion}'
    body_pose_value = bible_emotion.get('body') or clean(story_row.get('body_cue')) or clean(getattr(frame, 'body_cue', '')) or f'body pose clearly showing {emotion} while doing the action'
    dcee_delta = _dcee_appearance_delta(frame_index, total_frames, emotion, facial_expression_value, body_pose_value, weather)
    scene_contract = _scene_contract(story_sentence, visual_action, event_grounding or visual_action, location, weather, atmosphere, required_objects, scene_elements)
    background_contract = _background_contract(story_sentence, location, weather, atmosphere, scene_elements, required_objects)
    storytelling_contract = _storytelling_contract(frame_index, total_frames, story_sentence, visual_action, previous_hint, next_hint)
    foreground_elements = unique([visual_action, event_grounding] + required_objects[:4], 6)
    scene_summary = clean(
        f"{story_sentence}. Scene: {location or 'story environment'}. "
        f"Main action: {visual_action or event or story_sentence}. "
        f"Important visible props: {', '.join(required_objects[:4]) or 'story props'}. "
        f"Background details: {', '.join(scene_elements[:4]) or (location or 'story background')}. "
        f"Emotion: {emotion or 'current emotion'}."
    )

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
        primary_action=visual_action,
        visible_event=visual_action or event or story_sentence,
        visible_cause=event_grounding or visual_action or story_sentence,
        required_objects=required_objects,
        carry_over_entities=carry_over,
        recurring_entities=recurring,
        forbidden_objects=_forbidden_for_subject(protagonist),
        location=location,
        weather=weather,
        atmosphere=atmosphere,
        emotion=emotion,
        facial_expression=facial_expression_value,
        body_pose=body_pose_value,
        camera=camera + (f'; visual focus: {visual_focus}' if visual_focus else ''),
        continuity=continuity,
        negative=', '.join(unique(_forbidden_for_subject(protagonist) + as_list((story_bible or {}).get('global_negative', [])), 60)),
        identity_lock=identity_lock,
        dcee_appearance_delta=dcee_delta,
        scene_contract=scene_contract,
        storytelling_contract=storytelling_contract,
        background_contract=background_contract,
        critical_visual_nouns=critical_visual_nouns,
        previous_story_hint=previous_hint,
        next_story_hint=next_hint,
        scene_elements=scene_elements,
        foreground_elements=foreground_elements,
        scene_summary=scene_summary,
    )


def prompt_from_spec(spec: FrameVisualSpec, mode: str = 'caption_locked') -> str:
    obj = ', '.join(unique(spec.required_objects, 5)) or 'grounded story objects only'
    sentence = shorten(spec.story_sentence, 28)
    action = shorten(spec.primary_action or spec.visible_event, 10)
    identity = shorten(spec.subject_identity, 12)
    loc = shorten(spec.location, 8)
    emotion = shorten(spec.emotion, 4)
    critical = shorten(', '.join(unique(getattr(spec, 'critical_visual_nouns', []), 6)), 18)
    scene_summary = shorten(getattr(spec, 'scene_summary', ''), 40)
    return clean(
        f"full-color cinematic storybook illustration. exact story sentence: {sentence}. "
        f"exactly one protagonist: {spec.protagonist}; identity: {identity}. "
        f"main action: {action}. required visible props: {obj}. critical visible nouns: {critical}. "
        f"scene summary: {scene_summary}. location: {loc}. emotion: {emotion}. "
        f"show a complete medium-wide story scene with visible foreground, midground, and background. "
        f"show the full body and the environment; do not crop the protagonist. "
        f"no extra people, no extra animals, no duplicate protagonist."
    )


def negative_from_spec(spec: FrameVisualSpec) -> str:

    return clean(
        'split screen, multi panel, collage, comic page, multiple scenes, generic portrait, repeated static pose, unrelated poster image, '
        'missing protagonist, wrong protagonist, changed species, changed face, changed fur pattern, changed body proportions, baby animal, cub, juvenile, childlike body, malformed hands, malformed paws, missing hands, missing paws, extra limbs, '
        'different identity, inconsistent character, missing action, missing event, missing evidence, missing required object, missing critical noun, cropped out props, cropped feet, cropped paws, cropped face, '
        'wrong background, wrong weather, gray empty background, monochrome background, unrelated humans, person, man, woman, child, girl, boy, extra animal, duplicate characters, second protagonist, companion character, sidekick, '
        'unrelated object, unrelated prop, text, watermark, low quality, blurry, '
        + spec.negative
    )
