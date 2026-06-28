from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List
import re


def clean(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "").replace("\n", " ")).strip()


def as_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return [clean(v) for v in x if clean(v)]
    if isinstance(x, dict):
        out: List[str] = []
        for v in x.values():
            out.extend(as_list(v))
        return out
    s = clean(x)
    return [s] if s else []


def unique(items: List[str], limit: int | None = None) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        item = clean(item)
        key = item.lower()
        if item and key not in seen:
            out.append(item)
            seen.add(key)
    return out if limit is None else out[:limit]


_ACTION_WORDS = {
    "enter", "enters", "stand", "stands", "standing", "sit", "sits", "sitting", "walk", "walks", "walking",
    "step", "steps", "stepping", "gasp", "gasps", "gasping", "stare", "stares", "staring", "watch", "watches",
    "watching", "hold", "holds", "holding", "clutch", "clutches", "clutching", "reach", "reaches", "reaching",
    "look", "looks", "looking", "cry", "cries", "crying", "run", "runs", "running", "chase", "chases",
    "chasing", "breathe", "breath", "breathing", "kneel", "kneels", "kneeling"
}

_STOP = {
    "the", "a", "an", "his", "her", "their", "with", "without", "into", "onto", "under", "over", "near",
    "while", "where", "there", "here", "of", "to", "from", "and", "or", "in", "on", "at", "by", "for",
    "quietly", "gently", "slowly", "sadly", "eagerly", "anxiously", "resolutely", "frantically",
    "showing", "reflecting", "feeling", "emotion", "cause", "evidence", "need", "importance"
}

_ABSTRACT_PHRASES = {
    "heavy heart", "the desire", "the fear", "the loss", "loss of", "the need to", "importance",
    "acceptance", "honesty", "feelings", "emotion", "journey", "bittersweet", "reflective",
    "quietly on the", "water with a", "staring at the", "the current", "the moment"
}

_VISUAL_NOUN_HINTS = {
    "bear", "panda", "jar", "honey", "river", "riverbank", "water", "forest", "tree", "trees", "pine",
    "fir", "bamboo", "rock", "rocks", "stone", "mud", "path", "leaf", "leaves", "rain", "cloud", "sky",
    "snow", "ground", "grass", "hill", "shadow", "sunlight", "stream", "bank", "shore", "log"
}


def is_visual_object(item: str, protagonist: str = "") -> bool:
    s = clean(item)
    low = s.lower()
    if not s:
        return False
    protagonist_low = clean(protagonist).lower()
    if protagonist_low and low == protagonist_low:
        return True
    if any(p in low for p in _ABSTRACT_PHRASES):
        return False
    toks = [t for t in re.split(r"[^a-zA-Z0-9가-힣]+", low) if t]
    if not toks:
        return False
    if any(t in _ACTION_WORDS for t in toks):
        return False
    if all(t in _STOP for t in toks):
        return False
    if len(toks) > 4:
        # Long phrases are usually event descriptions, not drawable object labels.
        return False
    if len(toks) == 1 and toks[0] in {"white", "brown", "black", "gray", "grey", "sad", "happy", "heavy"}:
        return False
    # Keep clear visual nouns or Korean nouns; reject abstract English leftovers.
    if any(t in _VISUAL_NOUN_HINTS for t in toks):
        return True
    if re.search(r"[가-힣]", s):
        return True
    # Two-word object names like "honey jar", "empty spot".
    if len(toks) <= 3 and not any(t in _STOP for t in toks):
        return True
    return False


def shorten(text: str, max_words: int = 24) -> str:
    words = clean(text).split()
    return " ".join(words[:max_words])


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


def _identity_parts(seed: Any, frame: Any) -> str:
    parts: List[str] = []
    profiles = getattr(seed, "character_profiles", []) or []
    if profiles:
        prof = profiles[0]
        for key in [
            "name", "role", "species", "fur_color", "age_group", "gender", "face", "body",
            "outfit", "distinguishing_traits", "identity_anchor_prompt"
        ]:
            val = clean(getattr(prof, key, ""))
            if val:
                parts.append(f"{key}: {val}")
        sig = as_list(getattr(prof, "signature_items", []))
        if sig:
            parts.append("signature items: " + ", ".join(sig[:3]))
    fallback = clean(getattr(seed, "protagonist_identity_prompt", "")) or clean(getattr(frame, "character_reference_prompt", ""))
    if fallback:
        parts.append(fallback)
    return "; ".join(unique(parts, 12)) or clean(getattr(seed, "protagonist", "protagonist"))


def _forbidden_for_subject(protagonist: str, seed: Any = None, frame: Any = None) -> List[str]:
    p = protagonist.lower()
    base = [
        "text", "watermark", "logo",
        "split screen", "multi panel", "comic panel", "storyboard sheet", "collage", "diptych", "triptych",
        "duplicate protagonist", "second protagonist", "two protagonists", "multiple protagonists",
        "two bears", "multiple bears", "extra bear", "baby bear", "bear cub", "small bear", "tiny bear", "second white bear", "mirrored bear", "bear reflection as second subject",
        "extra character", "extra animal", "unrelated extra people", "unrelated humans",
        "wrong age", "wrong gender", "child version", "baby version", "juvenile version",
        "wrong species", "wrong fur color", "wrong coat color", "different identity", "standing portrait without event", "character looking at camera only",
        "generic portrait only", "empty background", "missing action", "missing event", "missing required object", "cropped head", "cropped face", "cropped feet", "cut off body", "partial body", "out of frame", "extreme close-up",
        "unrelated object", "unrelated prop", "human protagonist", "human face",
    ]
    neg = clean(getattr(seed, "protagonist_negative_identity_prompt", "")) if seed is not None else ""
    if neg:
        base += as_list(neg)
    if "white bear" in p:
        base += ["brown bear", "black bear", "panda", "gray bear"]
    elif "brown bear" in p:
        base += ["white bear", "polar bear", "panda"]
    elif "panda" in p:
        base += ["human face replacing panda", "panda turning into human", "human instead of panda", "brown bear", "white bear"]
    return unique(base, 100)


def _infer_action(sentence: str, event: str) -> str:
    return clean(event) or clean(sentence)


def _event_visual_inventory(frame: Any, story_row: Dict[str, Any], protagonist: str, location: str) -> List[str]:
    """V23: keep only concrete objects needed for this exact event."""
    items: List[str] = []

    # Current frame/event fields are the highest priority.
    items += as_list(story_row.get("required_objects", []))
    items += as_list(story_row.get("object", ""))
    items += as_list(story_row.get("background_elements", []))
    items += as_list(getattr(frame, "must_show", []))
    items += as_list(getattr(frame, "key_objects", []))
    items += as_list(getattr(frame, "evidence_objects", []))
    items += as_list(location)
    items += as_list(getattr(frame, "environment_details", []))

    cleaned = [x for x in items if is_visual_object(x, protagonist)]
    if protagonist:
        cleaned = [x for x in cleaned if clean(x).lower() != clean(protagonist).lower()]
        cleaned.insert(0, protagonist)

    # Avoid overlong inventory; large lists make SDXL add artifacts.
    return unique(cleaned, 6)


def build_frame_visual_spec(
    frame: Any,
    seed: Any,
    full_story: Dict[str, Any] | None,
    frame_index: int,
    total_frames: int,
    reference_image_path: str = "",
) -> FrameVisualSpec:
    rows = (full_story or {}).get("sentences", []) if isinstance(full_story, dict) else []
    story_row = rows[frame_index] if frame_index < len(rows) and isinstance(rows[frame_index], dict) else {}

    story_sentence = clean(getattr(frame, "image_sentence", "")) or clean(story_row.get("image_sentence") or story_row.get("sentence"))
    story_sentence = story_sentence or clean(getattr(frame, "story_sentence", "")) or clean(getattr(frame, "caption", ""))

    protagonist = clean(getattr(seed, "protagonist_visual_short", "")) or clean(getattr(seed, "protagonist", "")) or clean(getattr(frame, "identity_short", "")) or "protagonist"
    subject_identity = _identity_parts(seed, frame)

    event = clean(story_row.get("action") or story_row.get("event")) or clean(getattr(frame, "event", ""))
    event_grounding = clean(story_row.get("visible_cause") or story_row.get("event_grounding")) or clean(getattr(frame, "event_grounding", ""))
    location = clean(story_row.get("location")) or clean(getattr(frame, "scene_location", "")) or clean(getattr(seed, "setting", ""))
    weather = clean(story_row.get("weather")) or clean(getattr(frame, "weather", ""))
    atmosphere = clean(story_row.get("atmosphere")) or clean(getattr(frame, "atmosphere", ""))
    emotion = clean(story_row.get("emotion")) or clean(getattr(frame, "emotion", ""))
    visual_focus = clean(getattr(frame, "visual_focus", ""))
    camera = clean(getattr(frame, "camera_shot", "")) or clean(getattr(frame, "shot_type", "")) or "medium story shot"

    required = _event_visual_inventory(frame, story_row, protagonist, location)
    neg = _forbidden_for_subject(protagonist, seed, frame)

    prev_sentence = clean(getattr(frame, "previous_story_sentence", ""))
    prev_image_summary = clean(getattr(frame, "previous_image_summary", ""))
    continuity = "; ".join([
        f"frame {frame_index + 1}/{total_frames}",
        "one single coherent scene only",
        "exactly one protagonist only",
        "same protagonist identity across all frames",
        "follow the current event, not a generic portrait",
        (f"previous story sentence: {prev_sentence}" if prev_sentence else ""),
        (f"previous selected image summary: {prev_image_summary}" if prev_image_summary else ""),
    ])

    return FrameVisualSpec(
        frame_id=int(getattr(frame, "frame_id", frame_index + 1)),
        total_frames=total_frames,
        story_sentence=story_sentence,
        protagonist=protagonist,
        subject_identity=subject_identity,
        subject_reference_policy=(f"use the input reference image as identity anchor: {reference_image_path}" if reference_image_path else "use text identity strictly"),
        primary_action=_infer_action(story_sentence, event),
        visible_event=event or story_sentence,
        visible_cause=event_grounding or story_sentence,
        required_objects=required,
        carry_over_entities=[],
        recurring_entities=[],
        forbidden_objects=neg,
        location=location,
        weather=weather,
        atmosphere=atmosphere,
        emotion=emotion,
        facial_expression=clean(story_row.get("facial_cue")) or clean(getattr(frame, "facial_cue", "")) or f"clear facial expression of {emotion}",
        body_pose=clean(story_row.get("body_cue")) or clean(getattr(frame, "body_cue", "")) or f"body pose clearly showing {emotion} while doing the action",
        camera=camera + (f"; visual focus: {visual_focus}" if visual_focus else ""),
        continuity=continuity,
        negative=", ".join(neg),
    )


def prompt_from_spec(spec: FrameVisualSpec, mode: str = "event_locked") -> str:
    obj = ", ".join(unique(spec.required_objects, 8))
    identity = shorten(spec.subject_identity, 55)
    sentence = shorten(spec.story_sentence, 40)
    action = shorten(spec.primary_action or spec.visible_event, 24)
    cause = shorten(spec.visible_cause, 24)
    loc = shorten(spec.location, 14)
    weather = shorten(spec.weather, 8)
    atmosphere = shorten(spec.atmosphere, 12)
    face = shorten(spec.facial_expression, 16)
    body = shorten(spec.body_pose, 18)
    camera = shorten(spec.camera, 18)
    continuity = shorten(spec.continuity, 50)

    base = (
        f"full-color cinematic storybook illustration. frame {spec.frame_id}/{spec.total_frames}. "
        f"Generate one single coherent scene only. "
        f"Show exactly one {spec.protagonist}; no second protagonist and no extra characters. "
        f"Keep the same protagonist identity as the input image: {identity}. "
        f"Current story sentence: {sentence}. "
        f"Main visible action: {action}. "
        f"Visible cause or evidence: {cause}. "
        f"Allowed grounded inventory only: {obj}. "
        f"Location: {loc}; weather: {weather}; atmosphere: {atmosphere}. "
        f"Emotion: {spec.emotion}; face: {face}; body pose: {body}. "
        f"Camera/composition: {camera}. "
        f"Continuity guidance: {continuity}. "
        f"Prefer a readable medium or medium-wide story shot that clearly shows the protagonist, action, and evidence. "
        f"Do not invent unrelated props or background objects. "
        f"Do not turn continuity into duplicated subjects. "
    )

    if mode == "event_locked":
        extra = "Prioritize the current event and the required grounded objects."
    elif mode == "evidence_locked":
        extra = "Make the causal evidence clearly visible and easy to understand."
    elif mode == "emotion_causal_locked":
        extra = f"Make the viewer understand why the protagonist feels {spec.emotion} from the scene."
    elif mode == "continuity_locked":
        extra = "Keep identity and world continuity, but the pose and action must change to the current story step."
    else:
        extra = "Keep the frame concrete, grounded, and easy to read."
    return clean(base + " " + extra)


def negative_from_spec(spec: FrameVisualSpec) -> str:
    return clean(
        "split screen, diptych, triptych, comic panel, storyboard sheet, collage, multiple scenes in one image, "
        "duplicate protagonist, second protagonist, extra character, extra animal, unrelated humans, human face, "
        "wrong protagonist identity, wrong species, wrong fur color, different identity, "
        "generic portrait, repeated static pose, missing protagonist, missing action, missing event, missing evidence, "
        "missing required object, unrelated object, unrelated prop, text, watermark, low quality, blurry, "
        + spec.negative
    )
