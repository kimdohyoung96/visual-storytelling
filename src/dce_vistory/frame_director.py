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
    for item in items:
        item = clean(item)
        if item and item not in out:
            out.append(item)
    return out if limit is None else out[:limit]


_ABSTRACT = {
    "desire", "fear", "loss", "journey", "emotion", "feeling", "feelings", "reflection", "bittersweet",
    "nature", "heavy heart", "honesty", "importance", "acceptance", "cause", "evidence", "need to", "the desire",
    "the fear", "the loss", "quietly on the", "water with a", "staring", "heavy", "sits", "quietly", "white", "bear"
}

_STOP = {
    "the", "a", "an", "his", "her", "their", "with", "without", "into", "onto", "under", "over", "near",
    "while", "where", "there", "here", "quietly", "gently", "slowly", "sadly", "eagerly", "anxiously",
    "staring", "showing", "reflecting", "watching", "feeling", "heavy", "heart", "face", "body", "emotion"
}


def is_visual_object(item: str, protagonist: str = "") -> bool:
    s = clean(item)
    low = s.lower()
    if not s:
        return False
    if low == clean(protagonist).lower():
        return True
    if low in _ABSTRACT or any(a in low for a in _ABSTRACT):
        return False
    toks = low.split()
    if len(toks) > 4:
        return False
    if all(t in _STOP for t in toks):
        return False
    return True


def shorten(text: str, max_words: int = 20) -> str:
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


def _infer_action(sentence: str, event: str) -> str:
    return clean(event) or clean(sentence)


def _identity_parts(seed: Any, frame: Any) -> str:
    parts: List[str] = []
    profiles = getattr(seed, "character_profiles", []) or []
    if profiles:
        prof = profiles[0]
        for key in ["name", "role", "species", "fur_color", "age_group", "gender", "face", "body", "outfit", "distinguishing_traits", "identity_anchor_prompt"]:
            val = clean(getattr(prof, key, ""))
            if val:
                parts.append(f"{key}: {val}")
        sig = as_list(getattr(prof, "signature_items", []))
        if sig:
            parts.append("signature items: " + ", ".join(sig[:3]))
    fallback = clean(getattr(seed, "protagonist_identity_prompt", "")) or clean(getattr(frame, "character_reference_prompt", ""))
    if fallback and fallback not in parts:
        parts.append(fallback)
    return "; ".join(parts) or clean(getattr(seed, "protagonist", "protagonist"))


def _forbidden_for_subject(protagonist: str, seed: Any = None, frame: Any = None) -> List[str]:
    p = protagonist.lower()
    base = [
        "text", "watermark", "logo", "split screen", "multi panel", "comic panel", "storyboard sheet", "collage",
        "duplicate protagonist", "second protagonist", "two protagonists", "two bears", "multiple bears", "extra bear",
        "unrelated extra people", "extra character", "extra animal", "wrong age", "wrong gender", "child version", "baby version", "juvenile version",
        "completely different outfit", "generic portrait only", "empty background", "missing props", "wrong species", "wrong fur color", "wrong coat color"
    ]
    neg = clean(getattr(seed, "protagonist_negative_identity_prompt", "")) if seed is not None else ""
    if neg:
        base += as_list(neg)
    if "bear" in p:
        base += ["human protagonist", "human face", "panda", "brown bear" if "white" in p else "white bear" if "brown" in p else "wrong bear color"]
    return unique(base, 80)


def _required_objects(frame: Any, story_row: Dict[str, Any], protagonist: str, location: str) -> List[str]:
    items: List[str] = []
    # Priority order: planner-produced visual inventory first, then concrete storyboard fields.
    items += as_list(getattr(frame, "must_show", []))
    items += as_list(story_row.get("required_objects", []))
    items += as_list(story_row.get("object", ""))
    items += as_list(getattr(frame, "key_objects", []))
    items += as_list(getattr(frame, "evidence_objects", []))
    items += as_list(location)
    items += as_list(getattr(frame, "environment_details", []))
    cleaned = [x for x in items if is_visual_object(x, protagonist)]
    if protagonist and protagonist not in cleaned:
        cleaned.insert(0, protagonist)
    return unique(cleaned, 8)


def build_frame_visual_spec(frame: Any, seed: Any, full_story: Dict[str, Any] | None, frame_index: int, total_frames: int, reference_image_path: str = "") -> FrameVisualSpec:
    rows = (full_story or {}).get("sentences", []) if isinstance(full_story, dict) else []
    story_row = rows[frame_index] if frame_index < len(rows) and isinstance(rows[frame_index], dict) else {}
    story_sentence = clean(getattr(frame, "image_sentence", "")) or clean(story_row.get("image_sentence") or story_row.get("sentence"))
    story_sentence = story_sentence or clean(getattr(frame, "story_sentence", "")) or clean(getattr(frame, "caption", ""))

    protagonist = clean(getattr(seed, "protagonist_visual_short", "")) or clean(getattr(seed, "protagonist", "")) or clean(getattr(frame, "identity_short", "")) or "protagonist"
    subject_identity = _identity_parts(seed, frame)
    event = clean(story_row.get("action") or story_row.get("event")) or clean(getattr(frame, "event", ""))
    event_grounding = clean(story_row.get("visible_cause") or story_row.get("action")) or clean(getattr(frame, "event_grounding", ""))
    location = clean(story_row.get("location")) or clean(getattr(frame, "scene_location", "")) or clean(getattr(seed, "setting", ""))
    weather = clean(story_row.get("weather")) or clean(getattr(frame, "weather", ""))
    atmosphere = clean(story_row.get("atmosphere")) or clean(getattr(frame, "atmosphere", ""))
    emotion = clean(story_row.get("emotion")) or clean(getattr(frame, "emotion", ""))
    visual_focus = clean(getattr(frame, "visual_focus", ""))
    camera = clean(getattr(frame, "camera_shot", "")) or clean(getattr(frame, "shot_type", "")) or "medium story shot"
    required = _required_objects(frame, story_row, protagonist, location)

    continuity = "; ".join([
        f"frame {frame_index + 1}/{total_frames}",
        "one single coherent scene only",
        "exactly one protagonist only",
        "same protagonist identity across all frames",
        "advance the story instead of repeating the previous pose",
    ])
    neg = _forbidden_for_subject(protagonist, seed, frame)
    return FrameVisualSpec(
        frame_id=int(getattr(frame, "frame_id", frame_index + 1)),
        total_frames=total_frames,
        story_sentence=story_sentence,
        protagonist=protagonist,
        subject_identity=subject_identity,
        subject_reference_policy=(f"use the input reference image as the identity anchor: {reference_image_path}" if reference_image_path else "use text identity strictly"),
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


def prompt_from_spec(spec: FrameVisualSpec, mode: str = "sentence_locked") -> str:
    obj = ", ".join(unique(spec.required_objects, 8))
    identity = shorten(spec.subject_identity, 45)
    sentence = shorten(spec.story_sentence, 32)
    action = shorten(spec.primary_action or spec.visible_event, 18)
    cause = shorten(spec.visible_cause, 18)
    loc = shorten(spec.location, 10)
    weather = shorten(spec.weather, 5)
    atmosphere = shorten(spec.atmosphere, 8)
    face = shorten(spec.facial_expression, 12)
    body = shorten(spec.body_pose, 14)
    camera = shorten(spec.camera, 14)
    base = (
        f"full-color cinematic storybook illustration. frame {spec.frame_id}/{spec.total_frames}. "
        f"ONE SINGLE COHERENT SCENE, NOT split-screen, NOT comic panels. "
        f"exactly one {spec.protagonist}; no second {spec.protagonist}, no duplicate protagonist, no extra characters. "
        f"exact story sentence: {sentence}. "
        f"same protagonist identity: {identity}. "
        f"scene {loc}; weather {weather}; mood {atmosphere}. "
        f"action {action}. visible cause {cause}. "
        f"only show required visual inventory: {obj}. "
        f"emotion {spec.emotion}; face {face}; body {body}. camera {camera}. "
    )
    if mode == "object_locked":
        extra = "show the required visual inventory clearly, but do not add anything outside the list."
    elif mode == "continuity_locked":
        extra = "keep the same identity and world continuity while rendering only this new story beat."
    elif mode == "background_locked":
        extra = "background must support the story sentence, but the protagonist action must remain primary."
    elif mode == "emotion_locked":
        extra = f"the viewer must understand why the protagonist feels {spec.emotion} from the action and scene."
    else:
        extra = "sentence locking is mandatory: visualize this exact narrative beat, not a generic portrait."
    return clean(base + " " + extra)


def negative_from_spec(spec: FrameVisualSpec) -> str:
    return clean(
        "split screen, diptych, triptych, comic panel, storyboard sheet, collage, multiple scenes in one image, "
        "duplicate protagonist, second protagonist, two bears, multiple bears, extra bear, extra character, extra animal, "
        "generic portrait, repeated static pose, unrelated poster image, missing protagonist, wrong protagonist, wrong species, wrong fur color, "
        "baby animal, cub, juvenile, childlike body, different identity, inconsistent character, missing action, missing event, missing evidence, "
        "missing required object, cropped out props, wrong background, wrong weather, unrelated humans, text, watermark, low quality, blurry, "
        + spec.negative
    )
