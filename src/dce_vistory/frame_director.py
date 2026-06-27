from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List
import re
from .story_bible import emotion_cue_from_bible, world_for_frame


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
    base = ["text", "watermark", "logo", "duplicate protagonist", "unrelated extra people"]
    if "panda" in p or "bear" in p:
        base += [
            "human protagonist",
            "human face replacing panda",
            "panda turning into human",
            "human instead of panda",
            "extra human unless story sentence explicitly requires a woodcutter",
        ]
    if "woodcutter" in p:
        base += ["panda protagonist", "animal protagonist", "child version", "different man"]
    return base


def build_frame_visual_spec(frame: Any, seed: Any, full_story: Dict[str, Any] | None, frame_index: int, total_frames: int, reference_image_path: str = "") -> FrameVisualSpec:
    rows = (full_story or {}).get("sentences", []) if isinstance(full_story, dict) else []
    story_row = rows[frame_index] if frame_index < len(rows) and isinstance(rows[frame_index], dict) else {}
    story_sentence = clean(story_row.get("image_sentence") or story_row.get("sentence"))
    story_sentence = story_sentence or clean(getattr(frame, "story_sentence", "")) or clean(getattr(frame, "caption", ""))

    # Story Bible is attached to seed by pipeline_crossattn_butterfly.py.
    # Keep a safe empty dict when it is not available, so the frame director remains usable.
    story_bible = getattr(seed, "_story_bible", {}) or {}
    if not isinstance(story_bible, dict):
        story_bible = {}
    bible_world = world_for_frame(story_bible, frame) if story_bible else {}

    protagonist = clean(getattr(seed, "protagonist", "")) or clean(getattr(frame, "protagonist", "")) or "protagonist"

    identity_parts = []
    profiles = getattr(seed, "character_profiles", []) or []
    if profiles:
        prof = profiles[0]
        for key in ["name", "role", "age_group", "gender", "face", "hair", "body", "outfit", "identity_anchor_prompt"]:
            val = clean(getattr(prof, key, ""))
            if val:
                identity_parts.append(f"{key}: {val}")
        sig = as_list(getattr(prof, "signature_items", []))
        if sig:
            identity_parts.append("signature items: " + ", ".join(sig))
    subject_identity = "; ".join(identity_parts) or protagonist
    if isinstance(story_bible, dict) and story_bible.get("subject_identity_prompt"):
        subject_identity = story_bible.get("subject_identity_prompt") + "; " + subject_identity

    required_objects = unique(
        as_list(story_row.get("required_objects", []))
        + as_list(story_row.get("subject", ""))
        + as_list(story_row.get("object", ""))
        + as_list(getattr(frame, "key_objects", []))
        + as_list(getattr(frame, "evidence_objects", []))
        + as_list((bible_world or {}).get("stable_background", [])),
        10,
    )

    emotion = clean(story_row.get("emotion")) or clean(getattr(frame, "emotion", ""))
    bible_emotion = emotion_cue_from_bible(story_bible, emotion) if story_bible else {}
    event = clean(story_row.get("action")) or clean(getattr(frame, "event", ""))
    event_grounding = clean(story_row.get("visible_cause") or story_row.get("action")) or clean(getattr(frame, "event_grounding", ""))
    visual_focus = clean(getattr(frame, "visual_focus", ""))

    camera = (
        clean(getattr(frame, "camera_shot", ""))
        or clean(getattr(frame, "shot_type", ""))
        or "cinematic medium-wide story frame showing subject, action, key objects, and background"
    )

    continuity = (
        f"Frame {frame_index + 1} of {total_frames}; must visualize sentence {frame_index + 1} only; "
        "same protagonist identity across all frames; advance the story without repeating the previous frame composition."
    )

    return FrameVisualSpec(
        frame_id=int(getattr(frame, "frame_id", frame_index + 1)),
        total_frames=total_frames,
        story_sentence=story_sentence,
        protagonist=protagonist,
        subject_identity=subject_identity,
        subject_reference_policy=(
            f"Use reference image only for protagonist identity and appearance, not for pose or background: {reference_image_path}"
            if reference_image_path else
            "No external reference image available; use subject_identity text strictly."
        ),
        primary_action=_infer_action(story_sentence, event),
        visible_event=event,
        visible_cause=event_grounding or story_sentence,
        required_objects=required_objects,
        forbidden_objects=_forbidden_for_subject(protagonist),
        location=(bible_world or {}).get("fixed_setting") or clean(story_row.get("location")) or clean(getattr(frame, "scene_location", "")) or clean(getattr(seed, "setting", "")),
        weather=(bible_world or {}).get("weather") or clean(getattr(frame, "weather", "")),
        atmosphere=(bible_world or {}).get("atmosphere") or clean(getattr(frame, "atmosphere", "")),
        emotion=emotion,
        facial_expression=bible_emotion.get("face") or clean(story_row.get("facial_cue")) or clean(getattr(frame, "facial_cue", "")) or f"facial expression clearly showing {emotion}",
        body_pose=bible_emotion.get("body") or clean(story_row.get("body_cue")) or clean(getattr(frame, "body_cue", "")) or f"body pose clearly showing {emotion} while doing the action",
        camera=camera + (f"; visual focus: {visual_focus}" if visual_focus else ""),
        continuity=continuity,
        negative=", ".join(unique(_forbidden_for_subject(protagonist) + as_list((story_bible or {}).get("global_negative", [])), 50)),
    )


def prompt_from_spec(spec: FrameVisualSpec, mode: str = "sentence_locked") -> str:
    objects = ", ".join(spec.required_objects)
    forbidden = ", ".join(spec.forbidden_objects)
    common = (
        f"FRAME {spec.frame_id} OF {spec.total_frames}. "
        f"Create one full-color cinematic storybook illustration that visualizes exactly this sentence: \"{spec.story_sentence}\". "
        f"STRICT CHARACTER LOCK: {spec.subject_identity}. {spec.subject_reference_policy}. "
        f"The protagonist must be the same ADULT character in every frame; never a baby, cub, child, or different subject. "
        f"Primary visible action: {spec.primary_action}. Visible event: {spec.visible_event}. "
        f"Visible cause of emotion: {spec.visible_cause}. Must show these objects clearly: {objects}. "
        f"STRICT WORLD LOCK: Location/background: {spec.location}. Weather: {spec.weather}. Atmosphere: {spec.atmosphere}. "
        f"The background must visibly contain the riverbank/bamboo forest context when listed in required objects. "
        f"STRICT EMOTION LOCK: Emotion: {spec.emotion}; face must show {spec.facial_expression}; body must show {spec.body_pose}. "
        f"Use a medium shot or medium-close shot where the face, paws, action object, and background are all visible. "
        f"Camera/composition: {spec.camera}. Continuity: {spec.continuity}. "
        f"Do not draw: {forbidden}. "
    )
    if mode == "action":
        extra = "Prioritize the physical action and interaction between subject and objects. The action must be readable without captions."
    elif mode == "objects":
        extra = "Prioritize the required objects and evidence. All required objects must be visible inside the frame, not cropped out."
    elif mode == "background":
        extra = "Prioritize correct background, location, weather, and spatial relationship between protagonist and objects."
    elif mode == "emotion":
        extra = "Prioritize the emotional cause: the viewer must understand why the protagonist feels this emotion by looking at the scene."
    else:
        extra = "Sentence locking is mandatory: the image must match this sentence, not a generic portrait or unrelated scene."
    return common + extra


def negative_from_spec(spec: FrameVisualSpec) -> str:
    return (
        "generic portrait, repeated static pose, unrelated poster image, missing protagonist, wrong protagonist, baby animal, cub, juvenile, childlike body, "
        "different identity, inconsistent character, missing action, missing event, missing evidence, missing required object, "
        "cropped out hands or props, wrong background, wrong weather, unrelated humans, duplicate characters, "
        + spec.negative
        + ", text, watermark, low quality, blurry"
    )
