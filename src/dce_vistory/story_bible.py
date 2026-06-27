from __future__ import annotations
from typing import Any, Dict, List
import re


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


def _profile_dict(seed: Any) -> Dict[str, Any]:
    profiles = getattr(seed, "character_profiles", []) or []
    if not profiles:
        return {}
    p = profiles[0]
    keys = ["name", "role", "age_group", "gender", "face", "hair", "body", "outfit", "signature_items", "color_palette", "identity_anchor_prompt", "negative_identity_prompt"]
    return {k: getattr(p, k, "") for k in keys}


def _is_panda(seed: Any, sample: Dict[str, Any] | None = None) -> bool:
    text = " ".join([clean(getattr(seed, "protagonist", "")), clean((sample or {}).get("protagonist", "")), clean((sample or {}).get("text_prompt", ""))]).lower()
    return "panda" in text or "팬더" in text or "판다" in text


def _fixed_world(sample: Dict[str, Any], seed: Any, storyboard: List[Any]) -> Dict[str, Any]:
    locations, weather, atmosphere, objects = [], [], [], []
    for f in storyboard or []:
        locations += as_list(getattr(f, "scene_location", ""))
        weather += as_list(getattr(f, "weather", ""))
        atmosphere += as_list(getattr(f, "atmosphere", ""))
        objects += as_list(getattr(f, "key_objects", []))
        objects += as_list(getattr(f, "evidence_objects", []))
        objects += as_list(getattr(f, "must_show", []))
    sample_prompt = clean(sample.get("text_prompt", ""))
    setting = clean(getattr(seed, "setting", ""))
    object_text = " ".join(objects).lower() + " " + sample_prompt.lower() + " " + setting.lower()
    if "panda" in object_text or "bamboo" in object_text or "river" in object_text:
        fixed_setting = "deep green bamboo forest beside a visible riverbank"
        stable_background = ["deep green bamboo forest", "visible riverbank", "flowing river water", "bamboo grove", "river rocks", "green leaves"]
    else:
        fixed_setting = locations[0] if locations else (setting or "consistent story location")
        stable_background = unique(locations + objects, 6)
    return {
        "fixed_setting": fixed_setting,
        "stable_background": unique(stable_background, 6),
        "default_weather": weather[0] if weather else "clear storybook weather",
        "default_atmosphere": atmosphere[0] if atmosphere else "cinematic storybook atmosphere",
        "world_lock_prompt": "Keep the same world across all frames: " + fixed_setting + ". The background must visibly include: " + ", ".join(unique(stable_background, 6)) + ".",
    }


def build_story_bible(sample: Dict[str, Any], seed: Any, dce_plan: Any, storyboard: List[Any], full_story: Dict[str, Any] | None = None) -> Dict[str, Any]:
    profile = _profile_dict(seed)
    is_panda = _is_panda(seed, sample)
    protagonist = clean(getattr(seed, "protagonist", "")) or clean(sample.get("protagonist", "")) or "protagonist"
    outfit = clean(sample.get("outfit", "")) or clean(profile.get("outfit", ""))
    signature_items = unique(as_list(sample.get("signature_items", [])) + as_list(profile.get("signature_items", [])), 4)
    if is_panda:
        age_lock = "adult giant panda, not a cub, not a baby, not a juvenile"
        subject_identity = "same adult giant panda protagonist in every frame; large adult body proportions; black-and-white fur; round expressive face; large dark eyes; soft fluffy body; " + (f"wearing {outfit}; " if outfit else "") + (("signature items: " + ", ".join(signature_items)) if signature_items else "")
        negative_identity = ["baby panda", "panda cub", "small juvenile panda", "childlike panda", "different panda", "multiple pandas", "human panda hybrid", "human protagonist", "panda turning into a human", "different outfit", "missing yellow shirt"]
    else:
        age_lock = clean(profile.get("age_group", "")) or "same age group"
        subject_identity = f"same protagonist in every frame: {protagonist}; " + "; ".join([clean(profile.get(k, "")) for k in ["age_group", "gender", "face", "hair", "body", "outfit"] if clean(profile.get(k, ""))])
        negative_identity = ["different protagonist", "changed age", "changed gender", "different face", "different outfit", "extra duplicate protagonist"]
    world = _fixed_world(sample, seed, storyboard)
    emotion_rules = {
        "hope": {"face": "wide bright eyes and slight smile", "body": "upright open posture"},
        "hopeful": {"face": "wide bright eyes and slight smile", "body": "upright open posture"},
        "joy": {"face": "big smile and raised cheeks", "body": "open happy posture holding bamboo"},
        "joyful": {"face": "big smile and raised cheeks", "body": "open happy posture holding bamboo"},
        "shocked": {"face": "wide eyes and open mouth", "body": "frozen startled pose with paws lifted"},
        "fear": {"face": "wide fearful eyes", "body": "body leaning backward"},
        "despair": {"face": "downturned eyes and trembling mouth", "body": "reaching paws and collapsed shoulders"},
        "despairing": {"face": "downturned eyes and trembling mouth", "body": "reaching paws and collapsed shoulders"},
        "helpless": {"face": "sad helpless eyes", "body": "slumped shoulders and lowered head"},
        "sorrowful": {"face": "wet sad eyes", "body": "small folded posture"},
        "sad": {"face": "tearful eyes and downturned mouth", "body": "slumped shoulders, lowered head, empty paws visible"},
        "sadness": {"face": "tearful eyes and downturned mouth", "body": "slumped shoulders, lowered head, empty paws visible"},
    }
    return {
        "protagonist": protagonist,
        "subject_identity_prompt": subject_identity,
        "age_lock": age_lock,
        "negative_identity": negative_identity,
        "world": world,
        "emotion_rules": emotion_rules,
        "global_negative": unique(negative_identity + ["wrong background", "missing river", "missing bamboo forest", "empty background", "no facial expression", "emotionless face", "stiff body", "portrait only", "cropped face", "cropped paws", "low quality", "text", "watermark"], 40),
        "role_constraints": [
            "Only the protagonist should be the central subject.",
            "If a secondary character appears, place it in the background or side, not replacing the protagonist.",
            "The protagonist must never change age, species, outfit, or body proportions.",
        ],
    }


def emotion_cue_from_bible(story_bible: Dict[str, Any], emotion: str) -> Dict[str, str]:
    e = clean(emotion).lower()
    rules = story_bible.get("emotion_rules", {}) if isinstance(story_bible, dict) else {}
    if e in rules:
        return rules[e]
    for k, v in rules.items():
        if k in e or e in k:
            return v
    return {"face": f"clear facial expression of {emotion}", "body": f"clear body posture showing {emotion}"}


def world_for_frame(story_bible: Dict[str, Any], frame: Any) -> Dict[str, Any]:
    world = story_bible.get("world", {}) if isinstance(story_bible, dict) else {}
    return {
        "fixed_setting": world.get("fixed_setting", clean(getattr(frame, "scene_location", ""))),
        "stable_background": world.get("stable_background", []),
        "weather": clean(getattr(frame, "weather", "")) or world.get("default_weather", ""),
        "atmosphere": clean(getattr(frame, "atmosphere", "")) or world.get("default_atmosphere", ""),
        "world_lock_prompt": world.get("world_lock_prompt", ""),
    }
