from dataclasses import asdict
from typing import Any, Dict, List

from .schema import StoryboardFrame, CandidateImage, DCEPlan, EmotionArc


def lexical_score(a: str, b: str) -> float:
    aa = set(a.lower().replace(",", " ").replace(".", " ").split())
    bb = set(b.lower().replace(",", " ").replace(".", " ").split())
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / max(1, len(aa | bb))


def compact_frame(frame: StoryboardFrame | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(frame, StoryboardFrame):
        f = asdict(frame)
    else:
        f = dict(frame)

    return {
        "frame_id": f.get("frame_id"),
        "caption": f.get("caption", ""),
        "narrative_function": f.get("narrative_function", ""),
        "event": f.get("event", ""),
        "protagonist_state": f.get("protagonist_state", ""),
        "desire_link": f.get("desire_link", ""),
        "conflict_level": f.get("conflict_level", 1),
        "emotion": f.get("emotion", ""),
        "emotion_intensity": f.get("emotion_intensity", 3),
        "visual_focus": f.get("visual_focus", ""),
        "key_objects": f.get("key_objects", []),
        "facial_cue": f.get("facial_cue", ""),
        "body_cue": f.get("body_cue", ""),
        "event_cue": f.get("event_cue", ""),
        "scene_cue": f.get("scene_cue", ""),
        "cinematic_cue": f.get("cinematic_cue", ""),
        "character_identity": f.get("character_identity", ""),
        "character_reference_prompt": f.get("character_reference_prompt", ""),
        "emotion_delta": f.get("emotion_delta", ""),
        "emotion_visual_rule": f.get("emotion_visual_rule", ""),
    }


class NarrativeMemoryStore:
    def __init__(self):
        self.frames: List[Dict[str, Any]] = []

    def add(self, frame: StoryboardFrame, image: CandidateImage | None = None):
        self.frames.append({"frame": compact_frame(frame), "image": image.image_path if image else None})

    def select(self, current_frame: StoryboardFrame, dce_plan: DCEPlan, emotion_arc: EmotionArc, strategy: str = "multi_slot") -> Dict[str, Any]:
        if not self.frames:
            return {
                "identity_memory": current_frame.character_reference_prompt or "Use the same protagonist identity in every frame.",
                "object_memory": current_frame.key_objects,
                "emotion_memory": current_frame.emotion,
                "conflict_memory": dce_plan.conflict,
                "style_memory": "Preserve the same cinematic storybook illustration style.",
                "arc_memory": {"states": emotion_arc.states, "intensities": emotion_arc.intensities},
            }

        if strategy == "all":
            return {"all_history": self.frames, "current_identity_rule": current_frame.character_reference_prompt, "arc_memory": {"states": emotion_arc.states, "intensities": emotion_arc.intensities}}

        if strategy == "last":
            return {"last_history": self.frames[-1], "current_identity_rule": current_frame.character_reference_prompt, "arc_memory": {"states": emotion_arc.states, "intensities": emotion_arc.intensities}}

        if strategy == "salient":
            query = " ".join([current_frame.caption, current_frame.event, current_frame.emotion, dce_plan.conflict])
            scored = []
            for item in self.frames:
                f = item["frame"]
                text = " ".join([f.get("caption", ""), f.get("event", ""), f.get("emotion", ""), f.get("visual_focus", "")])
                scored.append((lexical_score(query, text), item))
            scored.sort(key=lambda x: x[0], reverse=True)
            return {"salient_history": scored[0][1], "current_identity_rule": current_frame.character_reference_prompt, "arc_memory": {"states": emotion_arc.states, "intensities": emotion_arc.intensities}}

        identity = self.frames[0]
        object_query = " ".join(current_frame.key_objects + [current_frame.visual_focus])
        object_item = max(
            self.frames,
            key=lambda item: lexical_score(object_query, " ".join(item["frame"].get("key_objects", [])) + " " + item["frame"].get("visual_focus", "")),
        )
        emotion_item = max(self.frames, key=lambda item: lexical_score(current_frame.emotion, item["frame"].get("emotion", "")))
        conflict_item = max(self.frames, key=lambda item: item["frame"].get("conflict_level", 0))

        return {
            "identity_memory": identity,
            "object_memory": object_item,
            "emotion_memory": emotion_item,
            "conflict_memory": conflict_item,
            "style_memory": {"style_rule": "Preserve the same cinematic storybook illustration style.", "reference_frame_id": self.frames[0]["frame"].get("frame_id")},
            "current_identity_rule": current_frame.character_reference_prompt,
            "current_emotion_delta": current_frame.emotion_delta,
            "global_desire": dce_plan.desire,
            "global_conflict": dce_plan.conflict,
            "target_ending_emotion": dce_plan.target_ending_emotion,
            "arc_memory": {"states": emotion_arc.states, "intensities": emotion_arc.intensities},
        }
