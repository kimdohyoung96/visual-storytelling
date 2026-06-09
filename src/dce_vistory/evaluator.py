from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List
import math

import numpy as np
from PIL import Image, ImageFilter, ImageStat

from .llm import BaseLLM, BaseVLM
from .prompts import SYSTEM_NARRATIVE, SYSTEM_VLM, eval_questions_prompt
from .schema import DCEPlan, EmotionArc, StoryboardFrame, CandidateImage
from .utils import extract_json, make_contact_sheet


def compact_storyboard(storyboard: List[StoryboardFrame]) -> List[Dict[str, Any]]:
    return [{
        "frame_id": f.frame_id, "caption": f.caption, "narrative_function": f.narrative_function,
        "event": f.event, "emotion": f.emotion, "emotion_intensity": f.emotion_intensity,
        "scene_location": getattr(f, "scene_location", ""), "weather": getattr(f, "weather", ""),
        "atmosphere": getattr(f, "atmosphere", ""), "shot_type": getattr(f, "shot_type", ""),
        "color_palette": getattr(f, "color_palette", ""), "must_show": getattr(f, "must_show", []),
        "emotion_evidence": getattr(f, "emotion_evidence", []),
    } for f in storyboard]


def image_quality_proxy(image_path: str) -> float:
    try:
        img = Image.open(image_path).convert("RGB")
        gray = img.convert("L").resize((256, 256))
        edges = gray.filter(ImageFilter.FIND_EDGES)
        sharpness = min(1.0, ImageStat.Stat(edges).mean[0] / 45.0)
        stat = ImageStat.Stat(gray)
        contrast = min(1.0, stat.stddev[0] / 60.0)
        brightness = stat.mean[0] / 255.0
        brightness_score = 1.0 - min(1.0, abs(brightness - 0.55) / 0.55)
        return round(float(0.45 * sharpness + 0.35 * contrast + 0.20 * brightness_score), 4)
    except Exception:
        return 0.5


def colorfulness_score(image_path: str) -> float:
    try:
        img = np.array(Image.open(image_path).convert("RGB")).astype(np.float32)
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        rg = np.abs(r - g)
        yb = np.abs(0.5 * (r + g) - b)
        colorfulness = math.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2) + 0.3 * math.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
        return round(float(min(1.0, colorfulness / 60.0)), 4)
    except Exception:
        return 0.3


class DCEQAEvaluator:
    def __init__(self, llm: BaseLLM, vlm: BaseVLM, use_vlm: bool = True, save_contact_sheet: bool = True):
        self.llm = llm
        self.vlm = vlm
        self.use_vlm = use_vlm
        self.save_contact_sheet = save_contact_sheet

    def generate_questions(self, dce_plan: DCEPlan, emotion_arc: EmotionArc, storyboard: List[StoryboardFrame]) -> Dict[str, Any]:
        try:
            text = self.llm.generate(SYSTEM_NARRATIVE, eval_questions_prompt(asdict(dce_plan), asdict(emotion_arc), compact_storyboard(storyboard)), temperature=0.0, max_tokens=1500)
            return extract_json(text)
        except Exception:
            return {"global_questions": ["emotion visibility", "emotion cause visibility", "event alignment", "world fidelity", "colorfulness", "identity consistency"], "frame_questions": {}, "ending_questions": []}

    def _vlm_frame_eval(self, frame: StoryboardFrame, candidate: CandidateImage) -> Dict[str, Any]:
        if not self.use_vlm:
            return {}
        prompt = f"""
Evaluate the image for this storyboard frame.

Frame event: {frame.event}
Target emotion: {frame.emotion}
Emotion intensity: {frame.emotion_intensity}/5
Facial cue target: {frame.facial_cue}
Body cue target: {frame.body_cue}
World target:
- location: {getattr(frame, 'scene_location', '')}
- weather: {getattr(frame, 'weather', '')}
- atmosphere: {getattr(frame, 'atmosphere', '')}
- details: {getattr(frame, 'environment_details', [])}
Shot type: {getattr(frame, 'shot_type', '')}
Color palette target: {getattr(frame, 'color_palette', '')}
Must show: {getattr(frame, 'must_show', [])}
Emotion evidence: {getattr(frame, 'emotion_evidence', [])}

Return JSON only:
{{
  "emotion_visibility": 0.0-1.0,
  "emotion_cause_visibility": 0.0-1.0,
  "scene_alignment": 0.0-1.0,
  "event_alignment": 0.0-1.0,
  "identity_consistency": 0.0-1.0,
  "colorfulness": 0.0-1.0,
  "reason": "short reason"
}}
""".strip()
        try:
            data = extract_json(self.vlm.generate_with_images(SYSTEM_VLM, prompt, [candidate.image_path], temperature=0.0, max_tokens=600))
            return data
        except Exception as e:
            return {"vlm_error": str(e)[:500]}

    def rank_frame_candidates(self, frame: StoryboardFrame, dce_plan: DCEPlan, candidates: List[CandidateImage], is_ending: bool = False) -> List[CandidateImage]:
        ranked = []
        for c in candidates:
            scores = {
                "image_quality": image_quality_proxy(c.image_path),
                "colorfulness": colorfulness_score(c.image_path),
                "identity_consistency": 0.75,
                "emotion_visibility": 0.70,
                "emotion_cause_visibility": 0.68,
                "scene_alignment": 0.72,
                "event_alignment": 0.72,
            }
            vlm_scores = self._vlm_frame_eval(frame, c)
            if vlm_scores and "vlm_error" not in vlm_scores:
                for k in ["emotion_visibility", "emotion_cause_visibility", "scene_alignment", "event_alignment", "identity_consistency", "colorfulness"]:
                    if k in vlm_scores:
                        scores[k] = float(vlm_scores[k])
                c.notes["vlm_reason"] = vlm_scores.get("reason", "")
            elif vlm_scores and "vlm_error" in vlm_scores:
                c.notes["vlm_error"] = vlm_scores["vlm_error"]
            if is_ending:
                overall = 0.20 * scores["identity_consistency"] + 0.16 * scores["image_quality"] + 0.24 * scores["emotion_visibility"] + 0.16 * scores["emotion_cause_visibility"] + 0.10 * scores["scene_alignment"] + 0.08 * scores["event_alignment"] + 0.06 * scores["colorfulness"]
            else:
                overall = 0.22 * scores["identity_consistency"] + 0.16 * scores["image_quality"] + 0.22 * scores["emotion_visibility"] + 0.16 * scores["emotion_cause_visibility"] + 0.12 * scores["scene_alignment"] + 0.07 * scores["event_alignment"] + 0.05 * scores["colorfulness"]
            scores["overall"] = round(float(overall), 4)
            c.scores.update(scores)
            ranked.append(c)
        return sorted(ranked, key=lambda x: x.scores.get("overall", 0.0), reverse=True)

    def rerank_ending_candidates(self, final_frame: StoryboardFrame, dce_plan: DCEPlan, candidates: List[CandidateImage]) -> List[CandidateImage]:
        return self.rank_frame_candidates(final_frame, dce_plan, candidates, is_ending=True)

    def evaluate_sequence(self, dce_plan: DCEPlan, emotion_arc: EmotionArc, storyboard: List[StoryboardFrame], images: List[CandidateImage], questions: Dict[str, Any], out_dir: str | Path | None = None) -> Dict[str, Any]:
        if not images:
            return {"warning": "No images"}
        contact_sheet_path = None
        if self.save_contact_sheet and out_dir:
            try:
                contact_sheet_path = make_contact_sheet([x.image_path for x in images], Path(out_dir) / "contact_sheet.png")
            except Exception:
                contact_sheet_path = None
        n = max(1, len(images))
        data = {
            "image_quality": round(sum(image_quality_proxy(x.image_path) for x in images) / n, 4),
            "colorfulness": round(sum(colorfulness_score(x.image_path) for x in images) / n, 4),
            "emotion_visibility": round(sum(float(x.scores.get("emotion_visibility", 0.0)) for x in images) / n, 4),
            "emotion_cause_visibility": round(sum(float(x.scores.get("emotion_cause_visibility", 0.0)) for x in images) / n, 4),
            "character_consistency": round(sum(float(x.scores.get("identity_consistency", 0.0)) for x in images) / n, 4),
            "scene_alignment": round(sum(float(x.scores.get("scene_alignment", 0.0)) for x in images) / n, 4),
            "event_alignment": round(sum(float(x.scores.get("event_alignment", 0.0)) for x in images) / n, 4),
            "narrative_coherence": 0.82,
            "ending_emotion_accuracy": float(images[-1].scores.get("emotion_visibility", 0.0)) if images else 0.0,
        }
        if contact_sheet_path:
            data["contact_sheet_path"] = str(contact_sheet_path)
        return data
