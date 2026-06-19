from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List
import json
import math

import numpy as np
from PIL import Image, ImageFilter, ImageStat, ImageDraw

from .llm import BaseLLM, BaseVLM
from .prompts import SYSTEM_NARRATIVE, SYSTEM_VLM, eval_questions_prompt
from .schema import DCEPlan, EmotionArc, StoryboardFrame, CandidateImage
from .utils import extract_json


def _safe_asdict(obj: Any):
    if is_dataclass(obj):
        d = asdict(obj)
        d.update({k: v for k, v in getattr(obj, "__dict__", {}).items() if k not in d})
        return _safe_asdict(d)
    if isinstance(obj, dict):
        return {str(k): _safe_asdict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_asdict(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def _compact_storyboard(storyboard: List[StoryboardFrame]) -> List[Dict[str, Any]]:
    return [
        {
            "frame_id": getattr(f, "frame_id", i + 1),
            "event": getattr(f, "event", ""),
            "event_grounding": getattr(f, "event_grounding", ""),
            "evidence_objects": getattr(f, "evidence_objects", []),
            "emotion_evidence": getattr(f, "emotion_evidence", []),
            "emotion": getattr(f, "emotion", ""),
            "must_show": getattr(f, "must_show", []),
            "scene_location": getattr(f, "scene_location", ""),
            "weather": getattr(f, "weather", ""),
        }
        for i, f in enumerate(storyboard)
    ]


def image_quality_proxy(path: str) -> float:
    try:
        img = Image.open(path).convert("RGB")
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


def colorfulness_score(path: str) -> float:
    try:
        img = np.array(Image.open(path).convert("RGB")).astype(np.float32)
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]
        rg = np.abs(r - g)
        yb = np.abs(0.5 * (r + g) - b)
        colorfulness = math.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2) + 0.3 * math.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
        return round(float(min(1.0, colorfulness / 60.0)), 4)
    except Exception:
        return 0.3


def _make_contact_sheet_local(image_paths: List[str], out_path: Path, cols: int = 3, thumb_size=(384, 384)) -> str:
    valid = []
    for p in image_paths:
        if not p:
            continue
        pp = Path(str(p))
        if pp.exists():
            valid.append(pp)

    if not valid:
        raise ValueError("No valid image paths for contact sheet.")

    rows = (len(valid) + cols - 1) // cols
    header_h = 54
    canvas = Image.new("RGB", (cols * thumb_size[0], rows * thumb_size[1] + header_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), "DCEE-CausalVerse Contact Sheet", fill=(0, 0, 0))

    for idx, p in enumerate(valid):
        img = Image.open(p).convert("RGB")
        img.thumbnail((thumb_size[0] - 20, thumb_size[1] - 44))
        x0 = (idx % cols) * thumb_size[0]
        y0 = header_h + (idx // cols) * thumb_size[1]
        draw.rectangle([x0, y0, x0 + thumb_size[0] - 1, y0 + thumb_size[1] - 1], outline=(180, 180, 180))
        draw.text((x0 + 10, y0 + 10), f"Frame {idx + 1}", fill=(0, 0, 0))
        canvas.paste(img, (x0 + (thumb_size[0] - img.width) // 2, y0 + 34 + (thumb_size[1] - 44 - img.height) // 2))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return str(out_path)


class DCEQAEvaluator:
    def __init__(self, llm: BaseLLM, vlm: BaseVLM, use_vlm: bool = True, save_contact_sheet: bool = True):
        self.llm = llm
        self.vlm = vlm
        self.use_vlm = use_vlm
        self.save_contact_sheet = save_contact_sheet

    def generate_questions(
        self,
        dce_plan: DCEPlan,
        emotion_arc: EmotionArc,
        storyboard: List[StoryboardFrame],
    ) -> Dict[str, Any]:
        try:
            return extract_json(
                self.llm.generate(
                    SYSTEM_NARRATIVE,
                    eval_questions_prompt(asdict(dce_plan), asdict(emotion_arc), _compact_storyboard(storyboard)),
                    temperature=0.0,
                    max_tokens=1000,
                )
            )
        except Exception:
            return {
                "global_questions": [
                    "Does the generated sequence follow the planned DCEE event chain?",
                    "Is the visual evidence for each event visible?",
                    "Does each event explain or intensify the protagonist emotion?",
                    "Is the target ending emotion achieved?",
                    "Are character identity and world state consistent?",
                ],
                "frame_questions": {},
                "ending_questions": [],
            }

    def _vlm_frame_eval(self, frame, cand) -> Dict[str, Any]:
        if not self.use_vlm:
            return {}

        prompt = f"""
Evaluate this image for DCEE visual storytelling. Prioritize story-event fit, evidence visibility, visible cause of emotion, and target emotion. Do not over-reward fixed shot type or fixed pose.

Planned event: {getattr(frame, 'event', '')}
Event grounding: {getattr(frame, 'event_grounding', '')}
Evidence objects: {getattr(frame, 'evidence_objects', [])}
Emotion evidence: {getattr(frame, 'emotion_evidence', [])}
Must show: {getattr(frame, 'must_show', [])}
Target emotion: {getattr(frame, 'emotion', '')} intensity {getattr(frame, 'emotion_intensity', '')}/5
World: {getattr(frame, 'scene_location', '')}, {getattr(frame, 'weather', '')}, {getattr(frame, 'atmosphere', '')}

Return JSON only with scores from 0 to 1:
event_grounding, evidence_visibility, emotion_visibility, emotion_cause_visibility,
event_emotion_causal_consistency, scene_alignment, event_alignment,
identity_consistency, colorfulness, reason.
""".strip()
        try:
            return extract_json(
                self.vlm.generate_with_images(
                    SYSTEM_VLM,
                    prompt,
                    [cand.image_path],
                    temperature=0.0,
                    max_tokens=500,
                )
            )
        except Exception as e:
            return {"vlm_error": str(e)[:300]}

    def rank_frame_candidates(self, frame, dce_plan, candidates, is_ending: bool = False):
        ranked = []
        for c in candidates:
            scores = {
                "image_quality": image_quality_proxy(c.image_path),
                "colorfulness": colorfulness_score(c.image_path),
                "identity_consistency": 0.74,
                "emotion_visibility": 0.70,
                "emotion_cause_visibility": 0.68,
                "event_grounding": 0.68,
                "evidence_visibility": 0.66,
                "event_emotion_causal_consistency": 0.68,
                "scene_alignment": 0.70,
                "event_alignment": 0.70,
            }

            vlm_scores = self._vlm_frame_eval(frame, c)
            if vlm_scores and "vlm_error" not in vlm_scores:
                for key in list(scores.keys()):
                    if key in vlm_scores:
                        try:
                            scores[key] = float(vlm_scores[key])
                        except Exception:
                            pass
                c.notes["vlm_reason"] = vlm_scores.get("reason", "")
            elif vlm_scores and "vlm_error" in vlm_scores:
                c.notes["vlm_error"] = vlm_scores["vlm_error"]

            if is_ending:
                overall = (
                    0.12 * scores["identity_consistency"]
                    + 0.12 * scores["image_quality"]
                    + 0.16 * scores["emotion_visibility"]
                    + 0.16 * scores["emotion_cause_visibility"]
                    + 0.17 * scores["event_grounding"]
                    + 0.15 * scores["evidence_visibility"]
                    + 0.08 * scores["event_emotion_causal_consistency"]
                    + 0.04 * scores["colorfulness"]
                )
            else:
                overall = (
                    0.15 * scores["identity_consistency"]
                    + 0.13 * scores["image_quality"]
                    + 0.14 * scores["emotion_visibility"]
                    + 0.13 * scores["emotion_cause_visibility"]
                    + 0.16 * scores["event_grounding"]
                    + 0.14 * scores["evidence_visibility"]
                    + 0.10 * scores["scene_alignment"]
                    + 0.05 * scores["colorfulness"]
                )

            scores["overall"] = round(float(overall), 4)
            c.scores.update(scores)
            ranked.append(c)

        return sorted(ranked, key=lambda x: x.scores.get("overall", 0.0), reverse=True)

    def rerank_ending_candidates(self, final_frame, dce_plan, candidates):
        return self.rank_frame_candidates(final_frame, dce_plan, candidates, is_ending=True)

    def evaluate_sequence(
        self,
        dce_plan,
        emotion_arc,
        storyboard,
        images,
        questions,
        out_dir=None,
    ) -> Dict[str, Any]:
        if not images:
            return {"warning": "No images"}

        n = max(1, len(images))
        keys = [
            "image_quality",
            "colorfulness",
            "identity_consistency",
            "emotion_visibility",
            "emotion_cause_visibility",
            "event_grounding",
            "evidence_visibility",
            "event_emotion_causal_consistency",
            "scene_alignment",
            "event_alignment",
        ]
        data = {
            key: round(sum(float(x.scores.get(key, 0.0)) for x in images) / n, 4)
            for key in keys
        }

        data["ending_emotion_accuracy"] = float(images[-1].scores.get("emotion_visibility", 0.0)) if images else 0.0
        data["narrative_coherence"] = round(
            (
                data.get("event_grounding", 0.0)
                + data.get("event_emotion_causal_consistency", 0.0)
                + data.get("scene_alignment", 0.0)
            )
            / 3,
            4,
        )

        if self.save_contact_sheet and out_dir:
            try:
                data["contact_sheet_path"] = _make_contact_sheet_local(
                    [getattr(x, "image_path", "") for x in images],
                    Path(out_dir) / "contact_sheet.png",
                )
            except Exception as e:
                data["contact_sheet_error"] = f"{type(e).__name__}: {e}"

        return data
