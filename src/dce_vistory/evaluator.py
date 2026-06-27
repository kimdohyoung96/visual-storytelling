from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List
import json
import math
import re

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
    return [{
        "frame_id": getattr(f, "frame_id", i + 1),
        "story_sentence": getattr(f, "story_sentence", ""),
        "image_sentence": getattr(f, "image_sentence", ""),
        "event": getattr(f, "event", ""),
        "event_grounding": getattr(f, "event_grounding", ""),
        "evidence_objects": getattr(f, "evidence_objects", []),
        "emotion": getattr(f, "emotion", ""),
        "must_show": getattr(f, "must_show", []),
        "scene_location": getattr(f, "scene_location", ""),
        "weather": getattr(f, "weather", ""),
    } for i, f in enumerate(storyboard)]


def _tokens(x):
    if isinstance(x, (list, tuple)):
        x = " ".join(str(v) for v in x)
    s = re.sub(r"[^A-Za-z0-9가-힣 ]+", " ", str(x or "").lower())
    stop = {"the", "a", "an", "of", "to", "and", "or", "with", "in", "on", "at", "by", "for", "his", "her"}
    return {t for t in s.split() if len(t) >= 2 and t not in stop}


def _overlap(a, b):
    A = _tokens(a)
    B = _tokens(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def _clamp01(x, default=0.0) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return float(default)


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
    valid = [Path(str(p)) for p in image_paths if p and Path(str(p)).exists()]
    if not valid:
        raise ValueError("No valid image paths for contact sheet.")
    rows = (len(valid) + cols - 1) // cols
    header_h = 54
    canvas = Image.new("RGB", (cols * thumb_size[0], rows * thumb_size[1] + header_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), "DCEE-CausalVerse V23 Contact Sheet", fill=(0, 0, 0))
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


def _frame_tifa_questions(frame: Any) -> List[str]:
    return [
        f"Does the image match the exact story sentence `{getattr(frame, 'story_sentence', '')}`?",
        f"Is the main visible action `{getattr(frame, 'event', '')}` clearly shown?",
        f"Does the image contain exactly one protagonist and no duplicate protagonist?",
        f"Can you see evidence objects {getattr(frame, 'evidence_objects', []) or getattr(frame, 'must_show', [])} or equivalent visual clues?",
        f"Can you understand the cause `{getattr(frame, 'event_grounding', '')}` from the image?",
        f"Is the protagonist emotion `{getattr(frame, 'emotion', '')}` clearly visible in face/body/scene?",
        f"Do the background/weather/location match `{getattr(frame, 'scene_location', '')}` / `{getattr(frame, 'weather', '')}`?",
        "Is the image one single coherent scene rather than split-screen or multi-panel?",
    ]


def _caption_bad_signals(caption: str) -> float:
    cap = str(caption or "").lower()
    penalty = 0.0
    if re.search(r"\btwo\b|\bthree\b|\bseveral\b|\bgroup\b|\bcrowd\b|\bmultiple\b", cap):
        penalty += 0.25
    if "bear" in cap and re.search(r"\bcub\b|\bbaby\b|\bsmall bear\b", cap):
        penalty += 0.15
    if re.search(r"\bperson\b|\bman\b|\bwoman\b|\bhuman\b", cap):
        penalty += 0.25
    return min(0.5, penalty)


def _variant_priority(candidate: CandidateImage, is_ending: bool = False) -> float:
    mode = str(getattr(candidate, "notes", {}).get("prompt_variant_mode", ""))
    base = {
        "event_locked": 0.055,
        "evidence_locked": 0.050,
        "emotion_causal_locked": 0.060 if is_ending else 0.045,
        "continuity_locked": 0.045 if is_ending else 0.025,
        "composition_locked": 0.010,
        # Backward compatibility with older candidates
        "sentence_locked": 0.030,
        "object_locked": 0.025,
        "background_locked": -0.060,
        "emotion_locked": 0.030,
    }
    return float(base.get(mode, 0.0))


def _candidate_static_penalty(frame: Any, candidate: CandidateImage) -> float:
    prompt = str(getattr(candidate, "prompt", "")).lower()
    event = str(getattr(frame, "event", "")).lower()
    # Penalize variants that drift to portrait/background instead of event.
    penalty = 0.0
    if "generic portrait" in prompt:
        penalty += 0.08
    if any(v in event for v in ["sit", "sits", "stand", "stands", "walk", "step", "gasp", "hold", "reach"]) and "action" not in prompt:
        penalty += 0.03
    return min(0.20, penalty)


class DCEQAEvaluator:
    def __init__(self, llm: BaseLLM, vlm: BaseVLM, use_vlm: bool = True, save_contact_sheet: bool = True, use_local_caption_scorer: bool = True):
        self.llm = llm
        self.vlm = vlm
        self.use_vlm = use_vlm
        self.save_contact_sheet = save_contact_sheet
        self.use_local_caption_scorer = use_local_caption_scorer
        self.local_captioner = None
        if use_local_caption_scorer:
            try:
                from transformers import pipeline
                self.local_captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
            except Exception:
                self.local_captioner = None

    def generate_questions(self, dce_plan: DCEPlan, emotion_arc: EmotionArc, storyboard: List[StoryboardFrame]) -> Dict[str, Any]:
        out = {
            "global_questions": [
                "Does the generated sequence follow the planned DCEE event chain?",
                "Is each frame grounded in its story sentence rather than a generic portrait?",
                "Are event, evidence, emotional cause, and target emotion visible?",
                "Are character identity and world state consistent across frames?",
                "Does each frame contain exactly one protagonist and one coherent scene?"
            ],
            "frame_questions": {str(getattr(f, "frame_id", i + 1)): _frame_tifa_questions(f) for i, f in enumerate(storyboard)},
            "ending_questions": [
                f"Does the final image reach the target ending emotion `{getattr(dce_plan, 'target_ending_emotion', '')}`?",
                "Does the final frame feel like a story ending rather than just another intermediate scene?"
            ],
        }
        try:
            llm_extra = extract_json(self.llm.generate(SYSTEM_NARRATIVE, eval_questions_prompt(asdict(dce_plan), asdict(emotion_arc), _compact_storyboard(storyboard)), temperature=0.0, max_tokens=800))
            if isinstance(llm_extra, dict):
                out["llm_generated_questions"] = llm_extra
        except Exception:
            pass
        return out

    def _local_caption_eval(self, frame, cand) -> Dict[str, Any]:
        if self.local_captioner is None:
            return {}
        try:
            cap = self.local_captioner(cand.image_path, max_new_tokens=48)
            if isinstance(cap, list) and cap:
                txt = cap[0].get("generated_text", "")
            elif isinstance(cap, dict):
                txt = cap.get("generated_text", "")
            else:
                txt = str(cap)
            return {
                "local_caption": txt,
                "story_alignment_local": _overlap(txt, getattr(frame, "story_sentence", "")),
                "event_alignment_local": max(_overlap(txt, getattr(frame, "event", "")), _overlap(txt, getattr(frame, "event_grounding", ""))),
                "evidence_visibility_local": max(_overlap(txt, getattr(frame, "evidence_objects", [])), _overlap(txt, getattr(frame, "must_show", []))),
                "scene_alignment_local": max(_overlap(txt, getattr(frame, "scene_location", "")), _overlap(txt, getattr(frame, "weather", ""))),
                "emotion_visibility_local": max(_overlap(txt, getattr(frame, "emotion", "")), _overlap(txt, getattr(frame, "emotion_evidence", []))),
            }
        except Exception as e:
            return {"local_caption_error": str(e)[:300]}

    def _vlm_frame_eval(self, frame, cand) -> Dict[str, Any]:
        if not self.use_vlm:
            return {}
        prompt = f"""
You are a strict visual storytelling candidate judge.
Evaluate the image against the planned frame. Return JSON only.

Planned story sentence: {getattr(frame, 'story_sentence', '')}
Image-friendly sentence: {getattr(frame, 'image_sentence', '')}
Planned event/action: {getattr(frame, 'event', '')}
Event cause/evidence: {getattr(frame, 'event_grounding', '')}
Required visual inventory: {getattr(frame, 'must_show', []) or getattr(frame, 'evidence_objects', [])}
Target emotion: {getattr(frame, 'emotion', '')}
Location/weather/background: {getattr(frame, 'scene_location', '')}, {getattr(frame, 'weather', '')}, {getattr(frame, 'environment_details', [])}
Protagonist identity: {getattr(frame, 'character_reference_prompt', '') or getattr(frame, 'character_identity', '')}

Return JSON with these exact keys:
story_alignment, event_alignment, event_grounding, evidence_visibility,
emotion_visibility, emotion_cause_visibility, scene_alignment, continuity,
identity_consistency, single_scene_score, qa_score,
exactly_one_protagonist, duplicate_protagonist, second_protagonist, split_panel, extra_character,
action_visible, protagonist_sitting_if_required, reason.

Scoring rules:
- Scores are 0.0 to 1.0.
- If two bears/two protagonists/duplicate protagonist appear, set duplicate_protagonist=true and identity_consistency<=0.25.
- If the required action is not visible, set event_alignment<=0.35.
- If the image is a portrait but the story requires an action/location, set story_alignment<=0.45.
- If the image has multiple panels/scenes, set split_panel=true and single_scene_score<=0.2.
- Prefer story/action/evidence correctness over beauty.
""".strip()
        try:
            data = extract_json(self.vlm.generate_with_images(SYSTEM_VLM, prompt, [cand.image_path], temperature=0.0, max_tokens=900))
            return data if isinstance(data, dict) else {}
        except Exception as e:
            return {"vlm_error": str(e)[:300]}

    def rank_frame_candidates(self, frame, dce_plan, candidates, is_ending: bool = False):
        ranked = []
        for c in candidates:
            scores = {
                "image_quality": image_quality_proxy(c.image_path),
                "colorfulness": colorfulness_score(c.image_path),
                "identity_consistency": 0.50,
                "story_alignment": 0.45,
                "event_alignment": 0.45,
                "event_grounding": 0.45,
                "evidence_visibility": 0.45,
                "emotion_visibility": 0.45,
                "emotion_cause_visibility": 0.45,
                "scene_alignment": 0.45,
                "continuity": 0.50,
                "qa_score": 0.45,
                "single_scene_score": 0.60,
                "duplicate_penalty": 0.0,
                "action_missing_penalty": 0.0,
                "variant_priority": _variant_priority(c, is_ending=is_ending),
                "static_penalty": _candidate_static_penalty(frame, c),
            }

            local_scores = self._local_caption_eval(frame, c)
            if local_scores:
                c.notes.update({k: v for k, v in local_scores.items() if "error" in k or k == "local_caption"})
                cap = local_scores.get("local_caption", "")
                scores["duplicate_penalty"] = max(scores["duplicate_penalty"], _caption_bad_signals(cap))
                # Do not let local caption override VLM later; it is only a weak fallback.
                scores["story_alignment"] = max(scores["story_alignment"], _clamp01(local_scores.get("story_alignment_local", 0.0)))
                scores["event_alignment"] = max(scores["event_alignment"], _clamp01(local_scores.get("event_alignment_local", 0.0)))
                scores["event_grounding"] = max(scores["event_grounding"], _clamp01(local_scores.get("event_alignment_local", 0.0)))
                scores["evidence_visibility"] = max(scores["evidence_visibility"], _clamp01(local_scores.get("evidence_visibility_local", 0.0)))
                scores["emotion_visibility"] = max(scores["emotion_visibility"], _clamp01(local_scores.get("emotion_visibility_local", 0.0)))
                scores["scene_alignment"] = max(scores["scene_alignment"], _clamp01(local_scores.get("scene_alignment_local", 0.0)))

            vlm_scores = self._vlm_frame_eval(frame, c)
            if vlm_scores and "vlm_error" not in vlm_scores:
                # V23: use VLM values directly rather than maxing with optimistic defaults.
                for key in [
                    "story_alignment", "event_alignment", "event_grounding", "evidence_visibility",
                    "emotion_visibility", "emotion_cause_visibility", "scene_alignment", "continuity",
                    "identity_consistency", "single_scene_score", "qa_score"
                ]:
                    if key in vlm_scores:
                        scores[key] = _clamp01(vlm_scores.get(key), scores[key])

                duplicate = any(bool(vlm_scores.get(k, False)) for k in ["duplicate_protagonist", "second_protagonist", "extra_character"])
                split = bool(vlm_scores.get("split_panel", False))
                exactly_one = bool(vlm_scores.get("exactly_one_protagonist", False))
                action_visible = bool(vlm_scores.get("action_visible", False))

                if duplicate:
                    scores["duplicate_penalty"] = max(scores["duplicate_penalty"], 0.85)
                    scores["identity_consistency"] = min(scores["identity_consistency"], 0.25)
                if split:
                    scores["duplicate_penalty"] = max(scores["duplicate_penalty"], 0.55)
                    scores["single_scene_score"] = min(scores["single_scene_score"], 0.2)
                if not exactly_one:
                    scores["duplicate_penalty"] = max(scores["duplicate_penalty"], 0.35)
                if not action_visible:
                    scores["action_missing_penalty"] = max(scores["action_missing_penalty"], 0.30)
                    scores["event_alignment"] = min(scores["event_alignment"], 0.35)

                c.notes["v23_vlm_judgment"] = vlm_scores
                c.notes["vlm_reason"] = vlm_scores.get("reason", "")
            elif vlm_scores and "vlm_error" in vlm_scores:
                c.notes["vlm_error"] = vlm_scores["vlm_error"]

            # V23: event and evidence correctness dominate. A beautiful but wrong image must lose.
            overall = (
                0.020 * scores["image_quality"]
                + 0.005 * scores["colorfulness"]
                + 0.170 * scores["identity_consistency"]
                + 0.210 * scores["story_alignment"]
                + 0.190 * scores["event_alignment"]
                + 0.135 * scores["event_grounding"]
                + 0.125 * scores["evidence_visibility"]
                + 0.075 * scores["emotion_visibility"]
                + 0.050 * scores["emotion_cause_visibility"]
                + 0.045 * scores["scene_alignment"]
                + 0.045 * scores["continuity"]
                + 0.055 * scores["single_scene_score"]
                + 0.035 * scores["qa_score"]
                + scores["variant_priority"]
                - scores["duplicate_penalty"]
                - scores["action_missing_penalty"]
                - scores["static_penalty"]
            )
            if is_ending:
                overall += 0.060 * scores["emotion_visibility"] + 0.040 * scores["emotion_cause_visibility"] + 0.025 * scores["continuity"]

            scores["overall"] = round(float(overall), 4)
            c.scores.update(scores)
            c.notes["v23_selection_reason"] = {
                "story_first_selection": True,
                "vlm_values_replace_defaults": True,
                "image_quality_weight_tiny": True,
                "variant_priority": scores["variant_priority"],
                "duplicate_penalty": scores["duplicate_penalty"],
                "action_missing_penalty": scores["action_missing_penalty"],
                "static_penalty": scores["static_penalty"],
                "ending_mode": bool(is_ending),
            }
            ranked.append(c)

        return sorted(ranked, key=lambda x: x.scores.get("overall", 0.0), reverse=True)

    def rerank_ending_candidates(self, final_frame, dce_plan, candidates):
        return self.rank_frame_candidates(final_frame, dce_plan, candidates, is_ending=True)

    def evaluate_sequence(self, dce_plan, emotion_arc, storyboard, images, questions, out_dir=None) -> Dict[str, Any]:
        if not images:
            return {"warning": "No images"}
        n = max(1, len(images))
        keys = [
            "image_quality", "colorfulness", "identity_consistency", "story_alignment", "event_alignment",
            "event_grounding", "evidence_visibility", "emotion_visibility", "emotion_cause_visibility",
            "scene_alignment", "continuity", "single_scene_score", "qa_score", "overall"
        ]
        avg = {k: round(sum(float(getattr(img, "scores", {}).get(k, 0.0)) for img in images) / n, 4) for k in keys}
        result = {
            "num_frames": len(images),
            "averages": avg,
            "selected_image_paths": [getattr(x, "image_path", "") for x in images],
            "questions": questions,
        }
        if self.save_contact_sheet and out_dir:
            try:
                result["contact_sheet_path"] = _make_contact_sheet_local([getattr(x, "image_path", "") for x in images], Path(out_dir) / "contact_sheet.png")
            except Exception as e:
                result["contact_sheet_error"] = str(e)
        return result
