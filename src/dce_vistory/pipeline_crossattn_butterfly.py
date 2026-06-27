from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List
import json
import traceback

from PIL import Image, ImageDraw

from .llm import build_llm, build_vlm
from .planner import DCEPlanner
from .causal_memory import DCEECausalMemoryStore
from .anchor_bank import DCEEAnchorBank
from .evaluator import DCEQAEvaluator
from .image_understanding import ImageUnderstandingModule
from .schema import PipelineResult, CandidateImage
from .prompts import QUALITY_SUFFIX, NEGATIVE_PROMPT
from .butterfly_adapter import ButterflyController
from .sdxl_cross_attention_generator import SDXLButterflyCrossAttentionGenerator


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


def _write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe_asdict(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def _image_path_exists(path: Any) -> bool:
    if not path:
        return False
    try:
        return Path(str(path)).exists()
    except Exception:
        return False


def _make_contact_sheet_force(image_paths: List[str], out_path: Path, cols: int = 3, thumb_size: tuple[int, int] = (384, 384), title: str = "DCEE-CausalVerse Visual Story") -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    valid_paths = [Path(str(p)) for p in image_paths if _image_path_exists(p)]
    if not valid_paths:
        raise ValueError("No valid image paths for contact sheet.")
    rows = (len(valid_paths) + cols - 1) // cols
    header_h = 54
    cell_w, cell_h = thumb_size
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h + header_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), title, fill=(0, 0, 0))
    for idx, img_path in enumerate(valid_paths):
        img = Image.open(img_path).convert("RGB")
        img.thumbnail((cell_w - 20, cell_h - 44))
        col = idx % cols
        row = idx // cols
        x0 = col * cell_w
        y0 = header_h + row * cell_h
        draw.rectangle([x0, y0, x0 + cell_w - 1, y0 + cell_h - 1], outline=(180, 180, 180))
        draw.text((x0 + 10, y0 + 10), f"Frame {idx + 1}", fill=(0, 0, 0))
        x = x0 + (cell_w - img.width) // 2
        y = y0 + 34 + (cell_h - 44 - img.height) // 2
        canvas.paste(img, (x, y))
    canvas.save(out_path)
    return str(out_path)


class CrossAttentionButterflyDCEViStoryPipeline:
    """Grounded incremental DCEE pipeline.

    Main change:
    sentence_1 -> frame_1 image
    sentence_2 (conditioned on previous story + frame_1 summary) -> frame_2 image
    ...
    This keeps every sentence image-friendly and makes frame generation more aligned with the story.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.llm = build_llm(cfg.get("llm", {}))
        self.vlm = build_vlm(cfg.get("vlm", {}))

        llm_cfg = cfg.get("llm", {})
        self.planner = DCEPlanner(
            self.llm,
            temperature=float(llm_cfg.get("temperature", 0.35)),
            max_tokens=int(llm_cfg.get("max_tokens", 1600)),
        )

        iu_cfg = cfg.get("image_understanding", {})
        self.image_understanding = ImageUnderstandingModule(
            iu_cfg.get("provider", "llm_caption"),
            self.llm,
            iu_cfg.get("caption_model", "Salesforce/blip-image-captioning-base"),
        )

        ev_cfg = cfg.get("evaluation", {})
        self.evaluator = DCEQAEvaluator(
            self.llm,
            self.vlm,
            use_vlm=bool(ev_cfg.get("use_vlm", False)),
            save_contact_sheet=bool(ev_cfg.get("save_contact_sheet", True)),
        )

        b_cfg = cfg.get("butterfly", {})
        self.controller = ButterflyController(
            b_cfg.get("quality_suffix", QUALITY_SUFFIX),
            b_cfg.get("negative_prompt", NEGATIVE_PROMPT),
            int(b_cfg.get("num_hypotheses", 3)),
        )

        img_cfg = cfg.get("image_generator", {})
        ad_cfg = cfg.get("cross_attention_adapters", {})
        self.image_generator = SDXLButterflyCrossAttentionGenerator(
            model_id=img_cfg.get("model_id", "stabilityai/stable-diffusion-xl-base-1.0"),
            device=img_cfg.get("device", "cuda"),
            width=int(img_cfg.get("width", 1024)),
            height=int(img_cfg.get("height", 1024)),
            num_inference_steps=int(img_cfg.get("num_inference_steps", 40)),
            guidance_scale=float(img_cfg.get("guidance_scale", 8.0)),
            seed=int(img_cfg.get("seed", 42)),
            adapter_ckpt=ad_cfg.get("adapter_ckpt"),
            enable_cpu_offload=bool(img_cfg.get("enable_cpu_offload", False)),
            character_tokens=int(ad_cfg.get("character_tokens", 8)),
            world_tokens=int(ad_cfg.get("world_tokens", 8)),
            emotion_tokens=int(ad_cfg.get("emotion_tokens", 8)),
            event_tokens=int(ad_cfg.get("event_tokens", 8)),
            evidence_tokens=int(ad_cfg.get("evidence_tokens", 8)),
        )

    def _strengthen_packet(self, packet, frame):
        packet.positive_prompt += (
            "\n\nRETRY CONTROL:"
            f"\n- Visualize EXACT sentence: {getattr(frame, 'story_sentence', '')}"
            f"\n- Current event must be visually obvious: {getattr(frame, 'event', '')}"
            f"\n- Required objects must be visible: {getattr(frame, 'must_show', [])}"
            f"\n- Visible cause of emotion: {getattr(frame, 'event_grounding', '')}"
            f"\n- Target emotion must be readable: {getattr(frame, 'emotion', '')}"
            "\n- Keep the same protagonist identity as previous frames."
            "\n- Use rich full color and a detailed background."
        )
        packet.negative_prompt += "; generic portrait, missing required objects, wrong protagonist identity, empty background, weak action, weak emotion"
        return packet

    def _save_core_plan_outputs(self, out_dir: Path, seed, abstract, dce_plan, emotion_arc, full_story, storyboard):
        _write_json(out_dir / "seed.json", seed)
        (out_dir / "abstract.txt").write_text(str(abstract), encoding="utf-8")
        _write_json(out_dir / "dcee_plan.json", dce_plan)
        _write_json(out_dir / "dce_plan.json", dce_plan)
        _write_json(out_dir / "emotion_arc.json", emotion_arc)
        _write_json(out_dir / "full_story.json", full_story)
        _write_json(out_dir / "storyboard.json", storyboard)

    def run(self, sample: Dict[str, Any], out_dir: Path) -> PipelineResult:
        out_dir = Path(out_dir)
        frames_dir = out_dir / "frames"
        ending_dir = out_dir / "ending_candidates"
        out_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        ending_dir.mkdir(parents=True, exist_ok=True)

        run_errors: List[Dict[str, Any]] = []
        image_summary = self.image_understanding.analyze(sample.get("image_path"), sample)
        seed = self.planner.build_seed(sample, image_summary)
        abstract = self.planner.generate_abstract(seed)
        dce_plan = self.planner.generate_dce_plan(seed, abstract)
        generation_policy = {
            "version": "V19",
            "mode": "protagonist_only_incremental",
            "protagonist_only": True,
            "no_secondary_characters": True,
            "allowed_visual_elements": [
                "protagonist",
                "protagonist props",
                "background objects",
                "weather",
                "lighting",
                "emotion cues"
            ],
            "blocked_story_entities": getattr(seed, "forbidden_ungrounded_entities", []),
            "reason": "When extra agents appear in story, SDXL may omit them or turn them into protagonist-like objects. V19 removes secondary agents from story generation."
        }
        _write_json(out_dir / "generation_policy_V19.json", generation_policy)
        total_frames = int(sample.get("num_frames", 6))
        emotion_arc = self.planner.generate_emotion_arc(seed, abstract, dce_plan, total_frames)

        memory = DCEECausalMemoryStore()
        try:
            memory.initialize(seed, dce_plan)
        except Exception as e:
            run_errors.append({"stage": "memory.initialize", "error": str(e), "traceback": traceback.format_exc()})
        anchor_bank = DCEEAnchorBank().build_from_seed_and_plan(seed, dce_plan)

        selected_images: List[CandidateImage] = []
        ending_candidates: List[CandidateImage] = []
        packet_log: List[Dict[str, Any]] = []
        memory_log: List[Dict[str, Any]] = []
        candidate_manifest: List[Dict[str, Any]] = []
        story_rows: List[Dict[str, Any]] = []
        storyboard: List[Any] = []

        img_cfg = self.cfg.get("image_generator", {})
        pipe_cfg = self.cfg.get("pipeline", {})
        num_candidates = int(img_cfg.get("num_candidates_per_frame", 2))
        num_ending_candidates = int(img_cfg.get("num_ending_candidates", 5))
        retry_enabled = bool(pipe_cfg.get("emotion_retry", True))
        emotion_threshold = float(pipe_cfg.get("emotion_visibility_threshold", 0.74))
        color_threshold = float(pipe_cfg.get("colorfulness_threshold", 0.35))
        event_threshold = float(pipe_cfg.get("event_grounding_threshold", 0.68))
        evidence_threshold = float(pipe_cfg.get("evidence_visibility_threshold", 0.66))

        style = sample.get("style", "full-color cinematic storybook illustration")
        if "color" not in style.lower():
            style = "full-color " + style

        previous_frame = None
        for idx in range(total_frames):
            frame_id = idx + 1
            is_last = idx == total_frames - 1
            target_dir = ending_dir if is_last else frames_dir
            candidate_count = num_ending_candidates if is_last else num_candidates

            story_step = self.planner.generate_story_step(seed, abstract, dce_plan, emotion_arc, story_rows, previous_frame, idx, total_frames)
            story_rows.append(story_step)
            frame = self.planner.story_step_to_frame(seed, dce_plan, emotion_arc, story_step, idx, total_frames)
            storyboard.append(frame)

            try:
                selected_memory = memory.select(frame, dce_plan, emotion_arc, strategy=pipe_cfg.get("memory_strategy", "adaptive_causal"))
            except Exception as e:
                selected_memory = {"error": str(e), "stage": "memory.select"}
                run_errors.append({"stage": f"memory.select.frame_{frame_id}", "error": str(e), "traceback": traceback.format_exc()})

            try:
                anchors = anchor_bank.select_for_frame(frame)
            except Exception as e:
                anchors = {"error": str(e), "stage": "anchor.select"}
                run_errors.append({"stage": f"anchor.select.frame_{frame_id}", "error": str(e), "traceback": traceback.format_exc()})

            packet = self.controller.create_packet(
                frame=frame,
                seed=seed,
                dce_plan=dce_plan,
                memory=selected_memory,
                style=style,
                previous_frame=previous_frame,
                anchors=anchors,
            )

            candidates = self.image_generator.generate_from_packet(packet=packet, frame_id=frame_id, out_dir=target_dir, num_candidates=candidate_count)
            ranked = self.evaluator.rerank_ending_candidates(frame, dce_plan, candidates) if is_last else self.evaluator.rank_frame_candidates(frame, dce_plan, candidates, is_ending=False)
            if not ranked:
                raise RuntimeError(f"No ranked candidates for frame {frame_id}")
            best = ranked[0]
            retried = False
            if retry_enabled and (
                best.scores.get("emotion_visibility", 0.0) < emotion_threshold
                or best.scores.get("colorfulness", 0.0) < color_threshold
                or best.scores.get("event_grounding", 0.0) < event_threshold
                or best.scores.get("evidence_visibility", 0.0) < evidence_threshold
            ):
                retried = True
                strong_packet = self._strengthen_packet(packet, frame)
                retry_candidates = self.image_generator.generate_from_packet(packet=strong_packet, frame_id=frame_id, out_dir=target_dir, num_candidates=max(2, candidate_count))
                retry_ranked = self.evaluator.rerank_ending_candidates(frame, dce_plan, retry_candidates) if is_last else self.evaluator.rank_frame_candidates(frame, dce_plan, retry_candidates, is_ending=False)
                if retry_ranked and retry_ranked[0].scores.get("overall", 0.0) >= best.scores.get("overall", 0.0):
                    best = retry_ranked[0]
                    ranked = retry_ranked
                    candidates = retry_candidates
                    packet = strong_packet

            selected_images.append(best)
            if is_last:
                ending_candidates = ranked
            try:
                memory.add(frame, best)
            except Exception as e:
                run_errors.append({"stage": f"memory.add.frame_{frame_id}", "error": str(e), "traceback": traceback.format_exc()})
            previous_frame = frame

            packet_log.append({"frame_id": frame_id, "packet": _safe_asdict(packet), "retried": retried})
            memory_log.append({"frame_id": frame_id, "memory": _safe_asdict(selected_memory)})
            candidate_manifest.append({"frame_id": frame_id, "candidates": _safe_asdict(candidates), "selected": _safe_asdict(best)})

            _write_json(out_dir / "full_story_partial.json", {"sentences": story_rows, "story_text": " ".join(x.get("sentence", "") for x in story_rows)})
            _write_json(out_dir / "storyboard_partial.json", storyboard)

        full_story = {"sentences": story_rows, "story_text": " ".join(x.get("sentence", "") for x in story_rows)}
        self._save_core_plan_outputs(out_dir, seed, abstract, dce_plan, emotion_arc, full_story, storyboard)

        _write_json(out_dir / "visual_control_packets.json", packet_log)
        _write_json(out_dir / "memory_log.json", memory_log)
        _write_json(out_dir / "candidate_manifest.json", candidate_manifest)
        _write_json(out_dir / "selected_images.json", selected_images)

        questions = self.evaluator.generate_questions(dce_plan, emotion_arc, storyboard)
        _write_json(out_dir / "evaluation_questions.json", questions)

        evaluation = self.evaluator.evaluate_sequence(dce_plan, emotion_arc, storyboard, selected_images, questions, out_dir=out_dir)
        if not evaluation.get("contact_sheet_path"):
            try:
                evaluation["contact_sheet_path"] = _make_contact_sheet_force([getattr(x, "image_path", "") for x in selected_images], out_dir / "contact_sheet.png")
            except Exception as e:
                evaluation["contact_sheet_error"] = f"{type(e).__name__}: {e}"
        if run_errors:
            evaluation["run_errors"] = run_errors
        _write_json(out_dir / "evaluation.json", evaluation)

        final_story_md = self._build_markdown(abstract, dce_plan, emotion_arc, full_story, storyboard, selected_images, ending_candidates, evaluation)
        if not str(final_story_md).strip():
            raise RuntimeError("Strict mode: final_story.md content is empty.")
        (out_dir / "final_story.md").write_text(final_story_md, encoding="utf-8")

        _write_json(out_dir / "output_manifest.json", {
            "contact_sheet": str(out_dir / "contact_sheet.png"),
            "final_story": str(out_dir / "final_story.md"),
            "evaluation": str(out_dir / "evaluation.json"),
            "selected_images": str(out_dir / "selected_images.json"),
            "candidate_manifest": str(out_dir / "candidate_manifest.json"),
            "storyboard": str(out_dir / "storyboard.json"),
            "full_story": str(out_dir / "full_story.json"),
            "dcee_plan": str(out_dir / "dcee_plan.json"),
            "generation_policy_V19": str(out_dir / "generation_policy_V19.json"),
            "has_contact_sheet": (out_dir / "contact_sheet.png").exists(),
            "num_selected_images": len(selected_images),
        })

        return PipelineResult(
            seed=seed,
            abstract=abstract,
            dce_plan=dce_plan,
            emotion_arc=emotion_arc,
            storyboard=storyboard,
            selected_images=selected_images,
            ending_candidates=ending_candidates,
            evaluation_questions=questions,
            evaluation=evaluation,
            final_story_markdown=final_story_md,
        )

    @staticmethod
    def _build_markdown(abstract, dce_plan, emotion_arc, full_story, storyboard, images, ending_candidates, evaluation):
        rows = (full_story or {}).get("sentences", []) if isinstance(full_story, dict) else []
        lines = [
            "# DCEE-CausalVerse Visual Story\n",
            "## Abstract\n",
            str(abstract) + "\n",
            "## Selected DCEE Plan\n",
            f"- Desire: {getattr(dce_plan, 'desire', '')}",
            f"- Conflict: {getattr(dce_plan, 'conflict', '')}",
            f"- Ending Emotion: {getattr(dce_plan, 'target_ending_emotion', '')}",
            "",
            "## Full Story\n",
        ]
        for idx, row in enumerate(rows, 1):
            lines.append(f"{idx}. {row.get('sentence', '')}")
        lines += ["", "## Emotion Arc\n", f"- States: {' → '.join([str(x) for x in getattr(emotion_arc, 'states', [])])}", f"- Intensities: {' → '.join([str(x) for x in getattr(emotion_arc, 'intensities', [])])}", ""]
        if evaluation.get("contact_sheet_path"):
            lines += ["## Contact Sheet\n", f"![Contact Sheet]({evaluation.get('contact_sheet_path')})", ""]
        lines.append("## Frames\n")
        for idx, (frame, image) in enumerate(zip(storyboard, images), 1):
            story_sentence = rows[idx - 1].get("sentence", "") if idx - 1 < len(rows) else getattr(frame, "story_sentence", "")
            lines += [
                f"### Frame {getattr(frame, 'frame_id', idx)}",
                f"![Frame {getattr(frame, 'frame_id', idx)}]({getattr(image, 'image_path', '')})",
                f"- Story sentence: {story_sentence}",
                f"- Event: {getattr(frame, 'event', '')}",
                f"- Event grounding: {getattr(frame, 'event_grounding', '')}",
                f"- Emotion: {getattr(frame, 'emotion', '')} ({getattr(frame, 'emotion_intensity', '')}/5)",
                f"- Required objects: {getattr(frame, 'must_show', [])}",
                f"- Scores: {getattr(image, 'scores', {})}",
                "",
            ]
        lines.append("## Sequence Evaluation\n")
        for k, v in evaluation.items():
            if k == "run_errors":
                continue
            lines.append(f"- {k}: {v}")
        return "\n".join(lines) + "\n"
