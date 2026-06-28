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



def _clear_previous_visual_outputs(out_dir: Path):
    """V25: avoid reading stale contact sheets or old candidate files from earlier runs."""
    out_dir = Path(out_dir)
    targets = [
        out_dir / "frames",
        out_dir / "ending_candidates",
    ]
    patterns = ["frame_*_cand_*.png", "frame_*.png", "*.tmp.png"]
    for folder in targets:
        if not folder.exists():
            continue
        for pattern in patterns:
            for p in folder.glob(pattern):
                try:
                    p.unlink()
                except Exception:
                    pass
    for name in ["contact_sheet.png", "candidate_manifest.json", "selected_images.json", "evaluation.json", "storyboard.json", "full_story.json"]:
        try:
            p = out_dir / name
            if p.exists():
                p.unlink()
        except Exception:
            pass


def _image_path_exists(path: Any) -> bool:
    if not path:
        return False
    try:
        return Path(str(path)).exists()
    except Exception:
        return False


def _make_contact_sheet_force(image_paths: List[str], out_path: Path, cols: int = 3, thumb_size: tuple[int, int] = (384, 384), title: str = "DCEE-CausalVerse V21 Visual Story") -> str:
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
    """V21 grounded incremental DCEE pipeline.

    Key V21 changes:
    1) The input image is treated as a hard identity anchor.
    2) Each frame is still generated with multiple candidates for selection, but every candidate must be a single-scene image.
    3) Prompting is explicitly story-locked: exact sentence, exact event, exact must-show objects, exact background, and exact protagonist identity.
    4) Previous selected frame continuity is injected into the current frame prompt.
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
            use_vlm=bool(ev_cfg.get("use_vlm", True)),
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

    def _identity_prompts(self, seed) -> tuple[str, str]:
        profiles = getattr(seed, "character_profiles", []) or []
        main_profile = profiles[0] if profiles else None
        pos = getattr(main_profile, "identity_anchor_prompt", "") or getattr(seed, "protagonist_identity_prompt", "")
        neg = getattr(main_profile, "negative_identity_prompt", "") or getattr(seed, "protagonist_negative_identity_prompt", "")
        return str(pos), str(neg)

    def _world_prompt(self, seed, frame) -> str:
        world_context = getattr(seed, "world_context", {}) or {}
        if not isinstance(world_context, dict):
            world_context = {}
        parts = [
            f"location: {getattr(frame, 'scene_location', '') or world_context.get('setting', '')}",
            f"weather: {getattr(frame, 'weather', '') or world_context.get('weather', '')}",
            f"time_of_day: {getattr(frame, 'time_of_day', '') or world_context.get('time_of_day', '')}",
            f"background: {getattr(frame, 'environment_details', []) or world_context.get('background_objects', [])}",
        ]
        return "; ".join([str(x) for x in parts if str(x).strip()])

    def _ground_packet_to_story_and_image(self, packet, frame, seed, previous_best: CandidateImage | None):
        identity_pos, identity_neg = self._identity_prompts(seed)
        protagonist_short = getattr(seed, "protagonist_visual_short", getattr(seed, "protagonist", "protagonist"))
        source_image_path = getattr(seed, "source_image_path", "")
        must_show = getattr(frame, "must_show", [])
        bg = getattr(frame, "environment_details", [])
        world_prompt = self._world_prompt(seed, frame)
        prev_note = ""
        if previous_best is not None:
            prev_note = (
                f"\n- Maintain continuity with the previous selected frame while updating the new action and emotion only."
                f"\n- Previous selected frame image path: {getattr(previous_best, 'image_path', '')}"
            )

        packet.positive_prompt += (
            "\n\nV25 DCEE STAGE + UNCROPPED EVENT-FRAME GROUNDING RULES:"
            f"\n- Render ONE single coherent scene only; never use split panels or multiple moments."\
            f"\n- Use centered full-body or mostly full-body composition with safe margins; do not crop head, face, paws, hands, feet, or required objects."
            f"\n- ConsiStory-lite subject anchor: preserve only protagonist identity across frames; do not copy background/layout mistakes."\
            f"\n- Render the SAME protagonist identity: {identity_pos}"
            f"\n- Use the input image as a hard reference anchor. If the subject is {protagonist_short}, keep {protagonist_short} in this frame."
            f"\n- Exact story sentence to render: {getattr(frame, 'story_sentence', '')}"
            f"\n- Image-friendly rendering sentence: {getattr(frame, 'image_sentence', '')}"
            f"\n- Exact event to show, with action visibly readable: {getattr(frame, 'event', '')}"
            f"\n- Exact visible cause/evidence: {getattr(frame, 'event_grounding', '')}"
            f"\n- Required visible objects only, no extra props: {must_show}"
            f"\n- Background/world that must remain grounded: {world_prompt}"
            f"\n- Emotion must be clearly readable: {getattr(frame, 'emotion', '')}; {getattr(frame, 'emotion_visual_rule', '')}"
            f"\n- Keep all required environment/background elements visible when relevant: {bg}"
            f"\n- Source input image path: {source_image_path}"
            f"{prev_note}"\
            f"\n- IMPORTANT: previous generated frames are text continuity only; do not copy any duplicate-subject artifact from them."
        )
        packet.negative_prompt += (
            "; split screen; diptych; triptych; comic panel; storyboard sheet; collage; multiple moments in one frame"
            "; extra character; extra animal; duplicated protagonist; no second protagonist; no duplicate protagonist; two bears; multiple bears; wrong protagonist identity; wrong species; wrong fur color; missing background"
            "; missing required object; unrelated object; unrelated prop; generic portrait; wrong fur color; weak emotion"
        )
        if identity_neg:
            packet.negative_prompt += f"; {identity_neg}"
        for attr, value in {
            "source_reference_image_path": source_image_path,
            "reference_image_path": source_image_path,
            "continuity_image_path": getattr(previous_best, 'image_path', '') if previous_best is not None else "",
        }.items():
            try:
                setattr(packet, attr, value)
            except Exception:
                pass
        return packet

    def _strengthen_packet(self, packet, frame, seed, previous_best: CandidateImage | None):
        packet = self._ground_packet_to_story_and_image(packet, frame, seed, previous_best)
        packet.positive_prompt += (
            "\n\nRETRY CONTROL:"
            f"\n- Make the story sentence visually obvious: {getattr(frame, 'story_sentence', '')}"
            f"\n- Make the current event visually obvious: {getattr(frame, 'event', '')}"
            f"\n- Show exact required objects clearly: {getattr(frame, 'must_show', [])}"
            f"\n- Show exact protagonist identity clearly: {getattr(frame, 'character_reference_prompt', '')}"
            f"\n- Show exact background and weather clearly: {getattr(frame, 'scene_location', '')}, {getattr(frame, 'weather', '')}, {getattr(frame, 'environment_details', [])}"
            f"\n- Show target emotion clearly: {getattr(frame, 'emotion', '')}"
            "\n- Use rich full color and a detailed but grounded background."
        )
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
        _clear_previous_visual_outputs(out_dir)

        run_errors: List[Dict[str, Any]] = []
        image_summary = self.image_understanding.analyze(sample.get("image_path"), sample)
        seed = self.planner.build_seed(sample, image_summary)
        abstract = self.planner.generate_abstract(seed)
        dce_plan = self.planner.generate_dce_plan(seed, abstract)
        generation_policy = {
            "version": "V25",
            "mode": "dcee_stage_uncropped_event_grounded_selector",
            "protagonist_only": True,
            "single_scene_per_frame": True,
            "multiple_candidates_for_selection": True,
            "input_image_is_hard_identity_anchor": True,
            "story_sentence_locked": True,
            "previous_selected_frame_used_for_text_continuity": True,
            "consistory_lite_subject_anchor": True,
            "dcee_event_contract_per_frame": True,
            "english_sdxl_visual_prompt": True,
            "pairwise_vlm_candidate_selection": True,
            "clear_stale_outputs_before_run": True,
            "prompt_hash_seed_policy": True,
            "uncropped_fullbody_composition": True,
            "vlm_story_event_candidate_selector": True,
            "previous_generated_images_not_used_as_ip_reference_by_default": True,
            "allowed_visual_elements": [
                "protagonist", "protagonist props", "grounded background objects", "weather", "lighting", "emotion cues"
            ],
            "blocked_story_entities": getattr(seed, "forbidden_ungrounded_entities", []),
            "reason": "V25 adds DCEE stage scaffolding, uncropped full-body composition constraints, prompt-hash seed policy, stale-output cleanup, and stronger crop/duplicate penalties."
        }
        _write_json(out_dir / "generation_policy_V25.json", generation_policy)
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
        previous_best: CandidateImage | None = None
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
            packet = self._ground_packet_to_story_and_image(packet, frame, seed, previous_best)

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
                strong_packet = self._strengthen_packet(packet, frame, seed, previous_best)
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
            try:
                setattr(frame, "selected_image_path", getattr(best, "image_path", ""))
            except Exception:
                pass
            previous_frame = frame
            previous_best = best

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
            "generation_policy_V23": str(out_dir / "generation_policy_V25.json"),
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
        lines += [
            "",
            "## Emotion Arc\n",
            f"- States: {' → '.join([str(x) for x in getattr(emotion_arc, 'states', [])])}",
            f"- Intensities: {' → '.join([str(x) for x in getattr(emotion_arc, 'intensities', [])])}",
            "",
        ]
        if evaluation.get("contact_sheet_path"):
            lines += ["## Contact Sheet\n", f"![Contact Sheet]({evaluation.get('contact_sheet_path')})", ""]
        lines.append("## Frames\n")
        for idx, (frame, image) in enumerate(zip(storyboard, images), 1):
            story_sentence = rows[idx - 1].get("sentence", "") if idx - 1 < len(rows) else getattr(frame, "story_sentence", "")
            lines += [
                f"### Frame {getattr(frame, 'frame_id', idx)}",
                f"![Frame {getattr(frame, 'frame_id', idx)}]({getattr(image, 'image_path', '')})",
                f"- Story sentence: {story_sentence}",
                f"- Image sentence: {getattr(frame, 'image_sentence', '')}",
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
