from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from .llm import build_llm, build_vlm
from .planner import DCEPlanner
from .memory import NarrativeMemoryStore
from .evaluator import DCEQAEvaluator
from .image_understanding import ImageUnderstandingModule
from .schema import PipelineResult, CandidateImage
from .utils import save_json
from .prompts import QUALITY_SUFFIX, NEGATIVE_PROMPT
from .butterfly_adapter import ButterflyController
from .sdxl_cross_attention_generator import SDXLButterflyCrossAttentionGenerator


class CrossAttentionButterflyDCEViStoryPipeline:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.llm = build_llm(cfg.get("llm", {}))
        self.vlm = build_vlm(cfg.get("vlm", {}))
        self.planner = DCEPlanner(self.llm, temperature=float(cfg.get("llm", {}).get("temperature", 0.4)), max_tokens=int(cfg.get("llm", {}).get("max_tokens", 2500)))
        iu_cfg = cfg.get("image_understanding", {})
        self.image_understanding = ImageUnderstandingModule(iu_cfg.get("provider", "llm_caption"), self.llm, iu_cfg.get("caption_model", "Salesforce/blip-image-captioning-base"))
        self.evaluator = DCEQAEvaluator(self.llm, self.vlm, use_vlm=bool(cfg.get("evaluation", {}).get("use_vlm", True)), save_contact_sheet=bool(cfg.get("evaluation", {}).get("save_contact_sheet", True)))
        bcfg = cfg.get("butterfly", {})
        self.controller = ButterflyController(quality_suffix=bcfg.get("quality_suffix", QUALITY_SUFFIX), negative_prompt=bcfg.get("negative_prompt", NEGATIVE_PROMPT), num_hypotheses=int(bcfg.get("num_hypotheses", 3)))
        icfg = cfg.get("image_generator", {})
        acfg = cfg.get("cross_attention_adapters", {})
        self.image_generator = SDXLButterflyCrossAttentionGenerator(
            model_id=icfg.get("model_id", "stabilityai/stable-diffusion-xl-base-1.0"),
            device=icfg.get("device", "cuda"), width=int(icfg.get("width", 1024)), height=int(icfg.get("height", 1024)),
            num_inference_steps=int(icfg.get("num_inference_steps", 40)), guidance_scale=float(icfg.get("guidance_scale", 8.0)),
            seed=int(icfg.get("seed", 42)), adapter_ckpt=acfg.get("adapter_ckpt"), enable_cpu_offload=bool(icfg.get("enable_cpu_offload", False)),
            character_tokens=int(acfg.get("character_tokens", 8)), world_tokens=int(acfg.get("world_tokens", 8)),
            emotion_tokens=int(acfg.get("emotion_tokens", 8)), event_tokens=int(acfg.get("event_tokens", 8)),
        )

    def _strengthen_packet(self, packet, frame):
        packet.positive_prompt += f"""

REINFORCE EMOTION AND COLOR:
The protagonist's emotion MUST be unmistakable.
Show the emotion clearly through:
- face: {frame.facial_cue}
- body: {frame.body_cue}
- environment: {getattr(frame, 'environment_details', [])}
- weather: {getattr(frame, 'weather', '')}
- color palette: {getattr(frame, 'color_palette', 'rich full color')}
- shot type: {getattr(frame, 'shot_type', 'medium shot')}

The scene must explicitly show the event:
{frame.event}

The image must show the visual evidence of why the protagonist feels {frame.emotion}:
{getattr(frame, 'emotion_evidence', [])}

This MUST be a full-color image with rich, emotionally meaningful colors.
Never produce grayscale, monochrome, black-and-white, pencil sketch, or line-art-only output.
"""
        packet.negative_prompt += ", grayscale, monochrome, black and white, pencil sketch, line art only, weak expression, emotionless face, flat pose, colorless image, empty background"
        meta = packet.control_metadata or {}
        meta["emotion_text"] = meta.get("emotion_text", "") + f"; emotion must be unmistakably visible through face, body, color, lighting and world state; emotion evidence: {getattr(frame, 'emotion_evidence', [])}"
        meta["world_text"] = meta.get("world_text", "") + f"; full-color world state; weather: {getattr(frame, 'weather', '')}; palette: {getattr(frame, 'color_palette', '')}; atmosphere: {getattr(frame, 'atmosphere', '')}"
        meta["event_text"] = meta.get("event_text", "") + f"; event must be clearly visible: {frame.event}"
        packet.control_metadata = meta
        packet.adapter_weights["emotion_adapter"] = min(0.55, packet.adapter_weights.get("emotion_adapter", 0.30) + 0.10)
        packet.adapter_weights["world_adapter"] = min(0.40, packet.adapter_weights.get("world_adapter", 0.25) + 0.05)
        packet.adapter_weights["event_adapter"] = min(0.35, packet.adapter_weights.get("event_adapter", 0.15) + 0.05)
        return packet

    def run(self, sample: Dict[str, Any], out_dir: Path) -> PipelineResult:
        out_dir = Path(out_dir)
        frames_dir = out_dir / "frames"
        ending_dir = out_dir / "ending_candidates"
        out_dir.mkdir(parents=True, exist_ok=True)
        image_summary = self.image_understanding.analyze(sample.get("image_path"), sample)
        seed = self.planner.build_seed(sample, image_summary)
        abstract = self.planner.generate_abstract(seed)
        dce_plan = self.planner.generate_dce_plan(seed, abstract)
        emotion_arc = self.planner.generate_emotion_arc(seed, abstract, dce_plan, int(sample.get("num_frames", 6)))
        storyboard = self.planner.generate_storyboard(seed, abstract, dce_plan, emotion_arc)
        save_json(asdict(seed), out_dir / "seed.json")
        (out_dir / "abstract.txt").write_text(abstract, encoding="utf-8")
        save_json(asdict(dce_plan), out_dir / "dce_plan.json")
        save_json(asdict(emotion_arc), out_dir / "emotion_arc.json")
        save_json([self._frame_to_dict(x) for x in storyboard], out_dir / "storyboard.json")
        memory_store = NarrativeMemoryStore()
        selected_images: List[CandidateImage] = []
        ending_candidates: List[CandidateImage] = []
        packet_log = []
        memory_log = []
        num_candidates = int(self.cfg.get("image_generator", {}).get("num_candidates_per_frame", 3))
        num_ending_candidates = int(self.cfg.get("image_generator", {}).get("num_ending_candidates", 5))
        retry_enabled = bool(self.cfg.get("pipeline", {}).get("emotion_retry", True))
        emotion_threshold = float(self.cfg.get("pipeline", {}).get("emotion_visibility_threshold", 0.78))
        color_threshold = float(self.cfg.get("pipeline", {}).get("colorfulness_threshold", 0.35))
        style = sample.get("style", "full-color cinematic storybook illustration")
        if "color" not in style.lower():
            style = "full-color " + style
        previous_frame = None
        for idx, frame in enumerate(storyboard):
            memory = memory_store.select(frame, dce_plan, emotion_arc, strategy=self.cfg.get("pipeline", {}).get("memory_strategy", "multi_slot"))
            packet = self.controller.create_packet(frame=frame, seed=seed, dce_plan=dce_plan, memory=memory, style=style, previous_frame=previous_frame)
            is_last = idx == len(storyboard) - 1
            target_dir = ending_dir if is_last else frames_dir
            candidate_count = num_ending_candidates if is_last else num_candidates
            candidates = self.image_generator.generate_from_packet(packet=packet, frame_id=frame.frame_id, out_dir=target_dir, num_candidates=candidate_count)
            ranked = self.evaluator.rerank_ending_candidates(frame, dce_plan, candidates) if is_last else self.evaluator.rank_frame_candidates(frame, dce_plan, candidates, is_ending=False)
            best = ranked[0]
            retried = False
            if retry_enabled and (best.scores.get("emotion_visibility", 0.0) < emotion_threshold or best.scores.get("colorfulness", 0.0) < color_threshold):
                retried = True
                strong_packet = self._strengthen_packet(packet, frame)
                retry_candidates = self.image_generator.generate_from_packet(packet=strong_packet, frame_id=frame.frame_id, out_dir=target_dir, num_candidates=max(2, candidate_count // 2))
                retry_ranked = self.evaluator.rerank_ending_candidates(frame, dce_plan, retry_candidates) if is_last else self.evaluator.rank_frame_candidates(frame, dce_plan, retry_candidates, is_ending=False)
                if retry_ranked and retry_ranked[0].scores.get("overall", 0.0) > best.scores.get("overall", 0.0):
                    ranked, best, packet = retry_ranked, retry_ranked[0], strong_packet
            best.notes["retried_for_emotion_or_color"] = retried
            if is_last:
                ending_candidates = ranked
            selected_images.append(best)
            memory_store.add(frame, best)
            previous_frame = frame
            packet_log.append({"frame_id": frame.frame_id, "visual_control_packet": asdict(packet), "retried": retried})
            memory_log.append({"frame_id": frame.frame_id, "memory": memory, "selected_image": asdict(best), "all_candidates": [asdict(c) for c in ranked]})
        save_json(packet_log, out_dir / "visual_control_packets.json")
        save_json(memory_log, out_dir / "memory_log.json")
        save_json([asdict(x) for x in ending_candidates], out_dir / "ending_candidates.json")
        questions = self.evaluator.generate_questions(dce_plan, emotion_arc, storyboard)
        save_json(questions, out_dir / "eval_questions.json")
        evaluation = self.evaluator.evaluate_sequence(dce_plan, emotion_arc, storyboard, selected_images, questions, out_dir)
        save_json(evaluation, out_dir / "evaluation.json")
        final_story_md = self._build_markdown(abstract, dce_plan, emotion_arc, storyboard, selected_images, ending_candidates, evaluation)
        (out_dir / "final_story.md").write_text(final_story_md, encoding="utf-8")
        return PipelineResult(seed=seed, abstract=abstract, dce_plan=dce_plan, emotion_arc=emotion_arc, storyboard=storyboard, selected_images=selected_images, ending_candidates=ending_candidates, evaluation_questions=questions, evaluation=evaluation, final_story_markdown=final_story_md)

    @staticmethod
    def _frame_to_dict(frame) -> Dict[str, Any]:
        data = asdict(frame)
        for key in ["shot_type", "camera_distance", "color_palette", "lighting_style", "must_show", "emotion_evidence"]:
            data[key] = getattr(frame, key, None)
        return data

    @staticmethod
    def _build_markdown(abstract, dce_plan, emotion_arc, storyboard, images, ending_candidates, evaluation) -> str:
        lines = ["# Final Emotion-Visible Visual Story\n", "## Abstract\n", abstract + "\n", "## DCE Plan\n"]
        lines += [f"- Desire: {dce_plan.desire}", f"- Conflict: {dce_plan.conflict}", f"- Turning Point: {dce_plan.turning_point}", f"- Ending Emotion: {dce_plan.target_ending_emotion}", f"- Ending State: {dce_plan.ending_state}\n"]
        lines += ["## Emotion Arc\n", f"- States: {' → '.join(emotion_arc.states)}", f"- Intensities: {' → '.join([str(x) for x in emotion_arc.intensities])}\n", "## Frames\n"]
        for frame, image in zip(storyboard, images):
            lines += [f"### Frame {frame.frame_id}: {frame.narrative_function}", f"![Frame {frame.frame_id}]({image.image_path})", f"- Event: {frame.event}", f"- Emotion: {frame.emotion} ({frame.emotion_intensity}/5)", f"- Emotion Evidence: {getattr(frame, 'emotion_evidence', [])}", f"- Shot: {getattr(frame, 'shot_type', '')}", f"- Weather: {getattr(frame, 'weather', '')}", f"- Palette: {getattr(frame, 'color_palette', '')}", f"- Must Show: {getattr(frame, 'must_show', [])}", f"- Scores: {image.scores}", f"- Notes: {image.notes}\n"]
        lines.append("## Ending Candidates\n")
        for cand in ending_candidates:
            lines += [f"### Candidate {cand.candidate_id}", f"![Candidate {cand.candidate_id}]({cand.image_path})", f"- Scores: {cand.scores}", f"- Notes: {cand.notes}\n"]
        lines.append("## Evaluation\n")
        for k, v in evaluation.items():
            lines.append(f"- {k}: {v}")
        return "\n".join(lines)
