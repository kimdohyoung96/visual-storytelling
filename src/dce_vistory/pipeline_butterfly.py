from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from .llm import build_llm, build_vlm
from .planner import DCEPlanner
from .memory import NarrativeMemoryStore
from .image_generator import build_image_generator
from .evaluator import DCEQAEvaluator
from .image_understanding import ImageUnderstandingModule
from .schema import PipelineResult, CandidateImage
from .utils import save_json
from .prompts import QUALITY_SUFFIX, NEGATIVE_PROMPT
from .butterfly_adapter import ButterflyController


class ButterflyDCEViStoryPipeline:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.llm = build_llm(cfg.get("llm", {}))
        self.vlm = build_vlm(cfg.get("vlm", {}))
        self.planner = DCEPlanner(
            self.llm,
            temperature=float(cfg.get("llm", {}).get("temperature", 0.4)),
            max_tokens=int(cfg.get("llm", {}).get("max_tokens", 2500)),
        )
        iu_cfg = cfg.get("image_understanding", {})
        self.image_understanding = ImageUnderstandingModule(
            iu_cfg.get("provider", "llm_caption"),
            self.llm,
            iu_cfg.get("caption_model", "Salesforce/blip-image-captioning-base"),
        )
        self.image_generator = build_image_generator(cfg.get("image_generator", {}))
        self.evaluator = DCEQAEvaluator(
            self.llm,
            self.vlm,
            use_vlm=bool(cfg.get("evaluation", {}).get("use_vlm", True)),
            save_contact_sheet=bool(cfg.get("evaluation", {}).get("save_contact_sheet", True)),
        )
        bcfg = cfg.get("butterfly", {})
        self.controller = ButterflyController(
            quality_suffix=bcfg.get("quality_suffix", QUALITY_SUFFIX),
            negative_prompt=bcfg.get("negative_prompt", NEGATIVE_PROMPT),
            num_hypotheses=int(bcfg.get("num_hypotheses", 3)),
        )

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
        save_json([asdict(x) for x in storyboard], out_dir / "storyboard.json")

        memory_store = NarrativeMemoryStore()
        selected_images: List[CandidateImage] = []
        ending_candidates: List[CandidateImage] = []
        packet_log = []
        memory_log = []

        num_candidates = int(self.cfg.get("image_generator", {}).get("num_candidates_per_frame", 2))
        num_ending_candidates = int(self.cfg.get("image_generator", {}).get("num_ending_candidates", 4))
        style = sample.get("style", "cinematic storybook illustration")
        previous_frame = None

        for idx, frame in enumerate(storyboard):
            memory = memory_store.select(
                frame,
                dce_plan,
                emotion_arc,
                strategy=self.cfg.get("pipeline", {}).get("memory_strategy", "multi_slot"),
            )
            packet = self.controller.create_packet(
                frame=frame,
                seed=seed,
                dce_plan=dce_plan,
                memory=memory,
                style=style,
                previous_frame=previous_frame,
            )
            frame.prompt = packet.positive_prompt

            is_last = idx == len(storyboard) - 1
            target_dir = ending_dir if is_last else frames_dir
            candidate_count = num_ending_candidates if is_last else num_candidates
            candidates = self.image_generator.generate(packet.positive_prompt, frame.frame_id, target_dir, candidate_count)

            if is_last:
                ranked = self.evaluator.rerank_ending_candidates(frame, dce_plan, candidates)
                ending_candidates = ranked
            else:
                ranked = self.evaluator.rank_frame_candidates(frame, dce_plan, candidates, is_ending=False)

            best = ranked[0]
            selected_images.append(best)
            memory_store.add(frame, best)
            previous_frame = frame

            packet_log.append({
                "frame_id": frame.frame_id,
                "visual_control_packet": asdict(packet),
            })
            memory_log.append({
                "frame_id": frame.frame_id,
                "memory": memory,
                "selected_image": asdict(best),
                "all_candidates": [asdict(c) for c in ranked],
            })

        save_json(packet_log, out_dir / "visual_control_packets.json")
        save_json(memory_log, out_dir / "memory_log.json")
        save_json([asdict(x) for x in ending_candidates], out_dir / "ending_candidates.json")

        questions = self.evaluator.generate_questions(dce_plan, emotion_arc, storyboard)
        save_json(questions, out_dir / "eval_questions.json")
        evaluation = self.evaluator.evaluate_sequence(dce_plan, emotion_arc, storyboard, selected_images, questions, out_dir)
        save_json(evaluation, out_dir / "evaluation.json")

        final_story_md = self._build_markdown(seed, abstract, dce_plan, emotion_arc, storyboard, selected_images, ending_candidates, evaluation)
        (out_dir / "final_story.md").write_text(final_story_md, encoding="utf-8")

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
    def _build_markdown(seed, abstract, dce_plan, emotion_arc, storyboard, images, ending_candidates, evaluation) -> str:
        lines = []
        lines.append("# Butterfly DCE-ViStory Output\n")
        lines.append("## Story Abstract\n")
        lines.append(abstract + "\n")
        lines.append("## DCE Plan\n")
        lines.append(f"- Desire: {dce_plan.desire}")
        lines.append(f"- Conflict: {dce_plan.conflict}")
        lines.append(f"- Target Ending Emotion: {dce_plan.target_ending_emotion}\n")
        lines.append("## Emotion Arc\n")
        lines.append(f"- States: {' → '.join(emotion_arc.states)}")
        lines.append(f"- Intensities: {' → '.join([str(x) for x in emotion_arc.intensities])}\n")
        lines.append("## Selected Visual Story\n")
        for frame, image in zip(storyboard, images):
            lines.append(f"### Frame {frame.frame_id}: {frame.narrative_function}")
            lines.append(f"![Frame {frame.frame_id}]({image.image_path})")
            lines.append(f"- Event: {frame.event}")
            lines.append(f"- Emotion: {frame.emotion} ({frame.emotion_intensity}/5)")
            lines.append(f"- Location: {getattr(frame, 'scene_location', '')}")
            lines.append(f"- Time: {getattr(frame, 'time_of_day', '')}")
            lines.append(f"- Weather: {getattr(frame, 'weather', '')}")
            lines.append(f"- Atmosphere: {getattr(frame, 'atmosphere', '')}")
            lines.append(f"- Scores: {image.scores}\n")
        lines.append("## Ending Candidates\n")
        for cand in ending_candidates:
            lines.append(f"### Candidate {cand.candidate_id}")
            lines.append(f"![Candidate {cand.candidate_id}]({cand.image_path})")
            lines.append(f"- Scores: {cand.scores}\n")
        lines.append("## Evaluation\n")
        for k, v in evaluation.items():
            lines.append(f"- {k}: {v}")
        return "\n".join(lines)
