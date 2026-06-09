from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from .llm import build_llm, build_vlm
from .planner import DCEPlanner
from .memory import NarrativeMemoryStore
from .image_generator import build_image_generator
from .evaluator import DCEQAEvaluator
from .image_understanding import ImageUnderstandingModule
from .prompts import frame_prompt
from .schema import PipelineResult, CandidateImage
from .utils import save_json


class DCEViStoryPipeline:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.llm = build_llm(cfg.get('llm', {}))
        self.vlm = build_vlm(cfg.get('vlm', {}))
        self.planner = DCEPlanner(self.llm, temperature=float(cfg.get('llm', {}).get('temperature', 0.4)), max_tokens=int(cfg.get('llm', {}).get('max_tokens', 2500)))
        iu_cfg = cfg.get('image_understanding', {})
        self.image_understanding = ImageUnderstandingModule(iu_cfg.get('provider', 'llm_caption'), self.llm, iu_cfg.get('caption_model', 'Salesforce/blip-image-captioning-base'))
        self.image_generator = build_image_generator(cfg.get('image_generator', {}))
        self.evaluator = DCEQAEvaluator(self.llm, self.vlm, use_vlm=bool(cfg.get('evaluation', {}).get('use_vlm', True)), save_contact_sheet=bool(cfg.get('evaluation', {}).get('save_contact_sheet', True)))

    def plan_only(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image_summary = self.image_understanding.analyze(sample.get('image_path'), sample)
        seed = self.planner.build_seed(sample, image_summary)
        abstract = self.planner.generate_abstract(seed)
        dce_plan = self.planner.generate_dce_plan(seed, abstract)
        emotion_arc = self.planner.generate_emotion_arc(seed, abstract, dce_plan, int(sample.get('num_frames', 6)))
        storyboard = self.planner.generate_storyboard(seed, abstract, dce_plan, emotion_arc)
        return {'seed': asdict(seed), 'abstract': abstract, 'dce_plan': asdict(dce_plan), 'emotion_arc': asdict(emotion_arc), 'storyboard': [asdict(x) for x in storyboard]}

    def run(self, sample: Dict[str, Any], out_dir: Path) -> PipelineResult:
        out_dir = Path(out_dir)
        frames_dir = out_dir / 'frames'
        ending_dir = out_dir / 'ending_candidates'
        out_dir.mkdir(parents=True, exist_ok=True)

        image_summary = self.image_understanding.analyze(sample.get('image_path'), sample)
        seed = self.planner.build_seed(sample, image_summary)
        abstract = self.planner.generate_abstract(seed)
        dce_plan = self.planner.generate_dce_plan(seed, abstract)
        emotion_arc = self.planner.generate_emotion_arc(seed, abstract, dce_plan, int(sample.get('num_frames', 6)))
        storyboard = self.planner.generate_storyboard(seed, abstract, dce_plan, emotion_arc)

        save_json(asdict(seed), out_dir / 'seed.json')
        (out_dir / 'abstract.txt').write_text(abstract, encoding='utf-8')
        save_json(asdict(dce_plan), out_dir / 'dce_plan.json')
        save_json(asdict(emotion_arc), out_dir / 'emotion_arc.json')
        save_json([asdict(x) for x in storyboard], out_dir / 'storyboard.json')

        memory_store = NarrativeMemoryStore()
        selected_images: List[CandidateImage] = []
        memory_log = []
        strategy = self.cfg.get('pipeline', {}).get('memory_strategy', 'multi_slot')
        num_candidates = int(self.cfg.get('image_generator', {}).get('num_candidates_per_frame', 1))
        num_ending_candidates = int(self.cfg.get('image_generator', {}).get('num_ending_candidates', 4))
        style = sample.get('style', 'cinematic storybook illustration')
        input_image_summary = asdict(image_summary) if image_summary else None
        ending_candidates: List[CandidateImage] = []

        for idx, frame in enumerate(storyboard):
            memory = memory_store.select(frame, dce_plan, emotion_arc, strategy=strategy)
            prompt = frame_prompt(asdict(frame), asdict(dce_plan), asdict(emotion_arc), memory, style, input_image_summary if self.cfg.get('pipeline', {}).get('include_input_image_summary', True) else None)
            frame.prompt = prompt
            is_last = idx == len(storyboard) - 1
            current_num_candidates = num_ending_candidates if is_last else num_candidates
            target_dir = ending_dir if is_last else frames_dir
            candidates = self.image_generator.generate(prompt, frame.frame_id, target_dir, current_num_candidates)
            if is_last:
                ending_candidates = self.evaluator.rerank_ending_candidates(frame, dce_plan, candidates)
                best = ending_candidates[0]
            else:
                for cand in candidates:
                    cand.scores = {'overall': 0.70 + min(0.20, frame.conflict_level * 0.03), 'emotion_alignment': 0.70}
                best = max(candidates, key=lambda c: c.scores.get('overall', 0.0))
            selected_images.append(best)
            memory_store.add(frame, best)
            memory_log.append({'frame_id': frame.frame_id, 'memory': memory, 'selected_image': asdict(best), 'all_candidates': [asdict(c) for c in (ending_candidates if is_last else candidates)]})

        save_json(memory_log, out_dir / 'memory_log.json')
        save_json([asdict(x) for x in ending_candidates], out_dir / 'ending_candidates.json')

        questions = self.evaluator.generate_questions(dce_plan, emotion_arc, storyboard)
        save_json(questions, out_dir / 'eval_questions.json')
        evaluation = self.evaluator.evaluate_sequence(dce_plan, emotion_arc, storyboard, selected_images, questions, out_dir)
        save_json(evaluation, out_dir / 'evaluation.json')

        final_story_md = self._build_markdown(seed, abstract, dce_plan, emotion_arc, storyboard, selected_images, ending_candidates, evaluation)
        (out_dir / 'final_story.md').write_text(final_story_md, encoding='utf-8')

        return PipelineResult(seed=seed, abstract=abstract, dce_plan=dce_plan, emotion_arc=emotion_arc, storyboard=storyboard, selected_images=selected_images, ending_candidates=ending_candidates, evaluation_questions=questions, evaluation=evaluation, final_story_markdown=final_story_md)

    @staticmethod
    def _build_markdown(seed, abstract, dce_plan, emotion_arc, storyboard, images, ending_candidates, evaluation) -> str:
        lines = []
        lines.append('# DCE-ViStory Output\n')
        lines.append('## Input Summary\n')
        lines.append(f'- Text Prompt: {seed.text_prompt}')
        lines.append(f'- Protagonist: {seed.protagonist}')
        lines.append(f'- Target Ending Emotion: {seed.target_ending_emotion}')
        lines.append(f'- Genre: {seed.genre}')
        lines.append(f'- Style: {seed.style}')
        if seed.image_summary:
            lines.append(f'- Input Image Caption: {seed.image_summary.caption}')
            lines.append(f'- Input Image Setting: {seed.image_summary.setting}')
            lines.append(f'- Input Image Mood: {seed.image_summary.mood}')
        lines.append('')
        lines.append('## Story Abstract\n')
        lines.append(abstract + '\n')
        lines.append('## DCE Plan\n')
        lines.append(f'- Protagonist: {dce_plan.protagonist}')
        lines.append(f'- Desire: {dce_plan.desire}')
        lines.append(f'- Fear: {dce_plan.fear}')
        lines.append(f'- Misbelief: {dce_plan.misbelief}')
        lines.append(f'- Obstacle: {dce_plan.obstacle}')
        lines.append(f'- Conflict: {dce_plan.conflict}')
        lines.append(f"- Event Spine: {' | '.join(dce_plan.event_spine)}")
        lines.append(f'- Turning Point: {dce_plan.turning_point}')
        lines.append(f'- Ending State: {dce_plan.ending_state}')
        lines.append(f'- Theme: {dce_plan.moral_or_theme}\n')
        lines.append('## Emotion Arc\n')
        lines.append(f"- States: {' → '.join(emotion_arc.states)}")
        lines.append(f"- Intensities: {' → '.join([str(x) for x in emotion_arc.intensities])}")
        lines.append(f'- Rationale: {emotion_arc.rationale}\n')
        lines.append('## Storyboard and Selected Images\n')
        for frame, image in zip(storyboard, images):
            lines.append(f'### Frame {frame.frame_id}: {frame.narrative_function}')
            lines.append(f'![Frame {frame.frame_id}]({image.image_path})')
            lines.append(f'- Caption: {frame.caption}')
            lines.append(f'- Event: {frame.event}')
            lines.append(f'- Emotion: {frame.emotion} (intensity {frame.emotion_intensity}/5)')
            lines.append(f'- Conflict Level: {frame.conflict_level}')
            lines.append(f'- Facial Cue: {frame.facial_cue}')
            lines.append(f'- Body Cue: {frame.body_cue}')
            lines.append(f'- Event Cue: {frame.event_cue}')
            lines.append(f'- Scene Cue: {frame.scene_cue}')
            lines.append(f'- Cinematic Cue: {frame.cinematic_cue}')
            lines.append('')
        lines.append('## Ending Candidates\n')
        for cand in ending_candidates:
            lines.append(f'### Ending Candidate {cand.candidate_id}')
            lines.append(f'![Ending Candidate {cand.candidate_id}]({cand.image_path})')
            for k, v in cand.scores.items():
                lines.append(f'- {k}: {v}')
            if cand.notes:
                lines.append(f'- notes: {cand.notes}')
            lines.append('')
        lines.append('## Evaluation\n')
        for k, v in evaluation.items():
            lines.append(f'- {k}: {v}')
        return '\n'.join(lines)
