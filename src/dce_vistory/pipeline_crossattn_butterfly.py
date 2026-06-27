
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List
import json
import traceback
import copy

from PIL import Image, ImageDraw

from .llm import build_llm, build_vlm
from .planner import DCEPlanner
from .causal_memory import DCEECausalMemoryStore
from .anchor_bank import DCEEAnchorBank
from .evaluator import DCEQAEvaluator
from .image_understanding import ImageUnderstandingModule
from .schema import PipelineResult, CandidateImage
from .utils import save_json
from .prompts import QUALITY_SUFFIX, NEGATIVE_PROMPT
from .butterfly_adapter import ButterflyController
from .sdxl_cross_attention_generator import SDXLButterflyCrossAttentionGenerator
from .story_bible import build_story_bible


def _safe_asdict(obj: Any):
    if is_dataclass(obj):
        d = asdict(obj)
        d.update({k: v for k, v in getattr(obj, '__dict__', {}).items() if k not in d})
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
    path.write_text(json.dumps(_safe_asdict(obj), ensure_ascii=False, indent=2), encoding='utf-8')


def _image_path_exists(path: Any) -> bool:
    if not path:
        return False
    try:
        return Path(str(path)).exists()
    except Exception:
        return False


def _make_contact_sheet_force(image_paths: List[str], out_path: Path, cols: int = 3, thumb_size: tuple[int, int] = (384, 384), title: str = 'DCEE-CausalVerse Visual Story') -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    valid_paths = [Path(str(p)) for p in image_paths if _image_path_exists(p)]
    if not valid_paths:
        raise ValueError(f'No valid image paths for contact sheet: {image_paths}')
    rows = (len(valid_paths) + cols - 1) // cols
    header_h = 54
    cell_w, cell_h = thumb_size
    canvas = Image.new('RGB', (cols * cell_w, rows * cell_h + header_h), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), title, fill=(0, 0, 0))
    for idx, img_path in enumerate(valid_paths):
        img = Image.open(img_path).convert('RGB')
        img.thumbnail((cell_w - 20, cell_h - 44))
        col = idx % cols
        row = idx // cols
        x0 = col * cell_w
        y0 = header_h + row * cell_h
        draw.rectangle([x0, y0, x0 + cell_w - 1, y0 + cell_h - 1], outline=(180, 180, 180))
        draw.text((x0 + 10, y0 + 10), f'Frame {idx + 1}', fill=(0, 0, 0))
        x = x0 + (cell_w - img.width) // 2
        y = y0 + 34 + (cell_h - 44 - img.height) // 2
        canvas.paste(img, (x, y))
    canvas.save(out_path)
    return str(out_path)


class CrossAttentionButterflyDCEViStoryPipeline:
    """
    Story-faithful DCEE-CausalVerse pipeline.

    Major upgrades in this version:
    - ViSTA-style salient history selection in memory
    - StoryGen / Make-A-Story style auto-regressive continuity constraints
    - TIFA-style frame-level QA / reranking for story-faithful image selection
    - stronger retry logic when a frame looks like a generic portrait instead of the target story beat
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.llm = build_llm(cfg.get('llm', {}))
        self.vlm = build_vlm(cfg.get('vlm', {}))

        llm_cfg = cfg.get('llm', {})
        self.planner = DCEPlanner(self.llm, temperature=float(llm_cfg.get('temperature', 0.35)), max_tokens=int(llm_cfg.get('max_tokens', 1600)))

        iu_cfg = cfg.get('image_understanding', {})
        self.image_understanding = ImageUnderstandingModule(iu_cfg.get('provider', 'llm_caption'), self.llm, iu_cfg.get('caption_model', 'Salesforce/blip-image-captioning-base'))

        ev_cfg = cfg.get('evaluation', {})
        self.evaluator = DCEQAEvaluator(self.llm, self.vlm, use_vlm=bool(ev_cfg.get('use_vlm', False)), save_contact_sheet=bool(ev_cfg.get('save_contact_sheet', True)), use_local_caption_scorer=bool(ev_cfg.get('use_local_caption_scorer', True)))

        b_cfg = cfg.get('butterfly', {})
        self.controller = ButterflyController(b_cfg.get('quality_suffix', QUALITY_SUFFIX), b_cfg.get('negative_prompt', NEGATIVE_PROMPT), int(b_cfg.get('num_hypotheses', 3)))

        img_cfg = cfg.get('image_generator', {})
        ad_cfg = img_cfg.get('adapter', {})
        self.image_generator = SDXLButterflyCrossAttentionGenerator(
            model_id=img_cfg.get('model_id', 'stabilityai/stable-diffusion-xl-base-1.0'),
            device=img_cfg.get('device', 'cpu'),
            width=int(img_cfg.get('width', 1024)),
            height=int(img_cfg.get('height', 1024)),
            num_inference_steps=int(img_cfg.get('num_inference_steps', 40)),
            guidance_scale=float(img_cfg.get('guidance_scale', 8.0)),
            seed=int(img_cfg.get('seed', 42)),
            adapter_ckpt=ad_cfg.get('adapter_ckpt'),
            enable_cpu_offload=bool(img_cfg.get('enable_cpu_offload', False)),
            character_tokens=int(ad_cfg.get('character_tokens', 8)),
            world_tokens=int(ad_cfg.get('world_tokens', 8)),
            emotion_tokens=int(ad_cfg.get('emotion_tokens', 8)),
            event_tokens=int(ad_cfg.get('event_tokens', 8)),
            evidence_tokens=int(ad_cfg.get('evidence_tokens', 8)),
            use_ip_adapter=bool(img_cfg.get('use_ip_adapter', True)),
            ip_adapter_repo=img_cfg.get('ip_adapter_repo', 'h94/IP-Adapter'),
            ip_adapter_subfolder=img_cfg.get('ip_adapter_subfolder', 'sdxl_models'),
            ip_adapter_weight_name=img_cfg.get('ip_adapter_weight_name', 'ip-adapter_sdxl.bin'),
            ip_adapter_scale=float(img_cfg.get('ip_adapter_scale', 0.28)),
            use_butterfly_adapter=bool(img_cfg.get('use_butterfly_adapter', False)),
        )

    def _strengthen_packet(self, packet, frame, best_candidate=None, selected_memory=None):
        packet = copy.deepcopy(packet)
        failure_reason = ''
        if best_candidate is not None:
            failure_reason = str(getattr(best_candidate, 'notes', {}).get('vlm_reason', ''))
        history_text = '' if not selected_memory else str(selected_memory.get('salient_history', ''))
        packet.positive_prompt += (
            '\n\nRETRY STORY-FAITHFUL CONTROL:'
            f"\n- Match this story sentence exactly: {getattr(frame, 'story_sentence', '')}"
            f"\n- The current event must be visually obvious: {getattr(frame, 'event', '')}"
            f"\n- The event grounding must be visible: {getattr(frame, 'event_grounding', '')}"
            f"\n- The visual evidence must appear: {getattr(frame, 'evidence_objects', [])} / {getattr(frame, 'emotion_evidence', [])}"
            f"\n- The emotion and its cause must be readable: {getattr(frame, 'emotion', '')}"
            f"\n- Use salient history for continuity but move the story forward: {history_text}"
            f"\n- Avoid a generic portrait or repeated pose."
            f"\n- If previous candidate failed, improve this: {failure_reason}"
            '\n- Use rich full color. Do not create grayscale or monochrome output.'
        )
        packet.negative_prompt += '; generic portrait only, repeated composition, missing event, missing evidence, weak emotion, emotionless face, missing visual cause, grayscale, monochrome, empty background'
        meta = packet.control_metadata or {}
        meta['story_text'] = meta.get('story_text', '') + f"; MUST MATCH STORY SENTENCE: {getattr(frame, 'story_sentence', '')}"
        meta['event_text'] = meta.get('event_text', '') + f"; MUST VISIBLY SHOW EVENT: {getattr(frame, 'event', '')}"
        meta['evidence_text'] = meta.get('evidence_text', '') + f"; MUST SHOW VISUAL EVIDENCE: {getattr(frame, 'must_show', [])}"
        meta['emotion_text'] = meta.get('emotion_text', '') + f"; emotion must be clear: {getattr(frame, 'emotion', '')}"
        meta['history_text'] = meta.get('history_text', '') + f"; salient history: {history_text}"
        packet.control_metadata = meta
        packet.adapter_weights['event_adapter'] = min(0.55, packet.adapter_weights.get('event_adapter', 0.18) + 0.14)
        packet.adapter_weights['evidence_adapter'] = min(0.58, packet.adapter_weights.get('evidence_adapter', 0.20) + 0.16)
        packet.adapter_weights['emotion_adapter'] = min(0.52, packet.adapter_weights.get('emotion_adapter', 0.28) + 0.10)
        packet.adapter_weights['world_adapter'] = min(0.38, packet.adapter_weights.get('world_adapter', 0.18) + 0.06)
        return packet

    def _save_core_plan_outputs(self, out_dir: Path, seed, abstract, full_story, dce_plan, emotion_arc, storyboard):
        _write_json(out_dir / 'seed.json', seed)
        (out_dir / 'abstract.txt').write_text(str(abstract), encoding='utf-8')
        _write_json(out_dir / 'full_story.json', full_story)
        _write_json(out_dir / 'dcee_plan.json', dce_plan)
        _write_json(out_dir / 'dce_plan.json', dce_plan)
        _write_json(out_dir / 'emotion_arc.json', emotion_arc)
        _write_json(out_dir / 'storyboard.json', storyboard)
        candidate_plans = getattr(dce_plan, 'candidate_plans', None) or getattr(dce_plan, 'dcee_candidate_plans', None)
        if candidate_plans is None:
            candidate_plans = {'note': 'No explicit candidate plan list was attached by planner.py.', 'selected_event_chain': getattr(dce_plan, 'event_chain', getattr(dce_plan, 'event_spine', []))}
        _write_json(out_dir / 'dcee_candidate_plans.json', candidate_plans)


    def _enforce_sentence_frame_lock(self, storyboard, full_story, seed):
        """
        The full story is the source of truth for visual storytelling.
        Sentence i must be visualized by frame i.
        """
        rows = (full_story or {}).get("sentences", []) if isinstance(full_story, dict) else []
        total = len(storyboard)
        try:
            setattr(seed, "_current_full_story", full_story)
            setattr(seed, "_total_frames", total)
        except Exception:
            pass

        for idx, frame in enumerate(storyboard):
            sentence = ""
            if idx < len(rows) and isinstance(rows[idx], dict):
                sentence = str(rows[idx].get("sentence", "")).strip()
            if sentence:
                try:
                    setattr(frame, "story_sentence", sentence)
                    setattr(frame, "caption", sentence)
                    setattr(frame, "sentence_frame_id", idx + 1)
                    setattr(frame, "sentence_locked", True)
                except Exception:
                    pass
            try:
                setattr(frame, "frame_id", idx + 1)
            except Exception:
                pass
        return storyboard

    def run(self, sample: Dict[str, Any], out_dir: Path) -> PipelineResult:
        out_dir = Path(out_dir)
        frames_dir = out_dir / 'frames'
        ending_dir = out_dir / 'ending_candidates'
        out_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        ending_dir.mkdir(parents=True, exist_ok=True)
        run_errors: List[Dict[str, Any]] = []

        image_summary = self.image_understanding.analyze(sample.get('image_path'), sample)
        seed = self.planner.build_seed(sample, image_summary)
        try:
            setattr(seed, 'source_image_path', sample.get('image_path', '') or '')
        except Exception:
            pass
        abstract = self.planner.generate_abstract(seed)
        dce_plan = self.planner.generate_dce_plan(seed, abstract)
        emotion_arc = self.planner.generate_emotion_arc(seed, abstract, dce_plan, int(sample.get('num_frames', 6)))
        full_story = self.planner.generate_full_story(seed, abstract, dce_plan, emotion_arc, int(sample.get('num_frames', 6)))
        storyboard = self.planner.generate_storyboard(seed, abstract, dce_plan, emotion_arc, full_story=full_story)
        storyboard = self._enforce_sentence_frame_lock(storyboard, full_story, seed)
        story_bible = build_story_bible(sample, seed, dce_plan, storyboard, full_story)
        try:
            setattr(seed, '_story_bible', story_bible)
        except Exception:
            pass
        self._save_core_plan_outputs(out_dir, seed, abstract, full_story, dce_plan, emotion_arc, storyboard)
        try:
            _write_json(out_dir / 'story_bible.json', story_bible)
        except Exception:
            pass

        memory = DCEECausalMemoryStore()
        try:
            memory.initialize(seed, dce_plan)
        except Exception as e:
            run_errors.append({'stage': 'memory.initialize', 'error': str(e), 'traceback': traceback.format_exc()})
        anchor_bank = DCEEAnchorBank().build_from_seed_and_plan(seed, dce_plan)

        selected_images: List[CandidateImage] = []
        ending_candidates: List[CandidateImage] = []
        packet_log: List[Dict[str, Any]] = []
        memory_log: List[Dict[str, Any]] = []
        candidate_manifest: List[Dict[str, Any]] = []

        img_cfg = self.cfg.get('image_generator', {})
        pipe_cfg = self.cfg.get('pipeline', {})
        num_candidates = int(img_cfg.get('num_candidates_per_frame', 4))
        num_ending_candidates = int(img_cfg.get('num_ending_candidates', 6))
        retry_enabled = bool(pipe_cfg.get('emotion_retry', True))
        emotion_threshold = float(pipe_cfg.get('emotion_visibility_threshold', 0.76))
        color_threshold = float(pipe_cfg.get('colorfulness_threshold', 0.35))
        event_threshold = float(pipe_cfg.get('event_grounding_threshold', 0.72))
        evidence_threshold = float(pipe_cfg.get('evidence_visibility_threshold', 0.70))
        story_threshold = float(pipe_cfg.get('story_alignment_threshold', 0.74))
        continuity_threshold = float(pipe_cfg.get('continuity_threshold', 0.68))

        style = sample.get('style', 'full-color cinematic storybook illustration')
        if 'color' not in style.lower():
            style = 'full-color ' + style

        previous_frame = None
        for idx, frame in enumerate(storyboard):
            frame_id = getattr(frame, 'frame_id', idx + 1)
            is_last = idx == len(storyboard) - 1
            target_dir = ending_dir if is_last else frames_dir
            candidate_count = num_ending_candidates if is_last else num_candidates

            try:
                selected_memory = memory.select(frame, dce_plan, emotion_arc, strategy=pipe_cfg.get('memory_strategy', 'adaptive_causal'))
            except Exception as e:
                selected_memory = {'error': str(e), 'stage': 'memory.select'}
                run_errors.append({'stage': f'memory.select.frame_{frame_id}', 'error': str(e), 'traceback': traceback.format_exc()})

            try:
                anchors = anchor_bank.select_for_frame(frame)
            except Exception as e:
                anchors = {'error': str(e), 'stage': 'anchor_bank.select_for_frame'}
                run_errors.append({'stage': f'anchor.select.frame_{frame_id}', 'error': str(e), 'traceback': traceback.format_exc()})

            packet = self.controller.create_packet(frame=frame, seed=seed, dce_plan=dce_plan, memory=selected_memory, style=style, previous_frame=previous_frame, anchors=anchors)
            candidates = self.image_generator.generate_from_packet(packet=packet, frame_id=frame_id, out_dir=target_dir, num_candidates=candidate_count)
            ranked = self.evaluator.rerank_ending_candidates(frame, dce_plan, candidates) if is_last else self.evaluator.rank_frame_candidates(frame, dce_plan, candidates, is_ending=False)
            if not ranked:
                raise RuntimeError(f'No ranked candidates for frame {frame_id}')
            best = ranked[0]
            retried = False

            if retry_enabled and (
                best.scores.get('emotion_visibility', 0.0) < emotion_threshold
                or best.scores.get('colorfulness', 0.0) < color_threshold
                or best.scores.get('event_grounding', 0.0) < event_threshold
                or best.scores.get('evidence_visibility', 0.0) < evidence_threshold
                or best.scores.get('story_alignment', 0.0) < story_threshold
                or best.scores.get('continuity', 0.0) < continuity_threshold
            ):
                retried = True
                strong_packet = self._strengthen_packet(packet, frame, best_candidate=best, selected_memory=selected_memory)
                retry_candidates = self.image_generator.generate_from_packet(packet=strong_packet, frame_id=frame_id, out_dir=target_dir, num_candidates=max(2, candidate_count))
                retry_ranked = self.evaluator.rerank_ending_candidates(frame, dce_plan, retry_candidates) if is_last else self.evaluator.rank_frame_candidates(frame, dce_plan, retry_candidates, is_ending=False)
                if retry_ranked and retry_ranked[0].scores.get('overall', 0.0) > best.scores.get('overall', 0.0):
                    ranked = retry_ranked
                    best = retry_ranked[0]
                    packet = strong_packet

            best.notes['retried_for_story_faithfulness'] = retried
            best.notes['salient_history_used'] = selected_memory.get('salient_history', '') if isinstance(selected_memory, dict) else ''
            if is_last:
                ending_candidates = ranked
            selected_images.append(best)
            memory.add(frame, best)
            previous_frame = frame

            packet_log.append({'frame_id': frame_id, 'visual_control_packet': _safe_asdict(packet), 'anchors': anchors, 'retried': retried})
            memory_log.append({'frame_id': frame_id, 'memory': selected_memory, 'selected_image': _safe_asdict(best)})
            candidate_manifest.append({'frame_id': frame_id, 'is_last': is_last, 'candidate_count': len(ranked), 'candidates': [_safe_asdict(x) for x in ranked]})

        _write_json(out_dir / 'visual_control_packets.json', packet_log)
        _write_json(out_dir / 'memory_log.json', memory_log)
        _write_json(out_dir / 'candidate_manifest.json', candidate_manifest)
        _write_json(out_dir / 'selected_images.json', selected_images)
        if ending_candidates:
            _write_json(out_dir / 'ending_candidates.json', ending_candidates)

        contact_sheet_path = None
        contact_sheet_error = None
        try:
            contact_sheet_path = _make_contact_sheet_force([getattr(x, 'image_path', '') for x in selected_images], out_dir / 'contact_sheet.png')
        except Exception as e:
            contact_sheet_error = str(e)
            run_errors.append({'stage': 'contact_sheet', 'error': str(e), 'traceback': traceback.format_exc()})

        questions = self.evaluator.generate_questions(dce_plan, emotion_arc, storyboard)
        try:
            evaluation = self.evaluator.evaluate_sequence(dce_plan, emotion_arc, storyboard, selected_images, questions, out_dir)
        except Exception as e:
            evaluation = {'error': 'evaluate_sequence_failed', 'error_type': type(e).__name__, 'message': str(e), 'num_selected_images': len(selected_images), 'selected_image_paths': [getattr(x, 'image_path', '') for x in selected_images]}
            run_errors.append({'stage': 'evaluate_sequence', 'error': str(e), 'traceback': traceback.format_exc()})
        if contact_sheet_path:
            evaluation['contact_sheet_path'] = str(contact_sheet_path)
        if contact_sheet_error:
            evaluation['contact_sheet_error'] = contact_sheet_error
        if run_errors:
            evaluation['run_errors'] = run_errors
        _write_json(out_dir / 'evaluation.json', evaluation)

        try:
            final_story_md = self._build_markdown(abstract, full_story, dce_plan, emotion_arc, storyboard, selected_images, ending_candidates, evaluation)
        except Exception as e:
            final_story_md = '# DCEE-CausalVerse Visual Story\n\nFinal markdown rendering failed, but core outputs were saved.\n\n' + f'- Error type: {type(e).__name__}\n- Message: {str(e)}\n'
            run_errors.append({'stage': 'final_story_md', 'error': str(e), 'traceback': traceback.format_exc()})
            evaluation['run_errors'] = run_errors
            _write_json(out_dir / 'evaluation.json', evaluation)
        if not str(final_story_md).strip():
            raise RuntimeError('Strict mode: final_story.md content is empty.')
        (out_dir / 'final_story.md').write_text(final_story_md, encoding='utf-8')

        _write_json(out_dir / 'output_manifest.json', {'contact_sheet': str(out_dir / 'contact_sheet.png'), 'final_story': str(out_dir / 'final_story.md'), 'evaluation': str(out_dir / 'evaluation.json'), 'selected_images': str(out_dir / 'selected_images.json'), 'candidate_manifest': str(out_dir / 'candidate_manifest.json'), 'storyboard': str(out_dir / 'storyboard.json'), 'full_story': str(out_dir / 'full_story.json'), 'dcee_plan': str(out_dir / 'dcee_plan.json'), 'has_contact_sheet': (out_dir / 'contact_sheet.png').exists(), 'num_selected_images': len(selected_images)})

        return PipelineResult(seed=seed, abstract=abstract, dce_plan=dce_plan, emotion_arc=emotion_arc, storyboard=storyboard, selected_images=selected_images, ending_candidates=ending_candidates, evaluation_questions=questions, evaluation=evaluation, final_story_markdown=final_story_md)

    @staticmethod
    def _build_markdown(abstract, full_story, dce_plan, emotion_arc, storyboard, images, ending_candidates, evaluation):
        lines = ['# DCEE-CausalVerse Visual Story\n', '## Abstract\n', str(abstract) + '\n', '## Full Story Draft\n', f"- Title: {(full_story or {}).get('story_title', '')}" if isinstance(full_story, dict) else '']
        if isinstance(full_story, dict):
            for i, row in enumerate((full_story or {}).get('sentences', [])):
                lines.append(f"  - [{row.get('frame_id', i+1)}] {row.get('sentence', '')}")
                if row.get('original_sentence'):
                    lines.append(f"    - Original: {row.get('original_sentence', '')}")
        lines += ['', '## Selected DCEE Plan\n', f"- Desire: {getattr(dce_plan, 'desire', '')}", f"- Conflict: {getattr(dce_plan, 'conflict', '')}", f"- Planning Structure: {getattr(dce_plan, 'planning_structure', 'DCEE: Desire-Conflict-Event-Ending Emotion')}", f"- Ending Emotion: {getattr(dce_plan, 'target_ending_emotion', '')}", '', '## Emotion Arc\n', f"- States: {' → '.join([str(x) for x in getattr(emotion_arc, 'states', [])])}", f"- Intensities: {' → '.join([str(x) for x in getattr(emotion_arc, 'intensities', [])])}", '']
        if evaluation.get('contact_sheet_path'):
            lines += ['## Contact Sheet\n', f"![Contact Sheet]({evaluation.get('contact_sheet_path')})", '']
        lines.append('## Frames\n')
        for frame, image in zip(storyboard, images):
            lines += [f"### Frame {getattr(frame, 'frame_id', '')}: {getattr(frame, 'narrative_function', '')}", f"![Frame {getattr(frame, 'frame_id', '')}]({getattr(image, 'image_path', '')})", f"- Story Sentence: {getattr(frame, 'story_sentence', '')}", f"- Event: {getattr(frame, 'event', '')}", f"- Event Grounding: {getattr(frame, 'event_grounding', '')}", f"- Emotion Arc State: {getattr(frame, 'emotion', '')}", f"- Emotion Intensity: {getattr(frame, 'emotion_intensity', '')}/5", f"- Conflict Level: {getattr(frame, 'conflict_level', '')}/5", f"- Evidence: {getattr(frame, 'emotion_evidence', [])}", f"- Must Show: {getattr(frame, 'must_show', [])}", f"- Scores: {getattr(image, 'scores', {})}", '']
        lines.append('## Ending Candidates\n')
        for cand in ending_candidates:
            lines += [f"### Candidate {getattr(cand, 'candidate_id', '')}", f"![Candidate {getattr(cand, 'candidate_id', '')}]({getattr(cand, 'image_path', '')})", f"- Scores: {getattr(cand, 'scores', {})}", '']
        lines.append('## Evaluation\n')
        for k, v in evaluation.items():
            lines.append(f'- {k}: {v}')
        return '\n'.join(lines)
