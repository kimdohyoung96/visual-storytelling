from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List
from types import SimpleNamespace
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


class TextOnlyImageSummary(SimpleNamespace):
    """Attribute-compatible image summary for runs without image_path.

    planner.py fallback code may access fields such as setting, mood, caption,
    objects, scene, style, and other image-summary attributes. Returning a
    conservative default prevents text-only runs from crashing when strict LLM
    seed JSON repair fails.
    """
    def __getattr__(self, name):
        # Important: copy.deepcopy/dataclasses.asdict probe dunder methods such as
        # __deepcopy__. Returning an empty string for those makes copy.py try to
        # call a string and raises: TypeError: 'str' object is not callable.
        if str(name).startswith('__') and str(name).endswith('__'):
            raise AttributeError(name)
        if name in {
            'objects', 'object_candidates', 'characters', 'people', 'animals',
            'key_objects', 'visible_objects', 'tags', 'colors', 'color_palette'
        }:
            return []
        return ''

    def __deepcopy__(self, memo):
        return TextOnlyImageSummary(**{k: v for k, v in self.__dict__.items()})


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

def _clear_generated_pngs(*dirs: Path):
    """Remove stale candidate images from previous runs."""
    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        for pattern in ["frame_*_cand_*.png", "frame_*.png", "*.tmp.png"]:
            for p in d.glob(pattern):
                try:
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


def _normalize_text_only_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    sample = dict(sample or {})
    sample.setdefault('image_path', '')
    sample.setdefault('protagonist_reference_paths', [])
    sample.setdefault('canonical_reference_sheet_path', '')
    sample.setdefault('signature_items', [])
    return sample


def _has_any_reference_image(sample: Dict[str, Any]) -> bool:
    sample = sample or {}
    if _image_path_exists(sample.get('image_path', '')):
        return True
    if _image_path_exists(sample.get('canonical_reference_sheet_path', '')):
        return True
    for p in sample.get('protagonist_reference_paths', []) or []:
        if _image_path_exists(p):
            return True
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
    """Grounded incremental DCEE pipeline with identity-lock and scene-contract image control.

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
            num_inference_steps=int(img_cfg.get("num_inference_steps", 36)),
            guidance_scale=float(img_cfg.get("guidance_scale", 7.5)),
            seed=int(img_cfg.get("seed", 42)),
            adapter_ckpt=ad_cfg.get("adapter_ckpt"),
            enable_cpu_offload=bool(img_cfg.get("enable_cpu_offload", False)),
            character_tokens=int(ad_cfg.get("character_tokens", 8)),
            world_tokens=int(ad_cfg.get("world_tokens", 8)),
            emotion_tokens=int(ad_cfg.get("emotion_tokens", 8)),
            event_tokens=int(ad_cfg.get("event_tokens", 8)),
            evidence_tokens=int(ad_cfg.get("evidence_tokens", 8)),
            use_refiner=bool(img_cfg.get("use_refiner", False)),
            refiner_model_id=img_cfg.get("refiner_model_id", "stabilityai/stable-diffusion-xl-refiner-1.0"),
            refiner_strength=float(img_cfg.get("refiner_strength", 0.80)),
            aesthetic_score=float(img_cfg.get("aesthetic_score", 6.0)),
            negative_aesthetic_score=float(img_cfg.get("negative_aesthetic_score", 2.5)),
            quality_model_preset=img_cfg.get("quality_model_preset", "sdxl_base"),
            use_ip_adapter=bool(img_cfg.get("use_ip_adapter", True)),
            ip_adapter_scale=float(img_cfg.get("ip_adapter_scale", 0.40)),
            canonical_reference_sheet_path=img_cfg.get("canonical_reference_sheet_path", ""),
            identity_backend_priority=img_cfg.get("identity_backend_priority", ["instantid", "photomaker", "canonical_reference_sheet", "character_lora", "ip_adapter", "text"]),
            use_instantid=bool(img_cfg.get("use_instantid", False)),
            instantid_adapter_path=img_cfg.get("instantid_adapter_path", ""),
            instantid_controlnet_path=img_cfg.get("instantid_controlnet_path", ""),
            use_photomaker=bool(img_cfg.get("use_photomaker", False)),
            photomaker_adapter_path=img_cfg.get("photomaker_adapter_path", ""),
            use_character_lora=bool(img_cfg.get("use_character_lora", False)),
            character_lora_path=img_cfg.get("character_lora_path", ""),
            character_lora_scale=float(img_cfg.get("character_lora_scale", 0.85)),
            use_subject_scene_fusion=bool(img_cfg.get("use_subject_scene_fusion", True)),
            subject_scene_fusion_scale=float(img_cfg.get("subject_scene_fusion_scale", 0.72)),
            subject_scene_fusion_first_n=int(img_cfg.get("subject_scene_fusion_first_n", 2)),
            use_previous_frame_img2img=False,
        )

    def _strengthen_packet(self, packet, frame):
        meta = getattr(packet, 'control_metadata', {}) or {}
        meta['retry_strong_mode'] = True
        meta['retry_exact_sentence'] = getattr(frame, 'story_sentence', '')
        meta['retry_required_objects'] = getattr(frame, 'must_show', [])
        meta['retry_critical_visual_nouns'] = meta.get('critical_visual_nouns', getattr(frame, 'must_show', []))
        meta['retry_location'] = getattr(frame, 'scene_location', '')
        meta['retry_emotion'] = getattr(frame, 'emotion', '')
        meta['retry_next_sentence'] = meta.get('next_frame_caption', '')
        meta['retry_single_scene'] = True
        meta['retry_full_color_background'] = True
        packet.control_metadata = meta
        try:
            packet.positive_prompt += (
                "\n\nRETRY CONTROL:"
                f"\n- Exact sentence: {getattr(frame, 'story_sentence', '')}"
                f"\n- Event: {getattr(frame, 'event', '')}"
                f"\n- Evidence: {getattr(frame, 'event_grounding', '')}"
                f"\n- Required objects: {getattr(frame, 'must_show', [])}"
                f"\n- Critical visual nouns: {meta.get('critical_visual_nouns', getattr(frame, 'must_show', []))}"
                f"\n- Next sentence (do not jump ahead to it): {meta.get('next_frame_caption', '')}"
                f"\n- Emotion: {getattr(frame, 'emotion', '')}"
                "\n- Show one protagonist only."
                "\n- Single scene only."
                "\n- Keep protagonist identity unchanged: same face, body size, fur/color pattern, hands/paws, feet, and signature items."
                "\n- DCEE may change only expression, pose, dirt/wetness, lighting, and emotional tension."
                "\n- Show full body with uncropped face, hands/paws, feet, and action."
                "\n- Preserve a readable full-color background that matches this exact story sentence."
                "\n- Show a layered environment with setting details; do not leave the background blank."
                "\n- Make every critical visual noun visible and recognizable."
                "\n- Make the frame read like one step of a progressing visual story, not an isolated portrait."
                "\n- Do not repeat the previous frame and do not jump ahead to the next frame."
            )
            packet.negative_prompt += (
                "; duplicate protagonist, extra character, extra animal, wrong action, "
                "missing object, missing critical noun, cropped body, cropped face, cropped hands, cropped paws, changed protagonist identity, gray empty background, unrelated background, blank background, static repeated scene, jumping to next scene"
            )
        except Exception:
            pass
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
                # V28: clean stale candidate images from previous runs in the same output folder.
        _clear_generated_pngs(frames_dir, ending_dir)

        run_errors: List[Dict[str, Any]] = []
        sample = _normalize_text_only_sample(sample)
        has_reference_image = _has_any_reference_image(sample)
        if has_reference_image:
            image_summary = self.image_understanding.analyze(sample.get("image_path"), sample)
        else:
            # Planner fallback expects attribute access such as image_summary.setting.
            # V41 used a dict here, which caused AttributeError after strict seed JSON repair failed.
            # V41.3 keeps text-only mode and makes the summary deepcopy-safe for dataclasses.asdict.
            image_summary = TextOnlyImageSummary(
                mode='text_only_story_generation',
                caption=sample.get('text_prompt', ''),
                summary=sample.get('text_prompt', ''),
                description=sample.get('text_prompt', ''),
                objects=list(sample.get('signature_items', []) or []),
                object_candidates=list(sample.get('signature_items', []) or []),
                key_objects=list(sample.get('signature_items', []) or []),
                visible_objects=list(sample.get('signature_items', []) or []),
                scene=sample.get('setting') or sample.get('text_prompt', '') or 'deep forest and serene lake in the forest',
                setting=sample.get('setting') or 'deep forest and serene lake in the forest',
                mood=sample.get('mood') or sample.get('target_ending_emotion', '') or 'happy',
                style=sample.get('style', ''),
                protagonist=sample.get('protagonist', ''),
                genre=sample.get('genre', ''),
                signature_items=list(sample.get('signature_items', []) or []),
                notes='No input reference image. Build story and visuals from text prompt, protagonist, and signature items only.',
            )
        seed = self.planner.build_seed(sample, image_summary)
        abstract = self.planner.generate_abstract(seed)
        dce_plan = self.planner.generate_dce_plan(seed, abstract)
        generation_policy = {
            "version": "V41.3",
            "mode": "v41_2_text_only_summary_hotfix",
            "protagonist_only": True,
            "no_secondary_characters": True,
            "training_free_consistency_removed": True,
            "english_only_text_generation": True,
            "sdxl_refiner_enabled": False,
            "previous_frame_img2img_enabled": False,
            "prompt_length_guard": True,
            "seed_validator_hotfix": True,
            "long_freeform_prompt_removed": True,
            "caption_is_image_contract": True,
            "identity_lock_is_image_contract": True,
            "scene_contract_is_image_contract": True,
            "dcee_appearance_delta_enabled": True,
            "multi_candidate_generation_restored": True,
            "cleanup_stale_candidates": True,
            "text_only_input_supported": True,
            "lightweight_multihistory_continuity": True,
            "visualized_nonvisual_event_grounding": True,
            "caption_locked_candidate_scoring": True,
            "lightweight_cross_paper_adaptations": ["ViSTA-style history selection", "StoryGen-style temporal stage cue", "StoryGPT-V-style story-to-visual bridge"],
            "uses_reference_image": has_reference_image,
            "allowed_visual_elements": [
                "protagonist",
                "caption-grounded protagonist props",
                "caption-grounded background objects",
                "weather",
                "lighting",
                "emotion cues",
                "visible cause/evidence"
            ],
            "blocked_story_entities": getattr(seed, "forbidden_ungrounded_entities", []),
            "reason": "V39 adds an optional subject-scene fusion path: first generate the protagonist according to the DCEE state, then generate the story background plate, and finally composite them. This is intended to recover protagonist consistency and readable story backgrounds when direct one-pass scene generation fails."
        }
        _write_json(out_dir / "generation_policy_V41_2_text_only_summary_hotfix.json", generation_policy)
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
        # V28: restore the useful V19/V20-style candidate selection.
        # We generate multiple candidates, but each candidate must be a single coherent scene.
        num_candidates = int(img_cfg.get("num_candidates_per_frame", 3))
        num_ending_candidates = int(img_cfg.get("num_ending_candidates", 4))
        retry_enabled = bool(pipe_cfg.get("emotion_retry", True))
        emotion_threshold = float(pipe_cfg.get("emotion_visibility_threshold", 0.74))
        color_threshold = float(pipe_cfg.get("colorfulness_threshold", 0.35))
        event_threshold = float(pipe_cfg.get("event_grounding_threshold", 0.78))
        evidence_threshold = float(pipe_cfg.get("evidence_visibility_threshold", 0.78))
        story_threshold = float(pipe_cfg.get("story_alignment_threshold", 0.82))
        background_threshold = float(pipe_cfg.get("background_presence_threshold", 0.52))
        progression_threshold = float(pipe_cfg.get("progression_threshold", 0.42))
        critical_noun_threshold = float(pipe_cfg.get("critical_noun_coverage_threshold", 0.72))
        progression_consistency_threshold = float(pipe_cfg.get("progression_consistency_threshold", 0.56))
        story_context_alignment_threshold = float(pipe_cfg.get("story_context_alignment_threshold", 0.60))

        style = sample.get("style", "full-color cinematic storybook illustration")
        if "color" not in style.lower():
            style = "full-color " + style

        planning_previous_frame = None
        for idx in range(total_frames):
            story_step = self.planner.generate_story_step(seed, abstract, dce_plan, emotion_arc, story_rows, planning_previous_frame, idx, total_frames)
            story_rows.append(story_step)
            frame = self.planner.story_step_to_frame(seed, dce_plan, emotion_arc, story_step, idx, total_frames)
            try:
                raw_crit = list(getattr(frame, 'must_show', []) or [])
                loc = str(getattr(frame, 'scene_location', '') or '')
                sent = str(getattr(frame, 'story_sentence', '') or story_step.get('sentence', '') or '')
                extra = []
                low = f"{sent} {loc}".lower()
                if 'jar' in low: extra.append('lost honey jar')
                if 'honey' in low: extra.append('honey')
                if 'root' in low: extra.append('tangled roots')
                if 'branch' in low: extra.append('fallen branches')
                if 'bush' in low: extra.append('bush')
                if 'slope' in low or 'hill' in low or 'incline' in low: extra.append('steep slope')
                if 'lake' in low or 'shore' in low or 'water' in low: extra.append('serene lake')
                if 'forest' in low or 'woods' in low or 'tree' in low: extra.append('dense forest')
                drop = {'enter','enters','search','searches','look','looks','climb','climbs','follow','follows','hear','hears','retrieve','retrieves','arrive','arrives'}
                seen = []
                for x in raw_crit + extra:
                    x = str(x).strip()
                    if not x:
                        continue
                    if x.lower() in drop:
                        continue
                    if x not in seen:
                        seen.append(x)
                setattr(frame, 'critical_visual_nouns', seen[:8])
            except Exception:
                pass
            storyboard.append(frame)
            planning_previous_frame = frame

        previous_frame = None
        for idx, frame in enumerate(storyboard):
            frame_id = idx + 1
            is_last = idx == total_frames - 1
            target_dir = ending_dir if is_last else frames_dir
            candidate_count = num_ending_candidates if is_last else num_candidates

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

            try:
                meta = getattr(packet, 'control_metadata', {}) or {}
                recent_ctx = []
                for row in story_rows[max(0, idx - 2): idx + 1]:
                    recent_ctx.append({
                        'sentence': row.get('sentence', ''),
                        'event': row.get('event', ''),
                        'emotion': row.get('emotion', ''),
                    })
                previous_story_summary = ' '.join(row.get('sentence', '') for row in story_rows[max(0, idx - 2): idx]).strip()
                next_frame = storyboard[idx + 1] if idx + 1 < len(storyboard) else None
                stage_names = ['setup', 'search', 'search', 'transition', 'discovery', 'resolution']
                meta.update({
                    'story_stage': stage_names[min(idx, len(stage_names) - 1)] if total_frames >= 6 else ('resolution' if is_last else 'progression'),
                    'recent_story_context': recent_ctx,
                    'previous_story_summary': previous_story_summary,
                    'previous_frame_caption': getattr(previous_frame, 'story_sentence', '') if previous_frame else '',
                    'previous_frame_event': getattr(previous_frame, 'event', '') if previous_frame else '',
                    'previous_frame_image_path': getattr(previous_frame, 'selected_image_path', '') if previous_frame else '',
                    'previous_frame_local_caption': getattr(previous_frame, 'selected_local_caption', '') if previous_frame else '',
                    'previous_frame_feedback': getattr(previous_frame, 'selected_feedback', {}) if previous_frame else {},
                    'next_frame_caption': getattr(next_frame, 'story_sentence', '') if next_frame else '',
                    'next_frame_event': getattr(next_frame, 'event', '') if next_frame else '',
                    'frame_transition_contract': 'Continue naturally from the previous frame and visually prepare for the next frame while keeping this exact current story step dominant.',
                    'caption_contract': getattr(frame, 'image_caption_en', '') or getattr(frame, 'story_sentence', ''),
                    'source_reference_image_path': sample.get('image_path', '') if has_reference_image else '',
                    'input_reference_image_path': sample.get('image_path', '') if has_reference_image else '',
                    'canonical_reference_sheet_path': (sample.get('canonical_reference_sheet_path', '') or img_cfg.get('canonical_reference_sheet_path', '')) if has_reference_image else '',
                    'protagonist_reference_paths': sample.get('protagonist_reference_paths', []) if has_reference_image else [],
                    'identity_backend_priority': img_cfg.get('identity_backend_priority', ['instantid', 'photomaker', 'canonical_reference_sheet', 'character_lora', 'ip_adapter', 'text']) if has_reference_image else ['text'],
                    'text_only_mode': not has_reference_image,
                    'target_objects': getattr(frame, 'must_show', []),
                    'critical_visual_nouns': getattr(frame, 'critical_visual_nouns', getattr(frame, 'must_show', [])),
                    'target_location': getattr(frame, 'scene_location', ''),
                    'identity_contract_required': True,
                    'scene_contract_required': True,
                    'dcee_appearance_delta_required': True,
                    'full_body_uncropped_required': True,
                    'background_story_alignment_required': True,
                    'background_nonempty_required': True,
                    'visual_storytelling_progression_required': True,
                    'previous_story_and_previous_frame_consistency_required': True,
                })
                packet.control_metadata = meta
            except Exception as e:
                run_errors.append({'stage': f'packet.metadata.frame_{frame_id}', 'error': str(e), 'traceback': traceback.format_exc()})

            candidates = self.image_generator.generate_from_packet(packet=packet, frame_id=frame_id, out_dir=target_dir, num_candidates=candidate_count)
            ranked = self.evaluator.rerank_ending_candidates(frame, dce_plan, candidates) if is_last else self.evaluator.rank_frame_candidates(frame, dce_plan, candidates, is_ending=False)
            if not ranked:
                raise RuntimeError(f"No ranked candidates for frame {frame_id}")
            best = ranked[0]
            retried = False
            if retry_enabled and (
                best.scores.get("story_alignment", 0.0) < story_threshold
                or best.scores.get("emotion_visibility", 0.0) < emotion_threshold
                or best.scores.get("colorfulness", 0.0) < color_threshold
                or best.scores.get("event_grounding", 0.0) < event_threshold
                or best.scores.get("evidence_visibility", 0.0) < evidence_threshold
                or best.scores.get("background_presence", 0.0) < background_threshold
                or best.scores.get("critical_noun_coverage", 0.0) < critical_noun_threshold
                or best.scores.get("missing_critical_noun_penalty", 0.0) > 0.12
                or best.scores.get("storytelling_progression", 0.0) < progression_threshold
                or best.scores.get("progression_consistency", 0.0) < progression_consistency_threshold
                or best.scores.get("story_context_alignment", 0.0) < story_context_alignment_threshold
                or best.scores.get("scene_grounding_penalty", 0.0) > 0.22
                or best.scores.get("blank_background_penalty", 0.0) > 0.10
                or best.scores.get("static_repeat_penalty", 0.0) > 0.12
                or best.scores.get("bad_extra_subject_penalty", 0.0) > 0.18
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

            setattr(frame, "selected_image_path", getattr(best, "image_path", ""))
            setattr(frame, "selected_local_caption", getattr(best, "notes", {}).get("local_caption", ""))
            setattr(frame, "selected_feedback", {
                "story_alignment": getattr(best, "scores", {}).get("story_alignment", 0.0),
                "event_grounding": getattr(best, "scores", {}).get("event_grounding", 0.0),
                "evidence_visibility": getattr(best, "scores", {}).get("evidence_visibility", 0.0),
                "background_presence": getattr(best, "scores", {}).get("background_presence", 0.0),
                "critical_noun_coverage": getattr(best, "scores", {}).get("critical_noun_coverage", 0.0),
                "missing_critical_noun_penalty": getattr(best, "scores", {}).get("missing_critical_noun_penalty", 0.0),
                "storytelling_progression": getattr(best, "scores", {}).get("storytelling_progression", 0.0),
                "progression_consistency": getattr(best, "scores", {}).get("progression_consistency", 0.0),
                "story_context_alignment": getattr(best, "scores", {}).get("story_context_alignment", 0.0),
                "blank_background_penalty": getattr(best, "scores", {}).get("blank_background_penalty", 0.0),
                "static_repeat_penalty": getattr(best, "scores", {}).get("static_repeat_penalty", 0.0),
                "bad_extra_subject_penalty": getattr(best, "scores", {}).get("bad_extra_subject_penalty", 0.0),
                "selected_prompt": getattr(best, "prompt", ""),
                "identity_backend_selected": getattr(best, "notes", {}).get("identity_backend_selected", ""),
            })
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
            "generation_policy_V40": str(out_dir / "generation_policy_V41_2_text_only_summary_hotfix.json"),
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
