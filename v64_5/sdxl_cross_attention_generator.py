
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from types import SimpleNamespace
import os
import gc
import re
import torch
import numpy as np
from PIL import Image, ImageFilter

from .schema import CandidateImage
from .frame_director import FrameVisualSpec


_MODEL_PRESETS = {
    "sdxl_base": "stabilityai/stable-diffusion-xl-base-1.0",
    "storybook_xl": "RunDiffusion/Juggernaut-XL-v9",
    "juggernaut_xl": "RunDiffusion/Juggernaut-XL-v9",
    "realvis_xl": "SG161222/RealVisXL_V5.0",
}


def _parse_gpu_ids(value: Any = None) -> List[int]:
    """Return CUDA device indices visible to this process.

    Important: these indices are *post* CUDA_VISIBLE_DEVICES remapping. If you run
    CUDA_VISIBLE_DEVICES=0,1,2,3, then cuda:0..cuda:3 are the four visible GPUs.
    """
    raw = value
    if raw is None or raw == "":
        raw = os.environ.get("DCEE_GPU_IDS", "") or os.environ.get("CUDA_VISIBLE_DEVICES", "")
    vals: List[int] = []
    if isinstance(raw, str):
        for part in raw.replace(";", ",").split(","):
            part = part.strip()
            if part.isdigit():
                vals.append(int(part))
    elif isinstance(raw, (list, tuple)):
        for part in raw:
            try:
                vals.append(int(part))
            except Exception:
                pass
    if not vals:
        try:
            if torch.cuda.is_available():
                vals = list(range(torch.cuda.device_count()))
        except Exception:
            vals = []
    out: List[int] = []
    for v in vals:
        if v not in out:
            out.append(v)
    return out


def _normalize_cuda_device(device: Any, fallback_gpu_id: int | None = None) -> str:
    dev = str(device or "cuda")
    if dev == "cuda" and fallback_gpu_id is not None:
        return f"cuda:{int(fallback_gpu_id)}"
    return dev


def _path_exists(x: Any) -> bool:
    try:
        return bool(x) and Path(str(x)).exists()
    except Exception:
        return False


def _is_cuda_oom_error(e: BaseException) -> bool:
    msg = str(e).lower()
    try:
        oom_cls = torch.cuda.OutOfMemoryError
    except Exception:
        oom_cls = RuntimeError
    return isinstance(e, oom_cls) or 'out of memory' in msg or 'cuda oom' in msg


def _is_retryable_cuda_error(e: BaseException) -> bool:
    msg = str(e).lower()
    retry_markers = [
        'out of memory',
        'cuda oom',
        'cublas_status_execution_failed',
        'cublas_status_alloc_failed',
        'cudnn_status_execution_failed',
        'cudnn error',
        'cuda error',
        'illegal memory access',
        'misaligned address',
    ]
    return _is_cuda_oom_error(e) or any(x in msg for x in retry_markers)


def _device_memory_info(device: Any) -> tuple[float, float]:
    try:
        if not torch.cuda.is_available():
            return 0.0, 0.0
        dev = torch.device(str(device or 'cuda'))
        idx = dev.index
        if idx is None:
            idx = torch.cuda.current_device()
        with torch.cuda.device(idx):
            free, total = torch.cuda.mem_get_info()
        return float(free) / (1024 ** 3), float(total) / (1024 ** 3)
    except Exception:
        return 0.0, 0.0


def _device_free_gb(device: Any) -> float:
    try:
        if not torch.cuda.is_available():
            return 0.0
        dev = torch.device(str(device or 'cuda'))
        idx = dev.index
        if idx is None:
            idx = torch.cuda.current_device()
        with torch.cuda.device(idx):
            free, total = torch.cuda.mem_get_info()
        return float(free) / (1024 ** 3)
    except Exception:
        return 0.0


def _jsonable_for_worker(x: Any):
    """Convert packet/control metadata into spawn-process-safe builtins."""
    from dataclasses import asdict, is_dataclass
    if is_dataclass(x):
        return _jsonable_for_worker(asdict(x))
    if isinstance(x, dict):
        return {str(k): _jsonable_for_worker(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable_for_worker(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if hasattr(x, "__dict__"):
        return {str(k): _jsonable_for_worker(v) for k, v in vars(x).items() if not str(k).startswith("_")}
    try:
        import json
        json.dumps(x)
        return x
    except Exception:
        return str(x)


def _packet_payload_for_worker(packet: Any) -> Dict[str, Any]:
    """Keep only the fields the SDXL generator actually needs."""
    return {
        "frame_id": _jsonable_for_worker(getattr(packet, "frame_id", 1)),
        "control_metadata": _jsonable_for_worker(getattr(packet, "control_metadata", {}) or {}),
        "reference_images": _jsonable_for_worker(getattr(packet, "reference_images", {}) or {}),
        "character_text": _jsonable_for_worker(getattr(packet, "character_text", "")),
        "event_text": _jsonable_for_worker(getattr(packet, "event_text", "")),
        "event_grounding": _jsonable_for_worker(getattr(packet, "event_grounding", "")),
        "positive_prompt": _jsonable_for_worker(getattr(packet, "positive_prompt", "")),
        "negative_prompt": _jsonable_for_worker(getattr(packet, "negative_prompt", "")),
    }


def _process_isolated_generate_worker(init_kwargs: Dict[str, Any], packet_payload: Dict[str, Any], frame_id: int, out_dir: str, num_candidates: int, candidate_ids: List[int], device: str):
    """Generate candidates in a fresh process so CUDA illegal-address failures cannot poison the parent process."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    try:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.set_device(torch.device(str(device)))
    except Exception:
        pass

    kwargs = dict(init_kwargs or {})
    kwargs["device"] = str(device)
    try:
        gid = int(str(device).split(":")[1]) if ":" in str(device) else 0
    except Exception:
        gid = 0
    kwargs["gpu_ids"] = [gid]
    kwargs["multi_gpu"] = False
    kwargs["max_parallel_generators"] = 1
    kwargs["process_isolated_multi_gpu"] = False
    kwargs["skip_busy_gpus"] = False
    kwargs["continue_on_worker_failure"] = False

    gen = SDXLButterflyCrossAttentionGenerator(**kwargs)
    packet = SimpleNamespace(**(packet_payload or {}))
    try:
        return gen._generate_from_packet_serial(packet, int(frame_id), Path(out_dir), int(num_candidates), [int(x) for x in candidate_ids])
    finally:
        try:
            gen._clear_cuda_cache()
        except Exception:
            pass
        try:
            del gen
        except Exception:
            pass
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass


def _load_reference_image(path: str, size=(1024, 1024)) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail(size)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def _compose_reference_sheet(paths: List[str], size=(1024, 1024)) -> tuple[Image.Image | None, List[str]]:
    keep = []
    for p in paths or []:
        if _path_exists(p) and str(p) not in keep:
            keep.append(str(p))
    if not keep:
        return None, []
    images = []
    for p in keep[:3]:
        try:
            images.append(Image.open(p).convert("RGB"))
        except Exception:
            pass
    if not images:
        return None, []
    cols = len(images)
    cell_w = max(1, size[0] // cols)
    cell_h = size[1]
    canvas = Image.new("RGB", size, "white")
    for i, img in enumerate(images):
        im = img.copy()
        im.thumbnail((cell_w, cell_h))
        x0 = i * cell_w + (cell_w - im.width) // 2
        y0 = (cell_h - im.height) // 2
        canvas.paste(im, (x0, y0))
    return canvas, keep


def _resolve_model_id(model_id: str, quality_model_preset: str | None = None) -> str:
    if quality_model_preset and quality_model_preset in _MODEL_PRESETS:
        return _MODEL_PRESETS[quality_model_preset]
    return model_id


def _clean(x: Any) -> str:
    s = str(x or "").replace("\n", " ").replace("\t", " ").strip()
    while "  " in s:
        s = s.replace("  ", " ")
    return s


def _clip_words(x: Any, max_words: int) -> str:
    s = _clean(x)
    if not s:
        return ""
    toks = s.split()
    return " ".join(toks[:max_words])


def _simple_sentence(text: str, max_words: int) -> str:
    s = _clip_words(text, max_words)
    s = s.strip(" .,")
    return s


def _join_unique(items: List[str], limit: int = 5, per_item_words: int = 4) -> str:
    vals = []
    for x in items or []:
        s = _clip_words(x, per_item_words)
        low = s.lower()
        if not s:
            continue
        if any(k in low for k in ["person", "human", "man", "woman", "child", "girl", "boy", "helper", "friend", "extra animal", "another protagonist"]):
            continue
        if s not in vals:
            vals.append(s)
    return ", ".join(vals[:limit])


def _lighting_hint(spec: FrameVisualSpec) -> str:
    mood = f"{spec.emotion} {spec.atmosphere} {spec.weather}".lower()
    if any(k in mood for k in ["sad", "empty", "lonely", "dark", "night", "gloom", "melancholy", "sorrow", "rain"]):
        return "moody lighting with preserved natural environment color, readable background, and bright clear light on the protagonist so the white fur and face stay visible"
    if any(k in mood for k in ["happy", "joy", "hope", "relief", "sunlight"]):
        return "clear bright lighting, vivid natural color, readable protagonist details, and colorful storybook atmosphere"
    return "balanced cinematic lighting, full natural color, readable background, and visible protagonist details"


def _critical_negative() -> str:
    return (
        "duplicate protagonist, second protagonist, extra character, extra animal, human, crowd, "
        "reflection that looks like another character, mirror clone, split screen, collage, comic panel, multiple scenes, "
        "cropped face, cropped body, cropped paws, cut off feet, hidden protagonist, tiny protagonist, "
        "wrong action, missing required object, wrong background, empty background, grayscale background, monochrome scene, washed out color, blurry, deformed, low detail, watermark, text"
    )


def _stage_name(frame_id: int, total_frames: int) -> str:
    if total_frames <= 1:
        return "single_scene"
    ratio = (frame_id - 1) / max(1, total_frames - 1)
    if ratio <= 0.15:
        return "setup"
    if ratio <= 0.45:
        return "search"
    if ratio <= 0.70:
        return "transition"
    if ratio <= 0.90:
        return "discovery"
    return "resolution"


def _visualize_action(action: str, caption: str, evidence: str) -> str:
    text = f"{action} {caption} {evidence}".lower()
    if any(k in text for k in ["hear", "hears", "listen", "sound of water"]):
        return "the protagonist pauses, turns toward a nearby stream, and reacts to the visible flowing water"
    if any(k in text for k in ["follow", "follows"]):
        return "the protagonist walks toward visible flowing water and follows the stream"
    if any(k in text for k in ["search", "look for", "looking for"]):
        return "the protagonist searches the area and looks around for the lost object"
    if any(k in text for k in ["arrive", "arrives"]):
        return "the protagonist reaches the place and notices the important object"
    if any(k in text for k in ["retrieve", "retrieves", "pick up", "recover"]):
        return "the protagonist picks up and holds the recovered object"
    if any(k in text for k in ["enter", "enters"]):
        return "the protagonist walks into the scene and begins the search"
    return _simple_sentence(action or caption, 12)


def _resolve_protagonist_anchor(spec: FrameVisualSpec) -> str:
    ident = _clean(spec.subject_identity)
    protagonist = _clean(spec.protagonist)
    text = f"{protagonist} {ident}".lower()
    parts = []
    if "white bear" in text or ("bear" in text and "white" in text):
        parts += ["one adult white bear", "creamy white fur", "rounded ears", "black nose", "black eyes", "large paws", "stocky body"]
    elif "brown bear" in text or ("bear" in text and "brown" in text):
        parts += ["one adult brown bear", "brown fur"]
    elif "polar bear" in text:
        parts += ["one adult polar bear", "creamy white fur", "rounded ears", "black nose", "black eyes", "large paws", "stocky body"]
    elif "panda" in text:
        parts += ["one adult panda", "black and white fur"]
    else:
        parts += [protagonist or "one protagonist"]
    for key in ["friendly", "gentle", "large", "fluffy"]:
        if key in text:
            parts.append(key)
    out = []
    for p in parts:
        p = _clean(p)
        if p and p not in out:
            out.append(p)
    return ", ".join(out[:7])




def _resolve_identity_lock(spec: FrameVisualSpec) -> str:
    lock = _clean(getattr(spec, "identity_lock", ""))
    if lock:
        # Keep the most important identity clauses while fitting SDXL token limits.
        clauses = [c.strip() for c in re.split(r"[;]+", lock) if c.strip()]
        return "; ".join(clauses[:4])
    return _resolve_protagonist_anchor(spec)


def _identity_short_for_prompt(spec: FrameVisualSpec) -> str:
    lock = _resolve_identity_lock(spec)
    anchor = _resolve_protagonist_anchor(spec)
    text = f"{anchor}; {lock}"
    # Never let a verbose identity block eat the whole 77-token text encoder budget.
    return _clip_words(text, 22)


def _scene_short_for_prompt(spec: FrameVisualSpec) -> str:
    contract = _clean(getattr(spec, "scene_contract", ""))
    if contract:
        return _clip_words(contract, 24)
    parts = []
    if _clean(spec.location):
        parts.append(f"background location {spec.location}")
    if spec.required_objects:
        parts.append("required visible objects " + _join_unique(spec.required_objects, limit=5, per_item_words=3))
    if _clean(spec.weather):
        parts.append(f"weather {spec.weather}")
    return _clip_words("; ".join(parts), 34)


def _appearance_delta_short(spec: FrameVisualSpec) -> str:
    delta = _clean(getattr(spec, "dcee_appearance_delta", ""))
    if not delta:
        delta = f"same identity; only expression and pose change for emotion {spec.emotion}"
    return _clip_words(delta, 24)


def _ip_scale_for_mode(base: float, mode: str, has_reference: bool) -> float:
    if not has_reference:
        return base
    # Higher on identity candidates, slightly lower on background candidates so the scene can move.
    if mode == "identity":
        return min(0.82, base + 0.08)
    if mode in {"background", "wide_scene"}:
        return max(0.55, base - 0.03)
    return base

def _spec_from_packet(packet) -> FrameVisualSpec:
    from .frame_director import build_frame_visual_spec
    m = getattr(packet, "control_metadata", {}) or {}
    d = m.get("frame_visual_spec", {}) or {}
    if d:
        return FrameVisualSpec(**d)
    # fallback minimal spec
    return FrameVisualSpec(
        frame_id=int(getattr(packet, "frame_id", 1)),
        total_frames=6,
        story_sentence=m.get("story_sentence", ""),
        protagonist=m.get("character_text", "protagonist"),
        subject_identity=m.get("character_text", "protagonist"),
        subject_reference_policy="text-only",
        primary_action=m.get("event_text", ""),
        visible_event=m.get("event_text", ""),
        visible_cause=m.get("event_grounding", ""),
        required_objects=m.get("must_show", []) or [],
        carry_over_entities=[],
        recurring_entities=[],
        forbidden_objects=["generic portrait"],
        location="",
        weather="",
        atmosphere="",
        emotion="",
        facial_expression="",
        body_pose="",
        camera="medium story shot",
        continuity="",
        negative="generic portrait",
    )


class SDXLButterflyCrossAttentionGenerator:
    """V34 identity-lock + scene-contract SDXL generator.

    Improvements over V33:
    - keeps compact identity anchor but adds a stricter identity lock for face/body/paws/signature consistency
    - separates fixed identity from DCEE-controlled appearance deltas such as expression, pose, dirt, wetness, and lighting
    - compiles a frame-level scene contract so the background follows the current story sentence instead of a generic template
    - uses identity, wide-scene, action, background, and emotion candidate modes with prompt/token auditing
    """

    def __init__(
        self,
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        device="cuda",
        width=1024,
        height=1024,
        num_inference_steps=40,
        guidance_scale=8.0,
        seed=42,
        adapter_ckpt=None,
        enable_cpu_offload=False,
        character_tokens=8,
        world_tokens=8,
        emotion_tokens=8,
        event_tokens=8,
        evidence_tokens=8,
        use_ip_adapter=True,
        ip_adapter_repo="h94/IP-Adapter",
        ip_adapter_subfolder="sdxl_models",
        ip_adapter_weight_name="ip-adapter_sdxl.bin",
        ip_adapter_scale=0.48,
        use_butterfly_adapter=False,
        quality_model_preset="sdxl_base",
        use_refiner=False,
        refiner_model_id=None,
        refiner_strength=0.80,
        aesthetic_score=6.0,
        negative_aesthetic_score=2.5,
        use_previous_frame_img2img=False,
        previous_frame_strength=0.40,
        canonical_reference_sheet_path='',
        identity_backend_priority=None,
        use_instantid=False,
        instantid_adapter_path='',
        instantid_controlnet_path='',
        use_photomaker=False,
        photomaker_adapter_path='',
        use_character_lora=False,
        character_lora_path='',
        character_lora_scale=0.85,
        use_subject_scene_fusion=True,
        subject_scene_fusion_scale=0.72,
        subject_scene_fusion_first_n=2,
        consistent_identity_seed_across_frames=True,
        multi_gpu=False,
        gpu_ids=None,
        max_parallel_generators=None,
        oom_safe_generation=True,
        oom_retry_width=768,
        oom_retry_height=768,
        enable_vae_tiling=True,
        enable_vae_slicing=True,
        enable_attention_slicing=True,
        min_free_memory_gb=14.0,
        skip_busy_gpus=True,
        continue_on_worker_failure=True,
        max_candidate_failures_per_frame=999,
        force_safe_lowres_generation=True,
        safe_generation_width=768,
        safe_generation_height=768,
        safe_num_inference_steps=34,
        disable_fusion_in_multigpu_safe=True,
        process_isolated_multi_gpu=True,
    ):
        from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

        parsed_gpu_ids = _parse_gpu_ids(gpu_ids)
        requested_device = str(device or "cuda")
        can_use_cuda = bool(requested_device.startswith("cuda") and torch.cuda.is_available())
        if parsed_gpu_ids and can_use_cuda:
            max_visible = torch.cuda.device_count()
            parsed_gpu_ids = [g for g in parsed_gpu_ids if 0 <= int(g) < max_visible]
        self.min_free_memory_gb = float(min_free_memory_gb or 0.0)
        self.skip_busy_gpus = bool(skip_busy_gpus)
        self.continue_on_worker_failure = bool(continue_on_worker_failure)
        self.max_candidate_failures_per_frame = int(max_candidate_failures_per_frame or 999)
        self.force_safe_lowres_generation = bool(force_safe_lowres_generation)
        self.safe_generation_width = int(safe_generation_width or 768)
        self.safe_generation_height = int(safe_generation_height or 768)
        self.safe_num_inference_steps = int(safe_num_inference_steps or 34)
        self.disable_fusion_in_multigpu_safe = bool(disable_fusion_in_multigpu_safe)
        self.process_isolated_multi_gpu = bool(process_isolated_multi_gpu)

        # V64.3: filter out busy CUDA devices before creating full SDXL replicas.
        # A visible A100 with another process using ~30GB can load partially but fail later
        # with OOM/CUBLAS errors. Skipping it is safer than crashing mid-frame.
        usable_gpu_ids = list(parsed_gpu_ids)
        if can_use_cuda and self.skip_busy_gpus and self.min_free_memory_gb > 0:
            keep = []
            for gid in usable_gpu_ids:
                free_gb, total_gb = _device_memory_info(f"cuda:{int(gid)}")
                if free_gb >= self.min_free_memory_gb:
                    keep.append(int(gid))
                else:
                    print(f"[GPU][SAFE] skip cuda:{gid}: free={free_gb:.2f}GB / total={total_gb:.2f}GB < min_free_memory_gb={self.min_free_memory_gb:.2f}")
            usable_gpu_ids = keep
        if not usable_gpu_ids and parsed_gpu_ids:
            # keep the first requested GPU only so the caller gets a normal CUDA error rather than index failure
            usable_gpu_ids = [int(parsed_gpu_ids[0])]
        self.gpu_ids = usable_gpu_ids
        self.multi_gpu = bool(multi_gpu and can_use_cuda and len(self.gpu_ids) > 1 and not enable_cpu_offload)
        self.max_parallel_generators = max(1, min(int(max_parallel_generators or len(self.gpu_ids) or 1), len(self.gpu_ids) or 1))
        if self.multi_gpu:
            device = _normalize_cuda_device(requested_device, self.gpu_ids[0])
        else:
            device = _normalize_cuda_device(requested_device, self.gpu_ids[0] if requested_device == "cuda" and self.gpu_ids else None)
        self._multi_gpu_workers = None
        self.oom_safe_generation = bool(oom_safe_generation)
        self.oom_retry_width = int(oom_retry_width or 768)
        self.oom_retry_height = int(oom_retry_height or 768)
        self.enable_vae_tiling = bool(enable_vae_tiling)
        self.enable_vae_slicing = bool(enable_vae_slicing)
        self.enable_attention_slicing = bool(enable_attention_slicing)

        self._worker_init_kwargs = dict(
            model_id=model_id,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            adapter_ckpt=adapter_ckpt,
            enable_cpu_offload=enable_cpu_offload,
            character_tokens=character_tokens,
            world_tokens=world_tokens,
            emotion_tokens=emotion_tokens,
            event_tokens=event_tokens,
            evidence_tokens=evidence_tokens,
            use_ip_adapter=use_ip_adapter,
            ip_adapter_repo=ip_adapter_repo,
            ip_adapter_subfolder=ip_adapter_subfolder,
            ip_adapter_weight_name=ip_adapter_weight_name,
            ip_adapter_scale=ip_adapter_scale,
            use_butterfly_adapter=use_butterfly_adapter,
            quality_model_preset=quality_model_preset,
            use_refiner=use_refiner,
            refiner_model_id=refiner_model_id,
            refiner_strength=refiner_strength,
            aesthetic_score=aesthetic_score,
            negative_aesthetic_score=negative_aesthetic_score,
            use_previous_frame_img2img=use_previous_frame_img2img,
            previous_frame_strength=previous_frame_strength,
            canonical_reference_sheet_path=canonical_reference_sheet_path,
            identity_backend_priority=identity_backend_priority,
            use_instantid=use_instantid,
            instantid_adapter_path=instantid_adapter_path,
            instantid_controlnet_path=instantid_controlnet_path,
            use_photomaker=use_photomaker,
            photomaker_adapter_path=photomaker_adapter_path,
            use_character_lora=use_character_lora,
            character_lora_path=character_lora_path,
            character_lora_scale=character_lora_scale,
            use_subject_scene_fusion=use_subject_scene_fusion,
            subject_scene_fusion_scale=subject_scene_fusion_scale,
            subject_scene_fusion_first_n=subject_scene_fusion_first_n,
            consistent_identity_seed_across_frames=consistent_identity_seed_across_frames,
            multi_gpu=False,
            gpu_ids=None,
            max_parallel_generators=1,
            oom_safe_generation=oom_safe_generation,
            oom_retry_width=oom_retry_width,
            oom_retry_height=oom_retry_height,
            enable_vae_tiling=enable_vae_tiling,
            enable_vae_slicing=enable_vae_slicing,
            enable_attention_slicing=enable_attention_slicing,
            min_free_memory_gb=min_free_memory_gb,
            skip_busy_gpus=skip_busy_gpus,
            continue_on_worker_failure=continue_on_worker_failure,
            max_candidate_failures_per_frame=max_candidate_failures_per_frame,
            force_safe_lowres_generation=force_safe_lowres_generation,
            safe_generation_width=safe_generation_width,
            safe_generation_height=safe_generation_height,
            safe_num_inference_steps=safe_num_inference_steps,
            disable_fusion_in_multigpu_safe=disable_fusion_in_multigpu_safe,
            process_isolated_multi_gpu=False,
        )

        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.seed = int(seed)
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model_id = _resolve_model_id(model_id, quality_model_preset)
        self.quality_model_preset = quality_model_preset
        self.use_refiner = False
        self.refiner_error = "disabled in V34 for stability"
        self.use_previous_frame_img2img = False
        self.story_img2img_error = "disabled in V35 for identity stability"
        self.canonical_reference_sheet_path = str(canonical_reference_sheet_path or '')
        self.identity_backend_priority = list(identity_backend_priority or ['instantid', 'photomaker', 'canonical_reference_sheet', 'character_lora', 'ip_adapter', 'text'])
        self.use_instantid = bool(use_instantid)
        self.instantid_adapter_path = str(instantid_adapter_path or '')
        self.instantid_controlnet_path = str(instantid_controlnet_path or '')
        self.use_photomaker = bool(use_photomaker)
        self.photomaker_adapter_path = str(photomaker_adapter_path or '')
        self.use_character_lora = bool(use_character_lora)
        self.character_lora_path = str(character_lora_path or '')
        self.character_lora_scale = float(character_lora_scale)
        self.use_subject_scene_fusion = bool(use_subject_scene_fusion)
        self.subject_scene_fusion_scale = float(subject_scene_fusion_scale)
        self.subject_scene_fusion_first_n = int(subject_scene_fusion_first_n)
        self.consistent_identity_seed_across_frames = bool(consistent_identity_seed_across_frames)
        self.instantid_loaded = False
        self.instantid_error = 'disabled unless InstantID dependencies and weights are provided'
        self.photomaker_loaded = False
        self.photomaker_error = 'disabled unless PhotoMaker dependencies and weights are provided'
        try:
            if self.use_instantid and _path_exists(self.instantid_adapter_path) and _path_exists(self.instantid_controlnet_path):
                self.instantid_loaded = True
                self.instantid_error = ''
        except Exception as e:
            self.instantid_loaded = False
            self.instantid_error = str(e)[:500]
        try:
            if self.use_photomaker and _path_exists(self.photomaker_adapter_path):
                self.photomaker_loaded = True
                self.photomaker_error = ''
        except Exception as e:
            self.photomaker_loaded = False
            self.photomaker_error = str(e)[:500]
        self.character_lora_loaded = False
        self.character_lora_error = ''

        try:
            if str(device).startswith('cuda') and torch.cuda.is_available():
                torch.cuda.set_device(torch.device(str(device)))
        except Exception:
            pass

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            use_safetensors=True,
        )
        try:
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config,
                use_karras_sigmas=True,
                algorithm_type="dpmsolver++",
            )
        except Exception:
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config, use_karras_sigmas=True)

        if enable_cpu_offload and device.startswith("cuda"):
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device)

        if self.enable_attention_slicing:
            try:
                self.pipe.enable_attention_slicing()
            except Exception:
                pass
        if self.enable_vae_slicing:
            try:
                self.pipe.vae.enable_slicing()
            except Exception:
                pass
        if self.enable_vae_tiling:
            try:
                self.pipe.vae.enable_tiling()
            except Exception:
                pass
        try:
            if str(device).startswith('cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        self._install_safe_vae_upcast_patch()

        self.ip_adapter_loaded = False
        self.ip_adapter_error = ""
        self.ip_adapter_scale = max(0.62, float(ip_adapter_scale))
        if use_ip_adapter:
            try:
                self.pipe.load_ip_adapter(
                    ip_adapter_repo,
                    subfolder=ip_adapter_subfolder,
                    weight_name=ip_adapter_weight_name,
                )
                # scale is set by generate_from_packet; do not overwrite dynamic per-mode scale here
                pass
                self.ip_adapter_loaded = True
            except Exception as e:
                self.ip_adapter_loaded = False
                self.ip_adapter_error = str(e)[:500]

        if self.use_character_lora and _path_exists(self.character_lora_path):
            try:
                lora_dir = str(Path(self.character_lora_path).parent)
                weight_name = Path(self.character_lora_path).name
                self.pipe.load_lora_weights(lora_dir, weight_name=weight_name)
                try:
                    self.pipe.fuse_lora(lora_scale=self.character_lora_scale)
                except Exception:
                    try:
                        self.pipe.set_adapters(['default'], adapter_weights=[self.character_lora_scale])
                    except Exception:
                        pass
                self.character_lora_loaded = True
            except Exception as e:
                self.character_lora_loaded = False
                self.character_lora_error = str(e)[:500]

    def _install_safe_vae_upcast_patch(self):
        """Patch diffusers SDXL upcast_vae to avoid the deprecation warning while preserving safe decode behavior.

        We do NOT force the VAE to float32 globally before generation, because that can destabilize
        the decode path and produce black images. Instead, we replace the pipeline's `upcast_vae()`
        method with a local implementation that only upcasts the VAE at the moment diffusers asks for it.
        The SDXL pipeline will then cast latents to the VAE parameter dtype internally before decode.
        """
        pipe = getattr(self, 'pipe', None)
        vae = getattr(pipe, 'vae', None) if pipe is not None else None
        if pipe is None or vae is None:
            return
        try:
            if getattr(pipe, '_dcee_safe_vae_patch_installed', False):
                return
        except Exception:
            pass

        def _patched_upcast_vae(pipe_self):
            try:
                pipe_self.vae.to(dtype=torch.float32)
            except Exception:
                try:
                    pipe_self.vae.to(torch.float32)
                except Exception:
                    pass
            return pipe_self.vae

        try:
            pipe.upcast_vae = _patched_upcast_vae.__get__(pipe, type(pipe))
            pipe._dcee_safe_vae_patch_installed = True
        except Exception:
            pass

    def _story_seed(self, spec: FrameVisualSpec, frame_id: int, cid: int, variant_name: str = '', strong_mode: bool = False) -> int:
        """Frame-aware seed that preserves identity stability but prevents repeated frames.

        Earlier versions re-used the same seed across frames when identity consistency was enabled.
        That made frames 2/3/5/6 collapse into near-identical images. We now keep a stable identity
        backbone while injecting frame id, sentence content, and variant focus into the seed.
        """
        sentence = _clean(getattr(spec, 'story_sentence', '') or '')
        action = _clean(getattr(spec, 'primary_action', '') or getattr(spec, 'visible_event', '') or '')
        identity = _resolve_protagonist_anchor(spec)
        sentence_hash = sum((i + 1) * ord(ch) for i, ch in enumerate(sentence[:120])) % 10007
        action_hash = sum((i + 1) * ord(ch) for i, ch in enumerate(action[:80])) % 2039
        identity_hash = sum((i + 1) * ord(ch) for i, ch in enumerate(identity[:80])) % 997
        variant_hash = sum((i + 1) * ord(ch) for i, ch in enumerate(_clean(variant_name)[:40])) % 389
        base = int(self.seed) + identity_hash * 11 + int(cid) * 97
        frame_component = sentence_hash + action_hash + variant_hash
        if not self.consistent_identity_seed_across_frames:
            frame_component += int(frame_id) * 1009
        if strong_mode:
            frame_component += 5003
        return int(base + frame_component)

    def _reference_bank_paths(self, packet) -> List[str]:
        refs = getattr(packet, "reference_images", {}) or {}
        meta = getattr(packet, "control_metadata", {}) or {}
        items = []
        if bool(meta.get('text_only_mode')):
            if bool(meta.get('allow_previous_frame_identity_reference', True)):
                # V56: keep frame-1 anchor first so all later frames lock onto the same protagonist identity.
                items += [meta.get('first_frame_image_path', ''), meta.get('identity_anchor_image_path', '')]
                items += list(meta.get('protagonist_reference_paths', []) or [])
                items += [meta.get('previous_frame_image_path', '')]
                items += list(meta.get('previous_two_selected_image_paths', []) or [])
            keep = []
            for p in items:
                if _path_exists(p) and str(p) not in keep:
                    keep.append(str(p))
            return keep
        items = []
        # Prefer a single stable protagonist identity source instead of a collage.
        items += list(meta.get('protagonist_reference_paths', []) or [])
        items += [
            meta.get('canonical_reference_sheet_path', ''),
            refs.get('canonical_reference_sheet', ''),
            self.canonical_reference_sheet_path,
            refs.get('subject', ''),
            meta.get('source_reference_image_path', ''),
            meta.get('input_reference_image_path', ''),
        ]
        keep = []
        for p in items:
            if _path_exists(p) and str(p) not in keep:
                keep.append(str(p))
        return keep

    def _resolve_identity_backend(self, meta: Dict[str, Any], reference_bank_paths: List[str]) -> str:
        if bool(meta.get('text_only_mode')):
            return 'text'
        priority = list(meta.get('identity_backend_priority', self.identity_backend_priority) or self.identity_backend_priority)
        has_ref = bool(reference_bank_paths)
        for name in priority:
            if name == 'instantid' and self.instantid_loaded and has_ref:
                return 'instantid'
            if name == 'photomaker' and self.photomaker_loaded and has_ref:
                return 'photomaker'
            if name == 'canonical_reference_sheet' and has_ref:
                return 'canonical_reference_sheet'
            if name == 'character_lora' and self.character_lora_loaded:
                return 'character_lora'
            if name == 'ip_adapter' and self.ip_adapter_loaded and has_ref:
                return 'ip_adapter'
            if name == 'text':
                return 'text'
        return 'text'

    def _subject_reference_image(self, packet):
        bank = self._reference_bank_paths(packet)
        chosen = bank[0] if bank else ''
        if chosen and _path_exists(chosen):
            return _load_reference_image(str(chosen), size=(1024, 1024)), str(chosen), [str(chosen)]
        return None, '', []

    def _token_count(self, tokenizer, text: str):
        try:
            if tokenizer is None:
                return None
            text = _clean(text)
            if not text:
                return 0
            pieces = tokenizer.tokenize(text)
            try:
                special = tokenizer.num_special_tokens_to_add(pair=False)
            except Exception:
                special = 2
            return len(pieces) + int(special)
        except Exception:
            return None

    def _token_report(self, prompt: str, prompt_2: str, negative_prompt: str) -> Dict[str, Any]:
        out = {}
        for name, tok, txt in [
            ("prompt_tokenizer_1", getattr(self.pipe, "tokenizer", None), prompt),
            ("prompt_tokenizer_2", getattr(self.pipe, "tokenizer_2", None), prompt_2),
            ("negative_tokenizer_1", getattr(self.pipe, "tokenizer", None), negative_prompt),
            ("negative_tokenizer_2", getattr(self.pipe, "tokenizer_2", None), negative_prompt),
        ]:
            if tok is None:
                continue
            max_len = int(getattr(tok, "model_max_length", 77) or 77)
            if max_len > 1000:
                max_len = 77
            n = self._token_count(tok, txt)
            out[name] = {"tokens": n, "max_length": max_len, "will_truncate": bool(n is not None and n > max_len)}
        return out

    def _fit_segments(self, tokenizer, segments: List[str], reserve: int = 4):
        max_len = int(getattr(tokenizer, "model_max_length", 77) or 77)
        if max_len > 1000:
            max_len = 77
        kept, dropped = [], []
        used = 2
        for seg in segments:
            seg = _clean(seg)
            if not seg:
                continue
            seg_tokens = self._token_count(tokenizer, seg)
            if seg_tokens is None:
                seg_tokens = len(seg.split()) + 2
            seg_tokens = max(1, seg_tokens - 2)
            join_cost = 1 if kept else 0
            if used + join_cost + seg_tokens <= (max_len - reserve):
                kept.append(seg)
                used += join_cost + seg_tokens
            else:
                dropped.append(seg)
        prompt = ". ".join(kept)
        return prompt, {
            "kept_segments": kept,
            "dropped_segments": dropped,
            "dropped_count": len(dropped),
            "max_length": max_len,
            "final_tokens": self._token_count(tokenizer, prompt),
        }

    def _fit_single_string(self, tokenizer, text: str, reserve: int = 4):
        segs = [s.strip() for s in re.split(r'[;,]+', text) if s.strip()]
        return self._fit_segments(tokenizer, segs, reserve=reserve)

    def _hard_trim_to_token_limit(self, tokenizer, text: str, reserve: int = 4):
        text = _clean(text)
        if tokenizer is None or not text:
            return text
        try:
            max_len = int(getattr(tokenizer, 'model_max_length', 77) or 77)
            if max_len > 1000:
                max_len = 77
            allowed = max(8, max_len - reserve)
            count = self._token_count(tokenizer, text)
            if count is not None and count <= allowed:
                return text
            enc = tokenizer(text, truncation=True, max_length=allowed, add_special_tokens=True)
            ids = enc.get('input_ids', [])
            if ids:
                trimmed = tokenizer.decode(ids, skip_special_tokens=True)
                trimmed = _clean(trimmed)
                if trimmed:
                    return trimmed
        except Exception:
            pass
        words = text.split()
        return ' '.join(words[: max(12, int(len(words) * 0.75)) ])

    def _text_story_terms(self, spec: FrameVisualSpec, meta: Dict[str, Any]):
        sent = _clean(getattr(spec, 'story_sentence', ''))
        low = sent.lower()
        must = []
        for item in list(getattr(spec, 'required_objects', []) or []) + list(getattr(spec, 'critical_visual_nouns', []) or []):
            item = _clean(item)
            if item and item not in must:
                must.append(item)
        env = []
        for k, v in [
            (['forest', 'woods', 'tree'], 'deep forest with trees'),
            (['bush', 'underbrush'], 'bushes and underbrush'),
            (['root'], 'tangled roots'),
            (['hill', 'slope', 'elevation'], 'small hill or slope'),
            (['path', 'trail'], 'forest path'),
            (['lake', 'water', 'shore'], 'serene lake and water edge'),
        ]:
            if any(x in low for x in k) and v not in env:
                env.append(v)
        action = _simple_sentence(meta.get('visualized_action') or spec.primary_action or spec.visible_event, 12)
        protagonist = _resolve_protagonist_anchor(spec)
        emotion = _simple_sentence(spec.emotion, 4)
        return protagonist, sent, action, emotion, must[:8], env[:6]

    def _text_only_negative_for_story(self, spec: FrameVisualSpec):
        sent = _clean(getattr(spec, 'story_sentence', '')).lower()
        negatives = [
            'fox, red fox, squirrel, deer, rabbit, wolf',
            'brown bear, grizzly bear, teddy bear, panda',
            'book, reading, chair, sofa, car, truck, bus, road vehicle',
            'city, room interior, classroom, library',
            'multiple animals, extra animal, extra human',
        ]
        if 'lake' not in sent and 'water' not in sent:
            negatives.append('boat, ocean, sea')
        return ', '.join(negatives)

    def _frame_hard_terms(self, spec: FrameVisualSpec):
        dirs = [_clean(x) for x in (getattr(spec, 'hard_visual_directives', []) or []) if _clean(x)]
        avoid = [_clean(x) for x in (getattr(spec, 'must_avoid_elements', []) or []) if _clean(x)]
        goal = _clean(getattr(spec, 'frame_goal', ''))
        return dirs[:8], avoid[:8], goal

    def _render_plan_from_meta(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        plan = meta.get('llm_render_plan', {}) or {}
        return plan if isinstance(plan, dict) else {}

    def _llm_render_plan_variant_segments(self, plan: Dict[str, Any], variant: str):
        must_show = _join_unique(plan.get('must_show', []) or [], limit=10, per_item_words=5)
        must_not = [_clean(x) for x in (plan.get('must_not_show', []) or []) if _clean(x)]
        base = [
            f"prompt core: {_clean(plan.get('prompt_core', ''))}" if _clean(plan.get('prompt_core', '')) else '',
            f"prompt detail: {_clean(plan.get('prompt_detail', ''))}" if _clean(plan.get('prompt_detail', '')) else '',
            f"persona anchor: {_clean(plan.get('persona_anchor', ''))}" if _clean(plan.get('persona_anchor', '')) else '',
            f"object state: {_clean(plan.get('object_state', ''))}" if _clean(plan.get('object_state', '')) else '',
            f"camera: {_clean(plan.get('camera', ''))}" if _clean(plan.get('camera', '')) else '',
            f"must show: {must_show}" if must_show else '',
        ]
        if variant == 'action':
            return base + [
                f"action pose: {_clean(plan.get('action_pose', ''))}" if _clean(plan.get('action_pose', '')) else '',
                'prioritize protagonist action and frame-stage correctness',
            ], must_not
        if variant == 'object':
            return base + [
                'prioritize object-state correctness and visible nouns',
                f"environment: {_clean(plan.get('environment', ''))}" if _clean(plan.get('environment', '')) else '',
            ], must_not
        if variant == 'scene':
            return base + [
                f"environment: {_clean(plan.get('environment', ''))}" if _clean(plan.get('environment', '')) else '',
                'prioritize literal background grounding and one coherent story scene',
            ], must_not
        return base, must_not

    def _text_story_variant_segments(self, spec: FrameVisualSpec, variant: str):
        phase = _clean(getattr(spec, 'frame_phase', 'progress'))
        sentence_lock = _clean(getattr(spec, 'sentence_lock', ''))
        dirs = [_clean(x) for x in (getattr(spec, 'hard_visual_directives', []) or []) if _clean(x)]
        avoid = [_clean(x) for x in (getattr(spec, 'must_avoid_elements', []) or []) if _clean(x)]
        if variant == 'action':
            return [
                sentence_lock,
                f'frame phase: {phase}' if phase else '',
                'prioritize the protagonist action and exact object interaction',
                'show the body pose and movement that explain the sentence',
                'do not simplify into a static portrait',
            ] + dirs[:4], avoid[:4]
        if variant == 'object':
            return [
                sentence_lock,
                f'frame phase: {phase}' if phase else '',
                'prioritize visible nouns and object state',
                'if the sentence mentions a jar, roots, bushes, or lake, they must be visible and recognizable',
                'make the object-stage correct for this moment',
            ] + dirs[:4], avoid[:4]
        if variant == 'scene':
            return [
                sentence_lock,
                f'frame phase: {phase}' if phase else '',
                'prioritize a readable environment that clearly matches the sentence',
                'the environment should help explain the story state',
                'show one coherent scene, not a decorative animal illustration',
            ] + dirs[:4], avoid[:4]
        if variant == 'emotion':
            return [
                sentence_lock,
                f'frame phase: {phase}' if phase else '',
                'prioritize the target emotion and facial expression',
                'make the protagonist emotion obvious through face, body pose, and posture',
                'the emotion must match the exact sentence and should not be neutral',
            ] + dirs[:4], avoid[:4]
        if variant == 'emotion_face':
            return [
                sentence_lock,
                f'frame phase: {phase}' if phase else '',
                'prioritize readable face emotion while keeping the story action visible',
                'the protagonist face must clearly show the sentence emotion through eyes, eyebrows, mouth, muzzle, and head pose',
                'use a composition where the face is readable, but still keep the key object and environment visible',
            ] + dirs[:5], avoid[:4]
        if variant == 'signature':
            return [
                sentence_lock,
                f'frame phase: {phase}' if phase else '',
                'prioritize the json signature item and its correct story-stage visibility',
                'the signature item must be obvious, recognizable, and not hidden or replaced with a vague prop',
                'keep the protagonist interacting with or searching for the signature item in a way that matches the sentence',
            ] + dirs[:5], avoid[:4]
        if variant == 'storycore':
            return [
                sentence_lock,
                f'frame phase: {phase}' if phase else '',
                'prioritize full story-core realization above all else',
                'show the protagonist, action, signature item, setting, and readable face emotion together in one coherent scene',
                'do not omit the causal clue or key object from the current story sentence',
                'the viewer should understand the whole sentence at a glance',
            ] + dirs[:6], avoid[:5]
        if variant == 'signature_emotion':
            return [
                sentence_lock,
                f'frame phase: {phase}' if phase else '',
                'prioritize both the signature item and the facial emotion together',
                'the signature item must be easy to notice and the face expression must be easy to read',
                'keep the action, object, and emotion visible in the same frame',
            ] + dirs[:6], avoid[:5]
        if variant == 'color':
            return [
                sentence_lock,
                f'frame phase: {phase}' if phase else '',
                'prioritize protagonist color fidelity and object color fidelity',
                'the white bear must clearly remain white, creamy white, and never turn brown, orange, tan, or gray',
                'keep the important object and environment colors readable and story-correct',
            ] + dirs[:4], avoid[:4]
        if variant == 'complete':
            return [
                sentence_lock,
                f'frame phase: {phase}' if phase else '',
                'prioritize complete sentence realization: protagonist, action, object, setting, and emotion must all be visible together',
                'render the full story sentence literally rather than partially',
                'do not omit the key object, key background, or facial expression',
            ] + dirs[:5], avoid[:5]
        if variant == 'identity':
            return [
                sentence_lock,
                f'frame phase: {phase}' if phase else '',
                'prioritize protagonist species and identity lock above all else',
                'the protagonist must be exactly one adult white bear and must never become a rabbit, raccoon, fox, squirrel, panda, cub, or brown bear',
                'keep the same white fur, face shape, muzzle, ears, nose, and stocky body as the recurring protagonist',
            ] + dirs[:5], avoid[:5]
        return [sentence_lock] + dirs[:5], avoid[:4]

    def _v64_emotion_face_line(self, emotion: str) -> str:
        emo = _clean(emotion).lower()
        if any(k in emo for k in ['anx', 'worried', 'uneasy']):
            return 'emotion face contract: anxious expression with alert eyes, tense brow, worried mouth, and cautious searching posture'
        if any(k in emo for k in ['frustrat', 'tense']):
            return 'emotion face contract: frustrated expression with furrowed brow, tense muzzle, strained mouth, and forceful pose'
        if any(k in emo for k in ['hope']):
            return 'emotion face contract: hopeful expression with bright attentive eyes and lifted head'
        if any(k in emo for k in ['relief', 'relieved']):
            return 'emotion face contract: relieved expression with softened eyes, relaxed brow, and gentle smile'
        if any(k in emo for k in ['joy', 'happy']):
            return 'emotion face contract: joyful expression with bright eyes and a clear happy smile'
        if any(k in emo for k in ['content', 'peaceful']):
            return 'emotion face contract: content expression with calm eyes and a satisfied smile'
        return 'emotion face contract: make the face and pose clearly express the target emotion'

    def _build_prompt_pair(self, spec: FrameVisualSpec, meta: Dict[str, Any], mode: str):
        caption = _simple_sentence(spec.story_sentence, 24)
        identity_lock = _identity_short_for_prompt(spec)
        protagonist_anchor = _resolve_protagonist_anchor(spec)
        action = _simple_sentence(meta.get("visualized_action") or spec.primary_action or spec.visible_event, 12)
        required = _join_unique(spec.required_objects, limit=6, per_item_words=4)
        critical_nouns = _join_unique(getattr(spec, 'critical_visual_nouns', []) or meta.get('critical_visual_nouns', []), limit=8, per_item_words=4)
        story_visual_checklist = _join_unique(getattr(spec, 'story_visual_checklist', []) or meta.get('story_visual_checklist', []), limit=12, per_item_words=5)
        sentence_contract_items = _join_unique(getattr(spec, 'sentence_contract_items', []) or meta.get('sentence_contract_items', []), limit=12, per_item_words=8)
        protagonist_color_lock = _clean(getattr(spec, 'protagonist_color_lock', '') or meta.get('protagonist_color_lock', ''))
        object_state_hint = _clean(getattr(spec, 'object_state_hint', '') or meta.get('object_state_hint', ''))
        literal_primary_prompt = _clean(getattr(spec, 'literal_primary_prompt', '') or meta.get('literal_primary_prompt', ''))
        action_keywords = _join_unique(getattr(spec, 'story_action_keywords', []) or meta.get('story_action_keywords', []), limit=6, per_item_words=4)
        emotion_keywords = _join_unique(getattr(spec, 'story_emotion_keywords', []) or meta.get('story_emotion_keywords', []), limit=6, per_item_words=4)
        color_keywords = _join_unique(getattr(spec, 'story_color_keywords', []) or meta.get('story_color_keywords', []), limit=4, per_item_words=4)
        location = _simple_sentence(spec.location, 10)
        weather = _simple_sentence(spec.weather, 5)
        atmosphere = _simple_sentence(spec.atmosphere, 5)
        emotion = _simple_sentence(spec.emotion, 4)
        previous_story_summary = _simple_sentence(meta.get("previous_story_summary"), 18)
        prev_caption = _simple_sentence(meta.get("previous_frame_caption"), 10)
        next_caption = _simple_sentence(meta.get("next_frame_caption") or getattr(spec, 'next_story_hint', ''), 10)
        scene_summary = _simple_sentence(getattr(spec, 'scene_summary', ''), 36)
        background_contract = _simple_sentence(getattr(spec, 'background_contract', ''), 24)
        retry_strong = bool(meta.get("retry_strong_mode"))
        text_only_mode = bool(meta.get('text_only_mode'))
        story_abstract = _simple_sentence(meta.get('story_abstract'), 44)
        generated_story_context = _simple_sentence(meta.get('generated_story_context'), 56)
        current_frame_caption = _simple_sentence(meta.get('current_frame_caption'), 32)
        current_story_sentence = _simple_sentence(meta.get('current_story_sentence'), 32)
        input_story_prompt = _simple_sentence(meta.get('input_story_prompt'), 40)
        input_protagonist = _clean(meta.get('input_protagonist', ''))
        input_signature_items = _join_unique(meta.get('input_signature_items', []) or [], limit=8, per_item_words=4)
        input_target_ending_emotion = _clean(meta.get('input_target_ending_emotion', ''))
        json_input_contract = meta.get('json_input_contract', {}) or {}
        json_protagonist_lock = _clean(meta.get('json_protagonist_lock', '') or json_input_contract.get('protagonist', ''))
        json_signature_items_lock = _join_unique(meta.get('json_signature_items_lock', []) or json_input_contract.get('signature_items', []) or [], limit=8, per_item_words=4)
        anchor_age_stage = _clean(meta.get('anchor_age_stage', 'adult'))
        anchor_identity_caption = _simple_sentence(meta.get('anchor_identity_caption'), 20)
        if bool(meta.get('v60_strict_story_frame_only')) or bool(meta.get('v61_story_abstract_caption_strict')):
            story_abstract = _simple_sentence(meta.get('story_abstract'), 120)
            generated_story_context = _simple_sentence(meta.get('generated_story_context'), 100)
            current_frame_caption = _simple_sentence(meta.get('current_frame_caption') or meta.get('caption_contract'), 90)
            current_story_sentence = _simple_sentence(meta.get('current_story_sentence') or spec.story_sentence, 90)
            dcee_plan_summary = _simple_sentence(meta.get('dcee_plan_summary'), 90)
            frame_sources = _join_unique(meta.get('frame_image_sources', []) or [], limit=8, per_item_words=5)
            json_contract = meta.get('json_input_contract', {}) or {}
            json_identity_contract = meta.get('json_identity_contract', {}) or {}
            v611_contract = meta.get('v611_frame_render_contract', {}) or {}
            v611_canonical = meta.get('v611_canonical_protagonist_contract', {}) or {}
            json_protagonist = _clean(meta.get('json_protagonist_lock', '') or json_contract.get('protagonist', '') or meta.get('global_protagonist_lock', '') or input_protagonist or 'white bear')
            json_items = _join_unique(meta.get('json_signature_items_lock', []) or json_contract.get('signature_items', []) or meta.get('global_signature_items_lock', []) or meta.get('input_signature_items', []) or [], limit=8, per_item_words=4)
            whitebear_contract = _simple_sentence(json_identity_contract.get('identity_anchor_prompt') or meta.get('anchor_identity_caption') or anchor_identity_caption or 'exactly one adult white bear with creamy white fur and the same face in every frame', 36)
            face_contract = _simple_sentence(json_identity_contract.get('face') or 'same face in every frame', 24)
            body_contract = _simple_sentence(json_identity_contract.get('body') or 'same body proportions in every frame', 24)
            color_contract = _simple_sentence(json_identity_contract.get('fur_phrase') or v611_canonical.get('fur_phrase') or 'creamy white fur', 12)
            v611_story_sentence = _simple_sentence(v611_contract.get('story_sentence') or current_story_sentence, 60)
            v611_caption = _simple_sentence(v611_contract.get('frame_caption') or current_frame_caption, 60)
            v611_action = _simple_sentence(v611_contract.get('action'), 36)
            v611_scene = _simple_sentence(v611_contract.get('scene_state'), 34)
            v611_emotion = _simple_sentence(v611_contract.get('emotion'), 18)
            v611_object_state = _simple_sentence(v611_contract.get('object_state'), 32)
            v611_must_show = _join_unique(v611_contract.get('must_show', []) or [], limit=18, per_item_words=5)
            v611_must_not_show = _join_unique(v611_contract.get('must_not_show', []) or [], limit=20, per_item_words=5)
            v61_checklist = _join_unique(
                list(getattr(spec, 'required_objects', []) or [])
                + list(getattr(spec, 'sentence_contract_items', []) or [])
                + list(getattr(spec, 'critical_visual_nouns', []) or [])
                + list(meta.get('critical_visual_nouns', []) or [])
                + list(meta.get('target_objects', []) or []),
                limit=18,
                per_item_words=5,
            )

            v64_emotion_face_line = self._v64_emotion_face_line(v611_emotion or emotion or spec.emotion)
            seg1 = [
                'V64 STRICT FRAME IMAGE GENERATION',
                'Generate the image ONLY from the current generated story sentence, current frame caption, story abstract, DCEE plan, and the input JSON contract.',
                'Every important action, object, emotion, and scene detail from the current story sentence and caption must be visible in the image.',
                'V64 STORY-CORE RULE: the frame must show the whole sentence core together: protagonist identity + current action + signature item + setting + readable facial emotion.',
                'V64 CONTINUITY RULE: keep the exact same protagonist face shape, muzzle, ears, body proportions, fur color, and silhouette across frames.',
                v64_emotion_face_line,
                f'paper-flow sources: {frame_sources}' if frame_sources else '',
                f'story abstract: {story_abstract}' if story_abstract else '',
                f'generated story context: {generated_story_context}' if generated_story_context else '',
                f'current frame caption: {current_frame_caption}' if current_frame_caption else '',
                f'current story sentence: {current_story_sentence}' if current_story_sentence else '',
                'LITERAL RENDER RULE: draw the current story sentence literally; do not replace the main object, action, or scene with a generic alternative.',
                'If the sentence mentions searching, show a readable searching pose. If it mentions the honey jar, show the honey jar visibly. If it mentions the lake, show the lake visibly.',
                f'DCEE plan summary: {dcee_plan_summary}' if dcee_plan_summary else '',
                f'JSON protagonist contract: exactly one {json_protagonist}' if json_protagonist else '',
                f'JSON signature items: {json_items}' if json_items else '',
                f'frame checklist: {v61_checklist}' if v61_checklist else '',
                f'V61.1 frame contract story sentence: {v611_story_sentence}' if v611_story_sentence else '',
                f'V61.1 frame contract caption: {v611_caption}' if v611_caption else '',
                f'V61.1 frame contract must-show: {v611_must_show}' if v611_must_show else '',
                'Do not omit any core detail from the current sentence.',
                'Do not jump to a later event or repeat an earlier frame.',
            ]
            seg2 = [
                'single coherent full-color cinematic storybook illustration',
                'medium-wide full-body composition with readable foreground action and readable background setting',
                f'protagonist identity contract: {whitebear_contract}',
                f'face contract: {face_contract}',
                f'body contract: {body_contract}',
                f'fur color contract: {color_contract}',
                'the protagonist must remain the same recurring white bear across frames: same species, same face, same age stage, same body proportions, same fur color',
                'show the caption action, caption emotion, caption object state, and caption environment together in one frame',
                f'contract action: {v611_action}' if v611_action else '',
                f'contract scene: {v611_scene}' if v611_scene else '',
                f'contract emotion: {v611_emotion}' if v611_emotion else '',
                f'contract object state: {v611_object_state}' if v611_object_state else '',
                'if the sentence says searching or pushing through underbrush, show active searching behavior and dense foliage',
                'if the sentence says spotting or finding the honey jar, the honey jar must be visible in the correct place',
                'if the sentence says savoring or enjoying honey, show the bear actively enjoying honey beside the lake',
                'maintain DCEE progression without adding unrelated animals, props, or locations',
                f'frame goal: {frame_goal}' if frame_goal else '',
            ]
            neg_raw = ', '.join([
                _critical_negative(),
                self._text_only_negative_for_story(spec),
                'random scene, fallback scene, generic animal scene, unrelated event, unrelated object, unrelated location, wrong frame event, missing sentence detail, missing caption detail, decorative portrait',
                'rabbit, raccoon, fox, squirrel, dog, cat, panda, grizzly, brown bear, black bear, teddy bear, plush toy, human, extra animal, two bears',
                'brown fur protagonist, orange fur protagonist, tan fur protagonist, gray fur protagonist, wrong protagonist color, wrong age stage, baby bear, cub bear, juvenile bear',
                'missing honey jar, hidden honey jar, unreadable action, missing dense foliage, missing lake when required, missing searching pose, missing joyful expression',
                f'contract must-not-show: {v611_must_not_show}' if v611_must_not_show else '',
                _clean(spec.negative),
            ])
            tok1 = getattr(self.pipe, 'tokenizer', None)
            tok2 = getattr(self.pipe, 'tokenizer_2', None) or tok1
            prompt, fit1 = self._fit_segments(tok1, seg1, reserve=3) if tok1 is not None else ('. '.join([s for s in seg1 if s]), {})
            prompt_2, fit2 = self._fit_segments(tok2, seg2, reserve=3) if tok2 is not None else ('. '.join([s for s in seg2 if s]), {})
            negative_prompt, negfit = self._fit_single_string(tok1 or tok2, neg_raw, reserve=3) if (tok1 or tok2) is not None else (neg_raw, {})
            prompt = self._hard_trim_to_token_limit(tok1, prompt, reserve=3)
            prompt_2 = self._hard_trim_to_token_limit(tok2, prompt_2, reserve=3)
            negative_prompt = self._hard_trim_to_token_limit(tok1 or tok2, negative_prompt, reserve=3)
            audit = self._token_report(prompt, prompt_2, negative_prompt)
            audit.update({
                'prompt_strategy': 'V61_story_abstract_caption_identity_lock',
                'story_abstract': story_abstract,
                'generated_story_context': generated_story_context,
                'current_frame_caption': current_frame_caption,
                'current_story_sentence': current_story_sentence,
                'dcee_plan_summary': dcee_plan_summary,
                'json_protagonist': json_protagonist,
                'json_signature_items': json_items,
                'visible_checklist': v61_checklist,
                'identity_contract': whitebear_contract,
                'v611_frame_contract': v611_contract,
                'v611_canonical_contract': v611_canonical,
                'prompt_fit_tokenizer_1': fit1,
                'prompt_fit_tokenizer_2': fit2,
                'negative_fit': negfit,
            })
            return prompt, prompt_2, negative_prompt, audit

        # V44: if available, use an LLM-authored frame render plan for text-only story grounding.
        if text_only_mode or mode == 'text_story':
            protagonist, sent, action2, emotion2, must_items, env_items = self._text_story_terms(spec, meta)
            hard_dirs, hard_avoid, frame_goal = self._frame_hard_terms(spec)
            variant_name = _clean(meta.get('sentence_lock_variant') or meta.get('variant_focus') or 'action')
            variant_seg, variant_avoid = self._text_story_variant_segments(spec, variant_name)
            sentence_lock = _clean(getattr(spec, 'sentence_lock', ''))
            render_plan = self._render_plan_from_meta(meta) if bool(meta.get('use_llm_frame_prompt', True)) else {}
            if render_plan:
                variant_seg, variant_avoid = self._llm_render_plan_variant_segments(render_plan, variant_name)
                protagonist = _clean(render_plan.get('persona_anchor', '')) or protagonist
                sent = _clean(render_plan.get('story_sentence', '')) or sent
                action2 = _clean(render_plan.get('action_pose', '')) or action2
                extra_must = [_clean(x) for x in (render_plan.get('must_show', []) or []) if _clean(x)]
                extra_env = [_clean(render_plan.get('environment', ''))] if _clean(render_plan.get('environment', '')) else []
                must_items = list(dict.fromkeys([x for x in (must_items + extra_must) if x]))[:10]
                env_items = list(dict.fromkeys([x for x in (env_items + extra_env) if x]))[:8]
            must_text = '; '.join(must_items)
            env_text = '; '.join(env_items)
            hard_dir_text = '; '.join(hard_dirs)
            variant_avoid_items = variant_avoid if isinstance(variant_avoid, list) else ([variant_avoid] if variant_avoid else [])
            hard_avoid_items = [_clean(x) for x in (hard_avoid + variant_avoid_items) if _clean(x)]
            hard_avoid_text = '; '.join(list(dict.fromkeys(hard_avoid_items)))
            v64_emotion_face_line = self._v64_emotion_face_line(emotion2 or emotion or spec.emotion)
            seg1 = [
                'FRAME MASTER RULE: use the input story and the current story sentence as the source of truth',
                'PAPER FLOW SOURCE: input JSON -> abstract -> DCEE plan -> generated story -> frame caption -> image',
                'V64 STORY-CORE RULE: render the whole sentence core in one frame: same protagonist + readable action + signature item + setting + visible facial emotion.',
                'V64 IDENTITY-CONSISTENCY RULE: keep the protagonist design unchanged from earlier frames, especially face shape, muzzle, ears, body proportions, and fur color.',
                v64_emotion_face_line,
                f'story abstract: {story_abstract}' if story_abstract else '',
                f'generated story context: {generated_story_context}' if generated_story_context else '',
                f'current frame caption: {current_frame_caption}' if current_frame_caption else '',
                f'current story sentence: {current_story_sentence}' if current_story_sentence else '',
                f'input story summary: {input_story_prompt}' if input_story_prompt else '',
                f'input protagonist: {input_protagonist}' if input_protagonist else '',
                f'input signature items: {input_signature_items}' if input_signature_items else '',
                f'input target ending emotion: {input_target_ending_emotion}' if input_target_ending_emotion else '',
                f'JSON SOURCE OF TRUTH protagonist: {json_protagonist_lock}' if json_protagonist_lock else '',
                f'JSON SOURCE OF TRUTH signature items: {json_signature_items_lock}' if json_signature_items_lock else '',
                'JSON HARD CONTRACT: obey the input json protagonist and signature items exactly',
                f'literal frame prompt: {literal_primary_prompt}' if literal_primary_prompt else '',
                f'exact story sentence: {sent or caption}',
                'MASTER CHARACTER CONTRACT: exactly one large adult white bear with creamy white fur, rounded ears, black eyes, black nose, large paws, and a stocky bear body',
                'V56 IDENTITY LOCK: keep the exact same protagonist identity as frame 1 across all frames',
                'V56 SHAPE LOCK: keep the same muzzle shape, ear size, head-to-body ratio, torso mass, limb thickness, and overall silhouette across all frames',
                f'AGE STAGE LOCK: the protagonist must remain {anchor_age_stage} in this frame and every frame' if anchor_age_stage else '',
                'FUR COLOR LOCK: the protagonist fur must remain white or creamy white, never brown, orange, tan, golden, gray, or red',
                f'FRAME-1 IDENTITY ANCHOR: {anchor_identity_caption}' if anchor_identity_caption else '',
                f'APPEARANCE CONTRACT: {anchor_appearance_contract}' if anchor_appearance_contract else '',
                'HARD SPECIES LOCK: never generate rabbit, raccoon, fox, squirrel, panda, brown bear, teddy bear, or any non-bear protagonist',
                'HARD JSON PROTAGONIST LOCK: the main character must be exactly the json protagonist and no substitute animal is allowed',
                'HARD JSON SIGNATURE LOCK: every json signature item must appear visibly when physically possible in the current story moment',
                'HARD AGE LOCK: do not change the protagonist to a cub, baby bear, or juvenile unless the first frame established that age',
                'HARD CHARACTER-CONSISTENCY LOCK: no redesign of the protagonist between frames',
                'no fox, no orange animal, no brown bear, no teddy bear, no human, no extra animal',
                f'object state lock: {object_state_hint}' if object_state_hint else '',
                f'current action: {action2 or action}',
                f'visible required nouns: {must_text}' if must_text else '',
                f'visible environment nouns: {env_text}' if env_text else '',
                f'emotion: {emotion2 or emotion}' if (emotion2 or emotion) else '',
                f'sentence completeness contract: {sentence_contract_items}' if sentence_contract_items else '',
                f'protagonist color lock: {protagonist_color_lock}' if protagonist_color_lock else '',
                'single coherent full-color illustrated story panel',
                f'sentence lock: {sentence_lock}' if sentence_lock else '',
                f'frame goal: {frame_goal}' if frame_goal else '',
                f'exactly one protagonist: {protagonist}',
                'show exactly one white bear only and no other animals or people',
                f'visual checklist: {story_visual_checklist}' if story_visual_checklist else '',
                f'action checklist: {action_keywords}' if action_keywords else '',
                f'emotion checklist: {emotion_keywords}' if emotion_keywords else '',
                f'color checklist: {color_keywords}' if color_keywords else '',
                f'hard visual directives: {hard_dir_text}' if hard_dir_text else '',
                f'location: {location}' if location else '',
                f'object state: {_clean(render_plan.get("object_state", ""))}' if render_plan and _clean(render_plan.get("object_state", "")) else '',
                f'camera/framing: {_clean(render_plan.get("camera", ""))}' if render_plan and _clean(render_plan.get("camera", "")) else '',
                'generate from text only; do not invent unrelated characters, props, or scenes',
                'strictly avoid mouse, bird, fish, fox, rabbit, person, child, or companion character',
                'render this exact moment from the story sentence, not a generic animal scene',
                'the white bear must be performing the current sentence action in the correct environment',
                'all important nouns from the sentence must appear visibly and recognizably when physically possible',
                'satisfy the visual checklist explicitly: action, expression, protagonist color, object, and background should all be readable',
                'complete the whole sentence in one frame: protagonist identity, action, emotion, environment, and object state must be visible together',
                'if the sentence mentions roots, show roots; if it mentions bushes, show bushes; if it mentions a lake, show the lake; if it mentions a honey jar, show the honey jar at the correct stage',
                'if a honey jar is mentioned, render a recognizable honey jar rather than a vague prop; when honey is being savored, show visible golden honey',
                'if the json signature item is honey jar, the object must read clearly as a honey jar and must not be omitted or replaced by a generic container',
                'the face expression must be easy to read: the viewer should immediately understand the protagonist emotion from the face',
                'show the emotional cause together with the emotion: anxious search, frustrated struggle, hopeful spotting, relieved discovery, joyful savoring',
                'the frame should capture the whole core of the sentence in one scene: protagonist identity, signature item, action, setting, and readable facial emotion',
            ] + [x for x in variant_seg if x]
            seg2 = [
                'frame 2 quality target: natural storybook composition, white bear integrated into forest or lake scene, readable action, no mascot look',
                'storybook illustration, readable composition, vivid natural colors',
                'show a full scene with foreground, midground, and background',
                'show exactly one white bear protagonist, full body visible, uncropped face, paws, and feet',
                'keep the same bear identity across frames: same white fur, same face shape, same ears, same nose, same body proportions',
                'keep the same protagonist appearance across frames: same silhouette, same relative size, same head shape, same muzzle, same paw size',
                f'keep the json protagonist exactly: {json_protagonist_lock}' if json_protagonist_lock else '',
                f'keep the json signature items readable: {json_signature_items_lock}' if json_signature_items_lock else '',
                'keep the same protagonist scale and silhouette across frames unless the camera distance naturally changes',
                f'keep the same age stage across frames: {anchor_age_stage}' if anchor_age_stage else '',
                'the protagonist should be inside the scene, integrated naturally with the environment',
                'make the current frame specific to the sentence and clearly different from previous and next frames',
                'make the protagonist face readable enough to identify the emotion while preserving the full-scene storytelling panel',
                'when the sentence allows it, place the honey jar where it is easy to notice in the composition',
                'show the strongest visual evidence for the current sentence only',
                'do not collapse multiple later sentences into this frame and do not repeat the previous frame composition when the action has changed',
                'make the action and facial expression obvious at a glance',
                'do not depict a decorative portrait, mascot, sticker, toy, or baby-like cub unless the sentence says so',
                'no icon, no sticker, no mascot, no poster, no collage',
                'no reading book, no vehicle, no indoor scene unless the sentence says so',
                'no fox, no brown bear, no orange bear, no tan bear, no red animal, no random substitute animal',
                'no mouse, no bird, no fish, no human, no child, no extra companion',
                f'avoid these wrong elements: {hard_avoid_text}' if hard_avoid_text else '',
                background_contract if background_contract else '',
                f'render plan environment: {_clean(render_plan.get("environment", ""))}' if render_plan and _clean(render_plan.get("environment", "")) else '',
                f'render plan detail: {_clean(render_plan.get("prompt_detail", ""))}' if render_plan and _clean(render_plan.get("prompt_detail", "")) else '',
            ]
            if bool(meta.get('hard_protagonist_lock')):
                seg1 += [
                    'EMERGENCY PROTAGONIST LOCK: the protagonist species must be a white bear only',
                    'if there is any temptation to draw a rabbit, raccoon, fox, squirrel, panda, human, or brown bear, do not; draw the white bear instead',
                    'show the same recurring adult white bear identity clearly enough to be recognized at a glance',
                    'obey the input json protagonist exactly and preserve that same appearance from frame to frame',
                    'do not change age stage; do not change white fur color; do not turn the bear into a cub or juvenile',
                    'do not redesign the face, silhouette, body size, or body type in this frame',
                ]
                seg2 += [
                    'emergency selection priority: protagonist species correctness is more important than decorative composition',
                    'do not sacrifice white-bear identity for style, cuteness, or variety',
                ]
            if bool(meta.get('is_final_frame')):
                seg1 += [
                    'this is the final frame; preserve the same protagonist identity as earlier frames',
                    'show the same adult white bear, not a cub, plush toy, or mascot-like bear',
                ]
                seg2 += [
                    'final-frame continuity is critical: keep the same fur color, face shape, ears, nose, body proportions, and painting style as previous frames',
                    'if the final sentence is about savoring honey, show the white bear beside the serene lake with the honey jar visibly present and honey enjoyment clearly readable',
                ]
            if previous_story_summary:
                seg2.append(f'story so far: {previous_story_summary}')
            if prev_caption:
                seg2.append(f'this frame should come after: {prev_caption}')
            if next_caption:
                seg2.append(f'do not jump ahead to: {next_caption}')
            if weather:
                seg2.append(f'weather: {weather}')
            if atmosphere:
                seg2.append(f'atmosphere: {atmosphere}')
            if scene_summary:
                seg2.append(f'scene summary: {scene_summary}')
            if bool(meta.get('json_grounded_story_lock')):
                seg1 += [
                    'JSON-GROUNDED STORY LOCK: the generated image must satisfy the input json contract before style or composition',
                    'JSON-GROUNDED PROTAGONIST: draw exactly one white bear only',
                    'JSON-GROUNDED SIGNATURE ITEM: if honey jar is required, it must be visible and recognizable in the correct story state',
                ]
                seg2 += [
                    'do not allow model drift away from the json protagonist or json signature items',
                    'if there is a conflict between style and json accuracy, choose json accuracy',
                ]
            if bool(meta.get('v58_strict_json_storygrounding')):
                seg1 += [
                    'V58 STORY LOCK: render the exact current story sentence literally and completely',
                    'V58 PROTAGONIST LOCK: the recurring protagonist is exactly one adult white bear with creamy white fur in every frame',
                    'V58 IDENTITY LOCK: keep the same face shape, ear shape, muzzle, paw size, body type, and silhouette from frame 1 through frame 6',
                    'V58 JSON LOCK: obey the input json protagonist and signature items exactly; do not invent a substitute animal or omit the honey jar when the sentence requires it',
                ]
                seg2 += [
                    'make this frame visually distinct by following the exact sentence action and environment, while preserving the same protagonist identity',
                    'if the current sentence is about searching, show searching; if it is about discovery, show discovery; if it is about enjoying honey, show enjoyment beside the lake',
                ]
            if bool(meta.get('v59_paper_flow_grounding')):
                seg1 += [
                    'V59 PAPER-FLOW LOCK: the image must be generated from the generated story, story abstract, and current frame caption, not from a generic animal prior',
                    'V59 ABSTRACT LOCK: the frame should remain consistent with the abstract while showing only the current frame event',
                    'V59 CAPTION LOCK: the current frame caption is the most important visual contract for this image',
                    'V59 STORYBOARD LOCK: do not skip, reorder, or merge story events across frames',
                ]
                seg2 += [
                    'make the image answer this question: what exact sentence from the generated story is this frame showing?',
                    'the frame must show the protagonist action, required objects, background, and emotion from the current caption',
                    'keep DCEE progression: desire, conflict, event/evidence, emotion change, and ending should be visible across the sequence',
                ]
            if retry_strong:
                seg1 += [
                    'strictly prioritize exact story faithfulness over decorative art',
                    'every important noun in the sentence must be visible and recognizable',
                    'do not choose a pretty but incorrect scene',
                ]
                seg2 += [
                    'increase scene detail and make the white bear clearly identifiable as a white bear',
                    'make the current action and object-state unmistakable at a glance',
                    'if the frame fails noun grounding, sacrifice style variation and enforce literal depiction',
                    'if a checklist item is missing, regenerate more literally until the action, object, color, and emotion are visible',
                ]
            tok1 = getattr(self.pipe, 'tokenizer', None)
            tok2 = getattr(self.pipe, 'tokenizer_2', None) or tok1
            prompt, fit1 = self._fit_segments(tok1, seg1, reserve=3) if tok1 is not None else ('. '.join([s for s in seg1 if s]), {})
            prompt_2, fit2 = self._fit_segments(tok2, seg2, reserve=3) if tok2 is not None else ('. '.join([s for s in seg2 if s]), {})
            neg_raw = ', '.join([_critical_negative(), self._text_only_negative_for_story(spec), _clean(render_plan.get('negative_prompt', '')) if render_plan else '', hard_avoid_text, 'fox, orange fox, red fox, squirrel, raccoon, cat, dog, brown bear, grizzly, teddy bear, plush toy, small animal, cub, baby bear, human, girl, boy, woman, man, child, companion character, two bears, extra animal, book, vehicle, missing honey jar, generic jar, wrong jar, missing honey, wrong protagonist color, brown fur, orange fur, tan fur, golden fur, gray fur, missing signature item, wrong protagonist species', _clean(spec.negative)])
            negative_prompt, negfit = self._fit_single_string(tok1 or tok2, neg_raw, reserve=3) if (tok1 or tok2) is not None else (neg_raw, {})
            prompt = self._hard_trim_to_token_limit(tok1, prompt, reserve=3)
            prompt_2 = self._hard_trim_to_token_limit(tok2, prompt_2, reserve=3)
            negative_prompt = self._hard_trim_to_token_limit(tok1 or tok2, negative_prompt, reserve=3)
            audit = self._token_report(prompt, prompt_2, negative_prompt)
            audit['prompt_fit_tokenizer_1'] = fit1
            audit['prompt_fit_tokenizer_2'] = fit2
            audit['negative_fit'] = negfit
            audit['prompt_strategy'] = 'V48_v462_restored_literal_story_grounding' if render_plan else 'V43_sentence_locked_story_grounding'
            audit['identity_lock'] = identity_lock
            audit['scene_summary'] = scene_summary
            audit['hard_visual_directives'] = hard_dirs
            audit['must_avoid_elements'] = hard_avoid
            audit['sentence_lock_variant'] = variant_name
            audit['story_visual_checklist'] = story_visual_checklist
            audit['action_keywords'] = action_keywords
            audit['emotion_keywords'] = emotion_keywords
            audit['color_keywords'] = color_keywords
            audit['sentence_contract_items'] = sentence_contract_items
            audit['protagonist_color_lock'] = protagonist_color_lock
            audit['object_state_hint'] = object_state_hint
            audit['literal_primary_prompt'] = literal_primary_prompt
            return prompt, prompt_2, negative_prompt, audit

        if mode == 'scene_first':
            variant = 'prioritize matching the exact story sentence with a readable environment and visible action'
        elif mode == 'identity':
            variant = 'prioritize stable protagonist identity while still keeping the full story scene visible'
        elif mode == 'background':
            variant = 'prioritize a complete readable background and setting details that match the sentence'
        elif mode == 'continuity':
            variant = 'prioritize continuity with the previous story step while clearly showing the current step'
        else:
            variant = 'prioritize exact story-step grounding'

        seg1 = [
            'single coherent full-color storybook illustration, one scene only',
            f'exact story sentence: {caption}',
            f'exactly one protagonist: {protagonist_anchor}',
            f'identity lock: {identity_lock}',
            'the reference image is for character identity only, not for layout or background duplication',
            f'main visible action: {action}' if action else '',
            f'required visible props: {required}' if required else '',
            f'critical visible nouns: {critical_nouns}' if critical_nouns else '',
            f'scene summary: {scene_summary}' if scene_summary else '',
            f'location: {location}' if location else '',
            f'weather: {weather}' if weather else '',
            f'atmosphere: {atmosphere}' if atmosphere else '',
            f'emotion: {emotion}' if emotion else '',
            variant,
        ]
        seg2 = [
            'cinematic storybook art, vivid natural colors, clean readable composition',
            'medium-wide or wide full-body framing, uncropped face, hands or paws, feet, and full environment visible',
            'show foreground action, midground props, and background setting clearly in one image',
            'the environment must be specific to the story sentence and not blank or plain',
            'show trees, ground, water, slope, roots, bush, or other story-specific setting details whenever the caption implies them',
            'the image must read like a visual story panel, not a sticker, icon, mascot, or isolated character portrait',
            'do not create duplicate protagonists, companions, or extra animals',
            'no collage, no split screen, no reference sheet appearance in the final image',
            background_contract if background_contract else '',
            _lighting_hint(spec),
        ]
        if previous_story_summary and mode in {'scene_first', 'continuity', 'identity'}:
            seg2.append(f'story so far: {previous_story_summary}')
        if prev_caption and mode in {'scene_first', 'continuity', 'identity'}:
            seg2.append(f'continue naturally after previous frame: {prev_caption}')
        if next_caption and mode in {'scene_first', 'continuity', 'background'}:
            seg2.append(f'do not jump ahead; next frame comes after this: {next_caption}')
        if retry_strong:
            seg1 += ['strictly prioritize story faithfulness over generic character art', 'the key noun objects and setting must be unmistakably visible']
            seg2 += ['increase environment detail and prop visibility', 'make the current action obvious at a glance', 'show a complete scene instead of a plain background character portrait']
        tok1 = getattr(self.pipe, 'tokenizer', None)
        tok2 = getattr(self.pipe, 'tokenizer_2', None) or tok1
        prompt, fit1 = self._fit_segments(tok1, seg1, reserve=3) if tok1 is not None else ('. '.join([s for s in seg1 if s]), {})
        prompt_2, fit2 = self._fit_segments(tok2, seg2, reserve=3) if tok2 is not None else ('. '.join([s for s in seg2 if s]), {})
        neg_raw = ', '.join([_critical_negative(), 'close-up portrait, mascot icon, sticker, blank background, gray background, plain background, isolated character', 'duplicate protagonist, second bear, extra animal, extra human, companion character', 'cropped body, cropped feet, cropped paws, cropped face, missing hands, missing paws, missing feet, malformed limbs', 'missing story prop, missing key object, missing setting, missing background, wrong background, unrelated background', 'reference sheet, collage, split screen, multiple panels, poster layout', _clean(spec.negative)])
        negative_prompt, negfit = self._fit_single_string(tok1 or tok2, neg_raw, reserve=3) if (tok1 or tok2) is not None else (neg_raw, {})
        audit = self._token_report(prompt, prompt_2, negative_prompt)
        audit['prompt_fit_tokenizer_1'] = fit1
        audit['prompt_fit_tokenizer_2'] = fit2
        audit['negative_fit'] = negfit
        audit['prompt_strategy'] = 'V41_scene_story_grounding'
        audit['identity_lock'] = identity_lock
        audit['scene_summary'] = scene_summary
        return prompt, prompt_2, negative_prompt, audit

    def _build_foreground_prompt(self, spec: FrameVisualSpec, meta: Dict[str, Any]):
        protagonist_anchor = _resolve_protagonist_anchor(spec)
        identity_lock = _identity_short_for_prompt(spec)
        action = _simple_sentence(meta.get("visualized_action") or spec.primary_action or spec.visible_event, 12)
        emotion = _simple_sentence(spec.emotion, 4)
        face = _simple_sentence(spec.facial_expression, 7)
        pose = _simple_sentence(spec.body_pose, 7)
        prompt = ". ".join([x for x in [
            'single full-body protagonist illustration, one subject only',
            f'exactly one protagonist: {protagonist_anchor}',
            f'identity lock: {identity_lock}',
            f'visible action: {action}' if action else '',
            f'emotion: {emotion}' if emotion else '',
            f'facial expression: {face}' if face else '',
            f'body pose: {pose}' if pose else '',
            'show the full body head-to-toe, centered, uncropped, with clear paws, feet, ears, and face',
            'simple clean studio-like background or plain soft backdrop for easy subject extraction',
            'no extra subjects, no environment clutter, no duplicate protagonist',
        ] if x])
        negative = ', '.join([
            _critical_negative(),
            'extra subject, second bear, two bears, multiple characters, busy background, forest background, lake background, duplicate protagonist, cropped body, missing feet, mascot icon, sticker'
        ])
        return prompt, negative

    def _build_background_prompt(self, spec: FrameVisualSpec, meta: Dict[str, Any]):
        caption = _simple_sentence(spec.story_sentence, 24)
        required = _join_unique(spec.required_objects, limit=5, per_item_words=3)
        critical_nouns = _join_unique(getattr(spec, 'critical_visual_nouns', []), limit=6, per_item_words=3)
        location = _simple_sentence(spec.location, 9)
        weather = _simple_sentence(spec.weather, 5)
        atmosphere = _simple_sentence(spec.atmosphere, 5)
        background_contract = _simple_sentence(getattr(spec, 'background_contract', ''), 26)
        previous_story_summary = _simple_sentence(meta.get('previous_story_summary'), 18)
        prompt = ". ".join([x for x in [
            'single coherent full-color storybook background plate',
            f'exact story sentence: {caption}',
            f'location: {location}' if location else '',
            f'background contract: {background_contract}' if background_contract else '',
            f'critical visible nouns in the environment: {critical_nouns}' if critical_nouns else '',
            f'props that must be visible in the scene: {required}' if required else '',
            f'weather: {weather}' if weather else '',
            f'atmosphere: {atmosphere}' if atmosphere else '',
            f'story so far: {previous_story_summary}' if previous_story_summary else '',
            'show a complete layered environment with foreground ground plane, midground props, and readable distant background',
            'leave clear open space in the center or lower-center for the protagonist to be placed into the scene',
            'make the environment story-specific and readable, not plain or blank',
            'no protagonist visible in the background plate',
        ] if x])
        negative = ', '.join([
            'protagonist, bear, panda, animal, human, crowd, duplicate character, blank background, gray empty background, portrait, close-up, sticker, mascot',
            _clean(spec.negative),
        ])
        return prompt, negative

    def _make_edge_mask(self, fg_image: Image.Image) -> Image.Image:
        img = fg_image.convert('RGB')
        arr = np.array(img).astype(np.int16)
        edges = np.concatenate([arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]], axis=0)
        bg = np.median(edges, axis=0)
        dist = np.sqrt(((arr - bg) ** 2).sum(axis=2))
        mask = (dist > 28).astype(np.uint8) * 255
        # also treat nearly white or nearly gray as background
        mean = arr.mean(axis=2)
        var = arr.var(axis=2)
        bg_like = ((mean > 235) | ((mean > 210) & (var < 120)))
        mask = np.where(bg_like, 0, mask).astype(np.uint8)
        pil = Image.fromarray(mask, mode='L').filter(ImageFilter.GaussianBlur(radius=2.2))
        return pil

    def _compose_subject_scene(self, fg_image: Image.Image, bg_image: Image.Image) -> Image.Image:
        bg = bg_image.convert('RGBA').resize((self.width, self.height))
        fg = fg_image.convert('RGBA')
        mask = self._make_edge_mask(fg_image)
        bbox = mask.getbbox()
        if not bbox:
            return bg_image
        fg_crop = fg.crop(bbox)
        mask_crop = mask.crop(bbox)
        target_h = int(self.height * max(0.56, min(0.82, self.subject_scene_fusion_scale)))
        scale = target_h / max(1, fg_crop.height)
        target_w = max(1, int(fg_crop.width * scale))
        fg_crop = fg_crop.resize((target_w, target_h), Image.LANCZOS)
        mask_crop = mask_crop.resize((target_w, target_h), Image.LANCZOS)
        x = (self.width - target_w) // 2
        y = self.height - target_h - int(self.height * 0.08)
        canvas = bg.copy()
        canvas.alpha_composite(fg_crop, (x, y))
        # soften hard edges by applying mask once more if supported
        out = Image.new('RGBA', (self.width, self.height), (255,255,255,0))
        out.alpha_composite(bg, (0,0))
        out.paste(fg_crop, (x,y), mask_crop)
        return out.convert('RGB')

    def _neutral_ip_adapter_image(self) -> Image.Image:
        # When IP-Adapter is loaded, diffusers may require image embeddings even for background-only generations.
        # Use a neutral blank conditioning image with adapter scale 0.0 so the background plate is not pulled toward the protagonist.
        return Image.new('RGB', (self.width, self.height), (255, 255, 255))

    def _generate_subject_scene_fusion(self, spec: FrameVisualSpec, meta: Dict[str, Any], reference_image, frame_id: int, cid: int):
        fg_prompt, fg_negative = self._build_foreground_prompt(spec, meta)
        bg_prompt, bg_negative = self._build_background_prompt(spec, meta)
        tok1 = getattr(self.pipe, 'tokenizer', None)
        tok2 = getattr(self.pipe, 'tokenizer_2', None) or tok1
        fg_prompt, _ = self._fit_single_string(tok1, fg_prompt, reserve=3) if tok1 is not None else (fg_prompt, {})
        fg_prompt_2, _ = self._fit_single_string(tok2, fg_prompt, reserve=3) if tok2 is not None else (fg_prompt, {})
        bg_prompt, _ = self._fit_single_string(tok1, bg_prompt, reserve=3) if tok1 is not None else (bg_prompt, {})
        bg_prompt_2, _ = self._fit_single_string(tok2, bg_prompt, reserve=3) if tok2 is not None else (bg_prompt, {})
        fg_negative, _ = self._fit_single_string(tok1 or tok2, fg_negative, reserve=3) if (tok1 or tok2) is not None else (fg_negative, {})
        bg_negative, _ = self._fit_single_string(tok1 or tok2, bg_negative, reserve=3) if (tok1 or tok2) is not None else (bg_negative, {})

        base_seed = self.seed + int(frame_id) * 1000 + cid
        gen_bg = torch.Generator(device=self.device).manual_seed(base_seed + 101) if self.device.startswith('cuda') else torch.Generator().manual_seed(base_seed + 101)
        bg_kwargs = dict(prompt=bg_prompt, prompt_2=bg_prompt_2, negative_prompt=bg_negative, negative_prompt_2=bg_negative, width=self.width, height=self.height, num_inference_steps=max(30, self.num_inference_steps - 2), guidance_scale=max(6.8, self.guidance_scale - 0.3), generator=gen_bg)
        # HOTFIX V39.1:
        # If IP-Adapter was loaded globally, diffusers sets encoder_hid_dim_type='ip_image_proj'.
        # Then even background-only calls require ip_adapter_image / image_embeds.
        # We pass a neutral image with adapter scale 0.0 so the background is not conditioned on the protagonist.
        if getattr(self, 'ip_adapter_loaded', False):
            try:
                self.pipe.set_ip_adapter_scale(0.0)
            except Exception:
                pass
            bg_kwargs['ip_adapter_image'] = self._neutral_ip_adapter_image()
        try:
            bg_kwargs['guidance_rescale'] = 0.15
        except Exception:
            pass
        bg_image = self._safe_pipe_call(bg_kwargs, expected_size=(self.width, self.height), frame_id=frame_id, cid=cid, mode='fusion_bg')

        gen_fg = torch.Generator(device=self.device).manual_seed(base_seed + 202) if self.device.startswith('cuda') else torch.Generator().manual_seed(base_seed + 202)
        fg_kwargs = dict(prompt=fg_prompt, prompt_2=fg_prompt_2, negative_prompt=fg_negative, negative_prompt_2=fg_negative, width=self.width, height=self.height, num_inference_steps=max(28, self.num_inference_steps - 4), guidance_scale=max(7.2, self.guidance_scale), generator=gen_fg)
        used_reference = False
        if reference_image is not None:
            try:
                self.pipe.set_ip_adapter_scale(min(0.72, max(0.60, self.ip_adapter_scale + 0.12)))
            except Exception:
                pass
        if self._apply_ip_adapter(reference_image):
            fg_kwargs['ip_adapter_image'] = reference_image
            used_reference = True
        try:
            fg_kwargs['guidance_rescale'] = 0.10
        except Exception:
            pass
        fg_image = self._safe_pipe_call(fg_kwargs, expected_size=(self.width, self.height), frame_id=frame_id, cid=cid, mode='fusion_fg')
        merged = self._compose_subject_scene(fg_image, bg_image)
        audit = {
            'prompt_strategy': 'V39_subject_scene_fusion',
            'fg_prompt': fg_prompt,
            'bg_prompt': bg_prompt,
            'fg_negative_prompt': fg_negative,
            'bg_negative_prompt': bg_negative,
            'used_reference': used_reference,
        }
        return merged, audit

    def _clear_cuda_cache(self):
        try:
            if str(self.device).startswith('cuda') and torch.cuda.is_available():
                torch.cuda.set_device(torch.device(str(self.device)))
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
            gc.collect()
        except Exception:
            pass

    def _safe_pipe_call(self, kwargs: Dict[str, Any], expected_size: tuple[int, int] | None = None, frame_id: int | None = None, cid: int | None = None, mode: str = '') -> Image.Image:
        try:
            if str(self.device).startswith('cuda') and torch.cuda.is_available():
                torch.cuda.set_device(torch.device(str(self.device)))
            return self.pipe(**kwargs).images[0]
        except Exception as e:
            if not (self.oom_safe_generation and _is_retryable_cuda_error(e)):
                raise
            self._clear_cuda_cache()
            retry_kwargs = dict(kwargs)
            original_size = (
                int(retry_kwargs.get('width', self.width)),
                int(retry_kwargs.get('height', self.height)),
            )
            retry_w = min(original_size[0], int(self.oom_retry_width or 768))
            retry_h = min(original_size[1], int(self.oom_retry_height or 768))
            retry_w = max(512, (retry_w // 8) * 8)
            retry_h = max(512, (retry_h // 8) * 8)
            retry_kwargs['width'] = retry_w
            retry_kwargs['height'] = retry_h
            retry_kwargs['num_inference_steps'] = max(24, int(retry_kwargs.get('num_inference_steps', self.num_inference_steps)) - 10)
            retry_kwargs['guidance_scale'] = min(float(retry_kwargs.get('guidance_scale', self.guidance_scale)), 8.5)
            try:
                retry_kwargs.pop('guidance_rescale', None)
            except Exception:
                pass
            print(f"[GPU][OOM-SAFE] {self.device} OOM on frame={frame_id} cand={cid} mode={mode}; retrying at {retry_w}x{retry_h}, steps={retry_kwargs['num_inference_steps']}.")
            try:
                if self.enable_vae_slicing:
                    self.pipe.vae.enable_slicing()
                if self.enable_vae_tiling:
                    self.pipe.vae.enable_tiling()
            except Exception:
                pass
            try:
                image = self.pipe(**retry_kwargs).images[0]
            except Exception:
                # The CUDA context on this worker may be in a bad state after OOM/CUBLAS.
                # Let the multi-GPU scheduler mark this worker as failed and continue/retry elsewhere.
                self._clear_cuda_cache()
                raise
            if expected_size and image.size != tuple(expected_size):
                image = image.resize(tuple(expected_size), Image.LANCZOS)
            self._clear_cuda_cache()
            return image

    def _apply_ip_adapter(self, reference_image):
        if self.ip_adapter_loaded and reference_image is not None:
            try:
                # scale is set by generate_from_packet; do not overwrite dynamic per-mode scale here
                pass
            except Exception:
                pass
            return True
        return False

    def _release_parent_cuda_pipeline_for_process_mode(self):
        """Free the parent SDXL pipeline so child processes can own CUDA contexts."""
        if getattr(self, "_parent_pipeline_released_for_process_mode", False):
            return
        if not bool(getattr(self, "process_isolated_multi_gpu", False)):
            return
        try:
            pipe = getattr(self, "pipe", None)
            if pipe is not None:
                try:
                    pipe.to("cpu")
                except Exception:
                    pass
                try:
                    del self.pipe
                except Exception:
                    self.pipe = None
        except Exception:
            pass
        self.pipe = None
        self._parent_pipeline_released_for_process_mode = True
        gc.collect()
        self._clear_cuda_cache()

    def _generate_from_packet_multi_gpu_process(self, packet, frame_id: int, out_dir: Path, num_candidates: int = 4) -> List[CandidateImage]:
        """Process-isolated multi-GPU generation.

        Thread-based multi-GPU can poison the whole Python process when one CUDA worker hits
        illegal memory access / cuDNN / cuBLAS failures. This mode runs each GPU worker in a
        fresh subprocess, so a failed CUDA context dies with that subprocess.
        """
        candidate_ids = list(range(int(num_candidates)))
        gpu_ids = list(getattr(self, "gpu_ids", []) or [])
        usable_devices = []
        for gid in gpu_ids[: max(1, int(getattr(self, "max_parallel_generators", len(gpu_ids)) or len(gpu_ids) or 1))]:
            dev = f"cuda:{int(gid)}"
            free_gb, total_gb = _device_memory_info(dev)
            if self.skip_busy_gpus and self.min_free_memory_gb > 0 and free_gb < self.min_free_memory_gb:
                print(f"[GPU][PROC-SAFE] skip {dev}: free={free_gb:.2f}GB / total={total_gb:.2f}GB < min_free_memory_gb={self.min_free_memory_gb:.2f}")
                continue
            usable_devices.append(dev)
        if not usable_devices:
            print("[GPU][PROC-SAFE] no usable CUDA worker devices; returning placeholder candidate")
            return [self._failed_placeholder_candidate(frame_id, 0, out_dir, "no usable CUDA worker devices")]

        self._release_parent_cuda_pipeline_for_process_mode()

        packet_payload = _packet_payload_for_worker(packet)
        chunks = []
        for i, dev in enumerate(usable_devices):
            ids = candidate_ids[i::len(usable_devices)]
            if ids:
                chunks.append((dev, ids))
        print("[GPU][PROC-SAFE] active process-isolated SDXL worker devices:", usable_devices)

        results: List[CandidateImage] = []
        failed_ids: List[int] = []
        failed_messages: List[str] = []
        ctx = get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(chunks), mp_context=ctx) as pool:
            futures = {}
            for dev, ids in chunks:
                kwargs = dict(self._worker_init_kwargs)
                kwargs["device"] = dev
                kwargs["gpu_ids"] = [int(dev.split(":")[1])]
                kwargs["multi_gpu"] = False
                kwargs["max_parallel_generators"] = 1
                kwargs["process_isolated_multi_gpu"] = False
                futures[pool.submit(_process_isolated_generate_worker, kwargs, packet_payload, int(frame_id), str(out_dir), int(num_candidates), ids, dev)] = (dev, ids)
            for fut in as_completed(futures):
                dev, ids = futures[fut]
                try:
                    part = fut.result()
                    results.extend(part)
                except Exception as e:
                    msg = f"{dev} subprocess failed for candidate_ids={ids}: {type(e).__name__}: {e}"
                    print(f"[GPU][PROC-SAFE] {msg}")
                    failed_messages.append(msg)
                    failed_ids.extend([int(x) for x in ids])

        if failed_ids:
            retry_devices = sorted(usable_devices, key=lambda d: _device_free_gb(d), reverse=True)
            recovered: List[CandidateImage] = []
            still_failed: List[int] = []
            for cid in failed_ids:
                ok = False
                for dev in retry_devices:
                    free_gb, _ = _device_memory_info(dev)
                    if self.skip_busy_gpus and self.min_free_memory_gb > 0 and free_gb < max(8.0, self.min_free_memory_gb * 0.45):
                        continue
                    try:
                        kwargs = dict(self._worker_init_kwargs)
                        kwargs["device"] = dev
                        kwargs["gpu_ids"] = [int(dev.split(":")[1])]
                        kwargs["multi_gpu"] = False
                        kwargs["max_parallel_generators"] = 1
                        kwargs["process_isolated_multi_gpu"] = False
                        with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
                            part = pool.submit(_process_isolated_generate_worker, kwargs, packet_payload, int(frame_id), str(out_dir), int(num_candidates), [int(cid)], dev).result()
                        recovered.extend(part)
                        ok = True
                        break
                    except Exception as e:
                        failed_messages.append(f"candidate_id={cid} retry on {dev} failed: {type(e).__name__}: {e}")
                        continue
                if not ok:
                    still_failed.append(int(cid))
            results.extend(recovered)
            if still_failed:
                if results:
                    print(f"[GPU][PROC-SAFE] continuing with {len(results)} generated candidates; dropped failed candidate_ids={still_failed[:20]}{'...' if len(still_failed) > 20 else ''}")
                else:
                    print("[GPU][PROC-SAFE] all subprocess candidates failed; returning placeholder candidate")
                    return [self._failed_placeholder_candidate(frame_id, 0, out_dir, "all process-isolated workers failed | " + " | ".join(failed_messages[-3:]))]

        dedup: Dict[int, CandidateImage] = {}
        for cand in results:
            dedup[int(getattr(cand, "candidate_id", 0))] = cand
        if not dedup:
            return [self._failed_placeholder_candidate(frame_id, 0, out_dir, "no process-isolated candidates produced")]
        return [dedup[k] for k in sorted(dedup)]

    def _ensure_multi_gpu_workers(self):
        if not self.multi_gpu:
            return [self]
        if self._multi_gpu_workers is not None:
            return self._multi_gpu_workers
        workers = [self]
        # Limit worker count. Each worker owns a full SDXL pipeline copy on one A100.
        gpu_ids = list(self.gpu_ids)[: max(1, self.max_parallel_generators)]
        for gid in gpu_ids[1:]:
            free_gb, total_gb = _device_memory_info(f"cuda:{int(gid)}")
            if self.skip_busy_gpus and self.min_free_memory_gb > 0 and free_gb < self.min_free_memory_gb:
                print(f"[GPU][SAFE] skip worker cuda:{gid}: free={free_gb:.2f}GB / total={total_gb:.2f}GB < min_free_memory_gb={self.min_free_memory_gb:.2f}")
                continue
            kwargs = dict(self._worker_init_kwargs)
            kwargs["device"] = f"cuda:{int(gid)}"
            kwargs["gpu_ids"] = [int(gid)]
            kwargs["multi_gpu"] = False
            kwargs["max_parallel_generators"] = 1
            workers.append(type(self)(**kwargs))
        self._multi_gpu_workers = workers
        print("[GPU][SAFE] active SDXL worker devices:", [getattr(w, "device", "") for w in workers])
        return workers

    def _failed_placeholder_candidate(self, frame_id: int, cid: int, out_dir: Path, reason: str, prompt: str = '') -> CandidateImage:
        """Last-resort placeholder so a CUDA worker failure never kills the whole story run."""
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"frame_{int(frame_id):03d}_cand_{int(cid):02d}_failed_placeholder.png"
        try:
            img = Image.new("RGB", (int(self.width), int(self.height)), (245, 245, 245))
            img.save(path)
        except Exception:
            # create a tiny image if the expected size somehow fails
            img = Image.new("RGB", (64, 64), (245, 245, 245))
            img.save(path)
        return CandidateImage(
            frame_id=int(frame_id),
            candidate_id=int(cid),
            image_path=str(path),
            prompt=str(prompt or ''),
            scores={
                "image_quality": 0.0,
                "colorfulness": 0.0,
                "identity_consistency": 0.0,
                "reference_subject_similarity": 0.0,
                "subject_visibility": 0.0,
                "crop_penalty": 1.0,
                "emotion_visibility": 0.0,
                "emotion_cause_visibility": 0.0,
                "event_grounding": 0.0,
                "evidence_visibility": 0.0,
                "event_emotion_causal_consistency": 0.0,
                "scene_alignment": 0.0,
                "event_alignment": 0.0,
                "story_alignment": 0.0,
                "continuity": 0.0,
                "overall": -999.0,
            },
            notes={
                "generator_version": "V64_4_cuda_failsafe_placeholder",
                "worker_device": str(getattr(self, "device", "")),
                "failed_generation_placeholder": True,
                "failure_reason": str(reason)[:1000],
            },
        )

    def _generate_from_packet_multi_gpu(self, packet, frame_id: int, out_dir: Path, num_candidates: int = 4) -> List[CandidateImage]:
        """CUDA-failsafe multi-GPU scheduler.

        V64.3 still submitted large candidate chunks to each worker. If a worker hit OOM/CUBLAS,
        the entire chunk was lost and CUDA illegal-address errors could poison the run. V64.4
        submits one candidate at a time, disables failed workers, and returns successfully
        generated candidates instead of raising when possible.
        """
        workers = self._ensure_multi_gpu_workers()
        workers = [w for w in workers if _device_free_gb(getattr(w, 'device', 'cuda')) >= max(0.1, float(getattr(self, 'min_free_memory_gb', 0.0)) * 0.35)]
        if len(workers) <= 1 or int(num_candidates) <= 1:
            try:
                return self._generate_from_packet_serial(packet, frame_id, out_dir, num_candidates=num_candidates)
            except Exception as e:
                if not self.continue_on_worker_failure:
                    raise
                print(f"[GPU][FAILSAFE] single-worker generation failed; returning placeholder candidate: {type(e).__name__}: {e}")
                return [self._failed_placeholder_candidate(frame_id, 0, out_dir, f"{type(e).__name__}: {e}")]

        pending = list(range(int(num_candidates)))
        results: List[CandidateImage] = []
        failed_ids: List[int] = []
        failed_messages: List[str] = []
        active_workers = list(workers)
        max_round_workers = max(1, min(len(active_workers), int(getattr(self, 'max_parallel_generators', len(active_workers))) or len(active_workers)))

        while pending and active_workers:
            batch = []
            for _ in range(min(max_round_workers, len(active_workers), len(pending))):
                batch.append(pending.pop(0))
            with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                futures = {}
                for i, cid in enumerate(batch):
                    worker = active_workers[i % len(active_workers)]
                    futures[pool.submit(worker._generate_from_packet_serial, packet, frame_id, out_dir, int(num_candidates), [int(cid)])] = (worker, int(cid))
                for fut in as_completed(futures):
                    worker, cid = futures[fut]
                    device = str(getattr(worker, 'device', 'cuda'))
                    try:
                        part = fut.result()
                        results.extend(part)
                    except Exception as e:
                        msg = f"{device} candidate_id={cid} failed: {type(e).__name__}: {e}"
                        failed_messages.append(msg)
                        failed_ids.append(int(cid))
                        print(f"[GPU][FAILSAFE] {msg}")
                        try:
                            worker._clear_cuda_cache()
                        except Exception:
                            pass
                        # CUDA illegal memory / CUBLAS / cuDNN failures usually mean the context is unsafe.
                        if _is_retryable_cuda_error(e) or 'cudnn' in str(e).lower():
                            active_workers = [w for w in active_workers if str(getattr(w, 'device', 'cuda')) != device]
                            print(f"[GPU][FAILSAFE] disabling unhealthy worker {device}; remaining={[str(getattr(w, 'device', '')) for w in active_workers]}")
                        # Retry the failed candidate at the end if there are still healthy workers.
                        if active_workers:
                            pending.append(int(cid))
                    finally:
                        try:
                            worker._clear_cuda_cache()
                        except Exception:
                            pass

            # If every worker failed in this round, do not infinite-loop.
            if not active_workers:
                break

            # If we already have enough candidates to rank, do not keep risking CUDA contexts for all failed ids.
            min_needed = max(4, min(8, int(num_candidates) // 3))
            if len(results) >= min_needed and len(failed_ids) >= self.max_candidate_failures_per_frame:
                break

        if not results:
            # Last resort: do not crash the full storytelling pipeline.
            print("[GPU][FAILSAFE] all CUDA workers failed; creating placeholder candidates so pipeline can continue.")
            keep = pending[:max(1, min(4, int(num_candidates)))] or [0]
            return [self._failed_placeholder_candidate(frame_id, cid, out_dir, "all CUDA workers failed | " + " | ".join(failed_messages[-3:])) for cid in keep]

        if failed_ids:
            print("[GPU][FAILSAFE] continuing with "
                  + str(len(results)) + " generated candidates; failed candidate_ids="
                  + str(sorted(set(failed_ids))[:30]) + ("..." if len(set(failed_ids)) > 30 else ""))

        dedup: Dict[int, CandidateImage] = {}
        for cand in results:
            dedup[int(getattr(cand, 'candidate_id', 0))] = cand
        return [dedup[k] for k in sorted(dedup)]

    @torch.no_grad()
    def generate_from_packet(self, packet, frame_id: int, out_dir: Path, num_candidates: int = 4) -> List[CandidateImage]:
        if self.multi_gpu and int(num_candidates) > 1:
            if bool(getattr(self, "process_isolated_multi_gpu", False)):
                return self._generate_from_packet_multi_gpu_process(packet, frame_id, out_dir, num_candidates=num_candidates)
            return self._generate_from_packet_multi_gpu(packet, frame_id, out_dir, num_candidates=num_candidates)
        try:
            return self._generate_from_packet_serial(packet, frame_id, out_dir, num_candidates=num_candidates)
        except Exception as e:
            if not self.continue_on_worker_failure:
                raise
            print(f"[GPU][FAILSAFE] single-worker generation failed; returning placeholder candidate: {type(e).__name__}: {e}")
            return [self._failed_placeholder_candidate(frame_id, 0, out_dir, f"{type(e).__name__}: {e}")]

    @torch.no_grad()
    def _generate_from_packet_serial(self, packet, frame_id: int, out_dir: Path, num_candidates: int = 4, candidate_ids: List[int] | None = None) -> List[CandidateImage]:
        out_dir.mkdir(parents=True, exist_ok=True)
        spec = _spec_from_packet(packet)
        meta = getattr(packet, "control_metadata", {}) or {}
        reference_image, reference_path, reference_bank_paths = self._subject_reference_image(packet)
        active_identity_backend = self._resolve_identity_backend(meta, reference_bank_paths)
        text_only_mode = bool(meta.get('text_only_mode'))
        strict_story_frame_only = bool(meta.get('v60_strict_story_frame_only')) or bool(meta.get('v61_story_abstract_caption_strict')) or bool(meta.get('v611_frame_contract_strict'))
        text_has_anchor_ref = bool(text_only_mode and reference_image is not None and str(meta.get('first_frame_image_path', '') or meta.get('identity_anchor_image_path', '')).strip())
        if strict_story_frame_only and text_has_anchor_ref:
            modes = ["text_story", "fusion", "text_story", "continuity", "text_story"]
        elif strict_story_frame_only:
            modes = ["text_story"]
        elif text_only_mode and text_has_anchor_ref:
            modes = ["fusion", "text_story", "fusion", "continuity", "text_story", "scene_first"]
        elif text_only_mode:
            modes = ["text_story", "text_story", "text_story", "text_story", "text_story", "text_story"]
        else:
            modes = ["fusion", "fusion", "scene_first", "continuity"]
        text_variants = ["storycore", "complete", "signature", "signature_emotion", "action", "scene", "emotion_face", "complete", "emotion", "object", "color"]

        results: List[CandidateImage] = []
        candidate_ids = list(range(int(num_candidates))) if candidate_ids is None else [int(x) for x in candidate_ids]
        for cid in candidate_ids:
            if strict_story_frame_only:
                mode = "text_story"
            else:
                mode = modes[cid % len(modes)] if (self.use_subject_scene_fusion and not text_only_mode) else (["text_story", "text_story", "text_story", "text_story"][cid % 4] if text_only_mode else ["scene_first", "identity", "background", "continuity"][cid % 4])
            meta2 = dict(meta)
            current_variant = ''
            if text_only_mode or strict_story_frame_only:
                current_variant = text_variants[cid % len(text_variants)]
                meta2['sentence_lock_variant'] = current_variant
                meta2['variant_focus'] = current_variant
                if strict_story_frame_only:
                    meta2['force_text_story_only'] = True
                    meta2['sentence_lock_only'] = True
                    meta2['retry_strong_mode'] = True
                    meta2['verify_words_strict'] = True
            meta2["visualized_action"] = _visualize_action(spec.visible_event or spec.primary_action, spec.story_sentence, spec.visible_cause)
            strong_mode = bool(meta2.get('retry_strong_mode')) or bool(meta2.get('verify_words_strict'))
            sd = self._story_seed(spec, frame_id=int(frame_id), cid=int(cid), variant_name=current_variant or mode, strong_mode=strong_mode)
            used_reference = False
            effective_ip_adapter_scale = self.ip_adapter_scale
            fusion_audit = {}
            use_anchor_fusion = bool((not bool(getattr(self, "disable_fusion_in_multigpu_safe", False))) and self.use_subject_scene_fusion and ((not text_only_mode) or text_has_anchor_ref) and ((not strict_story_frame_only) or text_has_anchor_ref))
            if use_anchor_fusion and cid < max(1, self.subject_scene_fusion_first_n + (2 if text_has_anchor_ref else 0)):
                image, fusion_audit = self._generate_subject_scene_fusion(spec, meta2, reference_image, frame_id, cid)
                prompt = fusion_audit.get('fg_prompt', '')
                prompt_2 = fusion_audit.get('bg_prompt', '')
                negative_prompt = fusion_audit.get('fg_negative_prompt', '')
                token_report = {'prompt_strategy': 'V39_subject_scene_fusion'}
                used_reference = bool(fusion_audit.get('used_reference'))
                mode = 'fusion'
            else:
                prompt, prompt_2, negative_prompt, token_report = self._build_prompt_pair(spec, meta2, mode)
                gen = torch.Generator(device=self.device).manual_seed(sd) if self.device.startswith("cuda") else torch.Generator().manual_seed(sd)
                mode_steps = self.num_inference_steps + (2 if mode in {'scene_first', 'background', 'continuity'} else 0)
                mode_guidance = self.guidance_scale + (0.15 if mode in {'scene_first', 'continuity'} else 0.0)
                kwargs = dict(
                    prompt=prompt,
                    prompt_2=prompt_2,
                    negative_prompt=negative_prompt,
                    negative_prompt_2=negative_prompt,
                    width=self.width,
                    height=self.height,
                    num_inference_steps=mode_steps,
                    guidance_scale=mode_guidance,
                    generator=gen,
                )
                if reference_image is not None:
                    if strict_story_frame_only and text_has_anchor_ref:
                        effective_ip_adapter_scale = min(0.82, max(0.68, _ip_scale_for_mode(self.ip_adapter_scale + 0.16, mode, True)))
                    elif text_has_anchor_ref:
                        effective_ip_adapter_scale = min(0.74, max(0.60, _ip_scale_for_mode(self.ip_adapter_scale + 0.12, mode, True)))
                    else:
                        effective_ip_adapter_scale = min(0.56, _ip_scale_for_mode(self.ip_adapter_scale, mode, True))
                    try:
                        self.pipe.set_ip_adapter_scale(effective_ip_adapter_scale)
                    except Exception:
                        pass
                if self._apply_ip_adapter(reference_image):
                    kwargs["ip_adapter_image"] = reference_image
                    used_reference = True
                elif getattr(self, 'ip_adapter_loaded', False):
                    # Avoid diffusers ValueError when IP-Adapter is loaded but this mode has no reference image.
                    try:
                        self.pipe.set_ip_adapter_scale(0.0)
                    except Exception:
                        pass
                    kwargs["ip_adapter_image"] = self._neutral_ip_adapter_image()
                try:
                    kwargs["guidance_rescale"] = 0.15
                except Exception:
                    pass
                # V64.4: avoid the full-resolution attempt in CUDA-failsafe mode.
                # Retrying after an OOM can leave the CUDA context unstable and cause
                # CUBLAS/cuDNN/illegal-address failures. Generate low-res first, then upscale.
                if bool(getattr(self, "force_safe_lowres_generation", False)) or bool(meta2.get("force_safe_lowres_generation", False)):
                    safe_w = max(512, (min(int(kwargs.get("width", self.width)), int(getattr(self, "safe_generation_width", 768) or 768)) // 8) * 8)
                    safe_h = max(512, (min(int(kwargs.get("height", self.height)), int(getattr(self, "safe_generation_height", 768) or 768)) // 8) * 8)
                    kwargs["width"] = safe_w
                    kwargs["height"] = safe_h
                    kwargs["num_inference_steps"] = min(int(kwargs.get("num_inference_steps", self.num_inference_steps)), int(getattr(self, "safe_num_inference_steps", 34) or 34))
                    kwargs["guidance_scale"] = min(float(kwargs.get("guidance_scale", self.guidance_scale)), 8.0)
                    kwargs.pop("guidance_rescale", None)
                image = self._safe_pipe_call(kwargs, expected_size=(self.width, self.height), frame_id=frame_id, cid=cid, mode=mode)
            path = out_dir / f"frame_{int(frame_id):03d}_cand_{cid:02d}.png"
            image.save(path)
            results.append(
                CandidateImage(
                    frame_id=int(frame_id),
                    candidate_id=cid,
                    image_path=str(path),
                    prompt=prompt,
                    scores={
                        "image_quality": 0.0,
                        "colorfulness": 0.0,
                        "identity_consistency": 0.0,
                        "reference_subject_similarity": 0.0,
                        "subject_visibility": 0.0,
                        "crop_penalty": 0.0,
                        "emotion_visibility": 0.0,
                        "emotion_cause_visibility": 0.0,
                        "event_grounding": 0.0,
                        "evidence_visibility": 0.0,
                        "event_emotion_causal_consistency": 0.0,
                        "scene_alignment": 0.0,
                        "event_alignment": 0.0,
                        "story_alignment": 0.0,
                        "continuity": 0.0,
                        "overall": 0.0,
                    },
                    notes={
                        "generator_version": "V64_multi_gpu_candidate_parallel",
                        "worker_device": self.device,
                        "multi_gpu_enabled": bool(self.multi_gpu),
                        "multi_gpu_visible_ids": list(self.gpu_ids),
                        "multi_gpu_worker_count": len(self._multi_gpu_workers or [self]),
                        "seed": sd,
                        "prompt_variant_mode": mode,
                        "generation_route": "llm_frame_prompt_story_text2img",
                        "caption_locked_generation": True,
                        "identity_locked_generation": True,
                        "dcee_controlled_appearance_delta": getattr(spec, "dcee_appearance_delta", ""),
                        "scene_contract_generation": getattr(spec, "scene_contract", ""),
                        "story_faithful_generation": True,
                        "scene_first_generation": True,
                        "subject_scene_fusion_generation": bool(mode == "fusion"),
                        "visual_storytelling_generation": True,
                        "nonempty_background_generation": True,
                        "quality_model_preset": self.quality_model_preset,
                        "model_id": self.model_id,
                        "subject_reference_path": reference_path,
                        "canonical_reference_sheet_path": meta.get('canonical_reference_sheet_path', self.canonical_reference_sheet_path),
                        "previous_frame_image_path": meta.get('previous_frame_image_path', ''),
                        "previous_frame_caption": meta.get('previous_frame_caption', ''),
                        "previous_story_summary": meta.get('previous_story_summary', ''),
                        "previous_frame_local_caption": meta.get('previous_frame_local_caption', ''),
                        "next_frame_caption": meta.get('next_frame_caption', ''),
                        "previous_two_selected_image_paths": meta.get('previous_two_selected_image_paths', []),
                        "critical_visual_nouns": getattr(spec, 'critical_visual_nouns', []),
                        "ip_adapter_loaded": self.ip_adapter_loaded,
                        "ip_adapter_used": used_reference,
                        "ip_adapter_scale": effective_ip_adapter_scale,
                        "ip_adapter_error": self.ip_adapter_error,
                        "character_lora_loaded": self.character_lora_loaded,
                        "character_lora_error": self.character_lora_error,
                        "instantid_loaded": self.instantid_loaded,
                        "instantid_error": self.instantid_error,
                        "photomaker_loaded": self.photomaker_loaded,
                        "photomaker_error": self.photomaker_error,
                        "identity_backend_priority": self.identity_backend_priority,
                        "identity_backend_selected": active_identity_backend,
                        "reference_bank_paths": reference_bank_paths,
                        "fusion_audit": fusion_audit,
                        "frame_visual_spec": spec.to_dict(),
                        "prompt_2": prompt_2,
                        "negative_prompt": negative_prompt,
                        "token_report": token_report,
                        "visualized_action": meta2.get("visualized_action", ""),
                    },
                )
            )
            try:
                del image
            except Exception:
                pass
            self._clear_cuda_cache()
        return results
