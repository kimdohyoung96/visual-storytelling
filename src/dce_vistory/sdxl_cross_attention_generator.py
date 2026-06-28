from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple, Dict
import torch
from PIL import Image, ImageEnhance, ImageStat

from .adapters_pytorch import ButterflyAdapterStack
from .latent_schema import VisualControlPacket
from .schema import CandidateImage
from .frame_director import FrameVisualSpec, prompt_from_spec, negative_from_spec


_MODEL_PRESETS = {
    "sdxl_base": "stabilityai/stable-diffusion-xl-base-1.0",
    "juggernaut_xl": "RunDiffusion/Juggernaut-XL-v9",
    "realvis_xl": "SG161222/RealVisXL_V5.0",
}


def _path_exists(x: Any) -> bool:
    try:
        return bool(x) and Path(str(x)).exists()
    except Exception:
        return False


def _load_rgb_image(path: str, size=(1024, 1024)) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail(size)
    return img


def _combine_reference_images(paths: List[str], size=(1024, 1024)):
    valid = [str(p) for p in paths if _path_exists(p)]
    if not valid:
        return None, ""
    imgs = [_load_rgb_image(p, size=(size[0] // max(1, min(3, len(valid))), size[1])) for p in valid[:3]]
    if len(imgs) == 1:
        return imgs[0], valid[0]
    cols = len(imgs)
    w = max(1, size[0] // cols)
    canvas = Image.new("RGB", size, "white")
    for i, img in enumerate(imgs):
        local = img.copy()
        local.thumbnail((w - 8, size[1] - 8))
        x0 = i * w + (w - local.width) // 2
        y0 = (size[1] - local.height) // 2
        canvas.paste(local, (x0, y0))
    return canvas, " | ".join(valid[:3])


def _spec_from_packet(packet: VisualControlPacket) -> FrameVisualSpec:
    m = packet.control_metadata or {}
    d = m.get("frame_visual_spec", {}) or {}
    if not d:
        d = {
            "frame_id": getattr(packet, "frame_id", 0),
            "total_frames": 6,
            "story_sentence": m.get("story_sentence", ""),
            "protagonist": "protagonist",
            "subject_identity": m.get("character_text", ""),
            "subject_reference_policy": "Use text identity strictly.",
            "primary_action": m.get("event_text", ""),
            "visible_event": m.get("event_text", ""),
            "visible_cause": m.get("event_grounding", ""),
            "required_objects": m.get("must_show", []),
            "carry_over_entities": [],
            "recurring_entities": [],
            "forbidden_objects": ["generic portrait"],
            "location": "",
            "weather": "",
            "atmosphere": "",
            "emotion": "",
            "facial_expression": "",
            "body_pose": "",
            "camera": "",
            "continuity": "",
            "negative": "generic portrait",
        }
    return FrameVisualSpec(**d)


def _resolve_model_id(model_id: str, quality_model_preset: str | None = None) -> str:
    if quality_model_preset and quality_model_preset in _MODEL_PRESETS:
        return _MODEL_PRESETS[quality_model_preset]
    return model_id


def _mean_brightness(img: Image.Image) -> float:
    return float(ImageStat.Stat(img.convert("L")).mean[0]) / 255.0


def _auto_fix_visibility(img: Image.Image) -> tuple[Image.Image, Dict[str, float]]:
    out = img
    gray = out.convert("L")
    stat = ImageStat.Stat(gray)
    brightness = float(stat.mean[0]) / 255.0
    contrast = float(stat.stddev[0]) / 255.0
    changed = False
    if brightness < 0.35:
        out = ImageEnhance.Brightness(out).enhance(1.18)
        changed = True
    if contrast < 0.18:
        out = ImageEnhance.Contrast(out).enhance(1.10)
        changed = True
    if changed:
        # keep color stable
        out = ImageEnhance.Color(out).enhance(1.03)
    return out, {
        "brightness_before": round(brightness, 4),
        "contrast_before": round(contrast, 4),
        "auto_fixed": changed,
    }


class SDXLButterflyCrossAttentionGenerator:
    """V31 story-faithful generator.

    Improvements over V30:
    - stronger story-to-image alignment via caption/evidence prompt pair
    - stronger protagonist consistency via subject reference + previous-frame continuity route
    - optional img2img continuity generation for later frames
    - visibility rescue for dark scenes
    - optional quality model preset while keeping the DCEE pipeline unchanged
    """

    def __init__(
        self,
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        device="cuda",
        width=1024,
        height=1024,
        num_inference_steps=44,
        guidance_scale=9.0,
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
        ip_adapter_scale=0.42,
        use_butterfly_adapter=False,
        use_refiner=True,
        refiner_model_id="stabilityai/stable-diffusion-xl-refiner-1.0",
        refiner_strength=0.80,
        aesthetic_score=6.2,
        negative_aesthetic_score=2.5,
        quality_model_preset="sdxl_base",
        use_previous_frame_img2img=True,
        previous_frame_strength=0.40,
    ):
        from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler

        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.seed = int(seed)
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.use_butterfly_adapter = bool(use_butterfly_adapter and adapter_ckpt)
        self.use_refiner = bool(use_refiner and device.startswith("cuda"))
        self.refiner_strength = float(refiner_strength)
        self.aesthetic_score = float(aesthetic_score)
        self.negative_aesthetic_score = float(negative_aesthetic_score)
        self.model_id = _resolve_model_id(model_id, quality_model_preset)
        self.quality_model_preset = quality_model_preset
        self.use_previous_frame_img2img = bool(use_previous_frame_img2img)
        self.previous_frame_strength = float(previous_frame_strength)

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            use_safetensors=True,
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config, use_karras_sigmas=True)
        if enable_cpu_offload and device.startswith("cuda"):
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device)
        try:
            self.pipe.vae.enable_slicing()
            self.pipe.vae.enable_tiling()
        except Exception:
            pass

        self.story_img2img = None
        self.story_img2img_error = ""
        if self.use_previous_frame_img2img:
            try:
                self.story_img2img = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=self.dtype,
                    use_safetensors=True,
                )
                self.story_img2img.scheduler = DPMSolverMultistepScheduler.from_config(self.story_img2img.scheduler.config, use_karras_sigmas=True)
                if enable_cpu_offload and device.startswith("cuda"):
                    self.story_img2img.enable_model_cpu_offload()
                else:
                    self.story_img2img.to(device)
            except Exception as e:
                self.story_img2img = None
                self.story_img2img_error = str(e)[:500]
                self.use_previous_frame_img2img = False

        self.refiner = None
        self.refiner_error = ""
        if self.use_refiner:
            try:
                self.refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    refiner_model_id,
                    text_encoder_2=self.pipe.text_encoder_2,
                    vae=self.pipe.vae,
                    torch_dtype=self.dtype,
                    use_safetensors=True,
                )
                self.refiner.scheduler = DPMSolverMultistepScheduler.from_config(self.refiner.scheduler.config, use_karras_sigmas=True)
                if enable_cpu_offload and device.startswith("cuda"):
                    self.refiner.enable_model_cpu_offload()
                else:
                    self.refiner.to(device)
            except Exception as e:
                self.refiner = None
                self.refiner_error = str(e)[:500]
                self.use_refiner = False

        self.ip_adapter_loaded = False
        self.ip_adapter_error = ""
        self.ip_adapter_scale = float(ip_adapter_scale)
        if use_ip_adapter:
            try:
                self.pipe.load_ip_adapter(ip_adapter_repo, subfolder=ip_adapter_subfolder, weight_name=ip_adapter_weight_name)
                self.pipe.set_ip_adapter_scale(self.ip_adapter_scale)
                if self.story_img2img is not None:
                    self.story_img2img.load_ip_adapter(ip_adapter_repo, subfolder=ip_adapter_subfolder, weight_name=ip_adapter_weight_name)
                    self.story_img2img.set_ip_adapter_scale(self.ip_adapter_scale)
                self.ip_adapter_loaded = True
            except Exception as e:
                self.ip_adapter_loaded = False
                self.ip_adapter_error = str(e)[:500]

        self.adapter_stack = None
        if self.use_butterfly_adapter:
            self.adapter_stack = ButterflyAdapterStack(character_tokens, world_tokens, emotion_tokens, event_tokens, evidence_tokens).to(device)
            if self.dtype == torch.float16:
                self.adapter_stack = self.adapter_stack.half()
            self.adapter_stack.load_state_dict(torch.load(adapter_ckpt, map_location=device), strict=False)
            self.adapter_stack.eval()

    def _gather_reference_paths(self, packet) -> Dict[str, List[str]]:
        refs = getattr(packet, "reference_images", {}) or {}
        meta = getattr(packet, "control_metadata", {}) or {}
        subject_paths: List[str] = []
        continuity_paths: List[str] = []

        for key in ["subject", "source_reference_image_path", "input_reference_image_path"]:
            val = refs.get(key) if key in refs else meta.get(key, "")
            if _path_exists(val) and str(val) not in subject_paths:
                subject_paths.append(str(val))

        mem = refs.get("memory_sequence", []) or meta.get("memory_sequence", []) or []
        if isinstance(mem, str):
            mem = [mem]
        for p in mem:
            if _path_exists(p) and str(p) not in continuity_paths:
                continuity_paths.append(str(p))

        for key in ["previous_frame_image_path", "previous_selected_image_path", "selected_image_path"]:
            val = meta.get(key, "")
            if _path_exists(val) and str(val) not in continuity_paths:
                continuity_paths.append(str(val))

        src = meta.get("source_reference_image_path", "")
        if _path_exists(src) and str(src) not in subject_paths:
            subject_paths.append(str(src))

        return {"subject": subject_paths[:1], "continuity": continuity_paths[-2:]}

    def _reference_images_for_packet(self, packet):
        paths = self._gather_reference_paths(packet)
        subject_img, subject_desc = _combine_reference_images(paths["subject"][:1], size=(1024, 1024))
        continuity_img, continuity_desc = _combine_reference_images(paths["continuity"][-1:], size=(1024, 1024))
        both_img, both_desc = _combine_reference_images(paths["subject"][:1] + paths["continuity"][-1:], size=(1024, 1024))
        return {
            "subject_image": subject_img,
            "subject_path": subject_desc,
            "continuity_image": continuity_img,
            "continuity_path": continuity_desc,
            "combined_image": both_img,
            "combined_path": both_desc,
        }

    def _build_prompt_pair(self, spec: FrameVisualSpec, mode: str) -> Tuple[str, str, str]:
        main_prompt = prompt_from_spec(spec, mode)
        compact_prompt_2 = (
            f"frame {spec.frame_id}/{spec.total_frames}; current caption: {spec.story_sentence}; "
            f"one protagonist only: {spec.protagonist}; "
            f"current action: {spec.primary_action or spec.visible_event}; "
            f"visible cause/evidence: {spec.visible_cause}; "
            f"location: {spec.location}; weather: {spec.weather}; atmosphere: {spec.atmosphere}; emotion: {spec.emotion}; "
            f"required visible objects: {', '.join(spec.required_objects[:8])}; "
            f"identity continuity: {spec.subject_identity}; "
            f"do not introduce other characters or duplicate the protagonist"
        )
        negative_prompt = negative_from_spec(spec)
        return main_prompt, compact_prompt_2, negative_prompt

    def _apply_ip_adapter(self, pipe, image):
        if self.ip_adapter_loaded and image is not None:
            try:
                pipe.set_ip_adapter_scale(self.ip_adapter_scale)
            except Exception:
                pass
            return True
        return False

    def _run_text2img(self, prompt: str, prompt_2: str, negative_prompt: str, generator, reference_image=None):
        kwargs = dict(
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
            width=self.width,
            height=self.height,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
            aesthetic_score=self.aesthetic_score,
            negative_aesthetic_score=self.negative_aesthetic_score,
        )
        used_reference = False
        if self._apply_ip_adapter(self.pipe, reference_image):
            kwargs["ip_adapter_image"] = reference_image
            used_reference = True
        if self.use_refiner and self.refiner is not None:
            latent = self.pipe(**kwargs, output_type="latent", denoising_end=self.refiner_strength).images
            image = self.refiner(
                prompt=prompt,
                prompt_2=prompt_2,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt,
                image=latent,
                num_inference_steps=max(20, self.num_inference_steps),
                denoising_start=self.refiner_strength,
                guidance_scale=max(5.5, self.guidance_scale - 1.0),
                generator=generator,
                aesthetic_score=self.aesthetic_score,
                negative_aesthetic_score=self.negative_aesthetic_score,
            ).images[0]
        else:
            image = self.pipe(**kwargs).images[0]
        return image, used_reference

    def _run_continuity_img2img(self, prompt: str, prompt_2: str, negative_prompt: str, generator, init_image: Image.Image, subject_reference=None):
        if self.story_img2img is None:
            return self._run_text2img(prompt, prompt_2, negative_prompt, generator, reference_image=subject_reference)
        kwargs = dict(
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
            image=init_image.resize((self.width, self.height)),
            strength=self.previous_frame_strength,
            num_inference_steps=max(34, self.num_inference_steps),
            guidance_scale=max(8.0, self.guidance_scale),
            generator=generator,
            aesthetic_score=self.aesthetic_score,
            negative_aesthetic_score=self.negative_aesthetic_score,
        )
        used_reference = False
        if self._apply_ip_adapter(self.story_img2img, subject_reference):
            kwargs["ip_adapter_image"] = subject_reference
            used_reference = True
        image = self.story_img2img(**kwargs).images[0]
        return image, used_reference

    @torch.no_grad()
    def generate_from_packet(self, packet: VisualControlPacket, frame_id: int, out_dir: Path, num_candidates: int = 1) -> List[CandidateImage]:
        out_dir.mkdir(parents=True, exist_ok=True)
        spec = _spec_from_packet(packet)
        refs = self._reference_images_for_packet(packet)
        subject_reference = refs["subject_image"]
        continuity_reference = refs["continuity_image"]
        combined_reference = refs["combined_image"] or subject_reference
        modes = ["caption_locked", "evidence_locked", "continuity_locked", "visibility_locked", "emotion_locked"]

        res = []
        for cid in range(num_candidates):
            mode = modes[cid % len(modes)]
            prompt, prompt_2, negative_prompt = self._build_prompt_pair(spec, mode)
            sd = self.seed + int(frame_id) * 1000 + cid
            gen = torch.Generator(device=self.device).manual_seed(sd) if self.device.startswith("cuda") else torch.Generator().manual_seed(sd)

            use_continuity_route = bool(frame_id > 1 and continuity_reference is not None and self.use_previous_frame_img2img and (cid % 2 == 1 or mode == "continuity_locked"))
            if use_continuity_route:
                img, used_reference = self._run_continuity_img2img(
                    prompt, prompt_2, negative_prompt, gen,
                    init_image=continuity_reference,
                    subject_reference=subject_reference,
                )
                route = "prev_frame_img2img"
            else:
                img, used_reference = self._run_text2img(prompt, prompt_2, negative_prompt, gen, reference_image=combined_reference)
                route = "caption_text2img"

            img, visibility_meta = _auto_fix_visibility(img)
            path = out_dir / f"frame_{int(frame_id):03d}_cand_{cid:02d}.png"
            img.save(path)

            res.append(CandidateImage(
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
                    "seed": sd,
                    "generator_version": "V31",
                    "prompt_variant_mode": mode,
                    "generation_route": route,
                    "caption_locked_generation": True,
                    "english_caption_contract": True,
                    "story_faithful_generation": True,
                    "quality_model_preset": self.quality_model_preset,
                    "model_id": self.model_id,
                    "subject_reference_path": refs["subject_path"],
                    "continuity_reference_path": refs["continuity_path"],
                    "combined_reference_path": refs["combined_path"],
                    "ip_adapter_loaded": self.ip_adapter_loaded,
                    "ip_adapter_used": used_reference,
                    "ip_adapter_scale": self.ip_adapter_scale,
                    "ip_adapter_error": self.ip_adapter_error,
                    "story_img2img_enabled": self.use_previous_frame_img2img,
                    "story_img2img_error": self.story_img2img_error,
                    "previous_frame_strength": self.previous_frame_strength,
                    "refiner_enabled": self.use_refiner,
                    "refiner_error": self.refiner_error,
                    "frame_visual_spec": spec.to_dict(),
                    "prompt_2": prompt_2,
                    "negative_prompt": negative_prompt,
                    **visibility_meta,
                },
            ))
        return res
