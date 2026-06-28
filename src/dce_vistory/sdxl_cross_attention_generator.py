from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple, Dict
import torch
from PIL import Image

from .adapters_pytorch import ButterflyAdapterStack
from .latent_schema import VisualControlPacket
from .schema import CandidateImage
from .frame_director import FrameVisualSpec


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


def _load_reference_image(path: str, size=(1024, 1024)) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail(size)
    canvas = Image.new("RGB", size, "white")
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


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


def _join_required_objects(items: List[str], limit: int = 5, per_item_words: int = 4) -> str:
    vals = []
    for x in items or []:
        s = _clip_words(x, per_item_words)
        low = s.lower()
        if not s:
            continue
        if any(k in low for k in ["person", "human", "man", "woman", "child", "girl", "boy", "helper", "friend", "extra animal", "another"]):
            continue
        if s not in vals:
            vals.append(s)
    return ", ".join(vals[:limit])


def _lighting_hint(spec: FrameVisualSpec) -> str:
    mood = f"{spec.emotion} {spec.atmosphere} {spec.weather}".lower()
    if any(k in mood for k in ["sad", "empty", "lonely", "dark", "night", "gloom", "melancholy", "sorrow", "rain"]):
        return "dark emotional background, but the protagonist is clearly visible with soft rim light"
    return "clear readable lighting"


def _critical_negative() -> str:
    return (
        "human, person, man, woman, child, boy, girl, people, crowd, helper, companion, "
        "extra animal, second protagonist, duplicate protagonist, multiple bears, two bears, "
        "unrelated object, unrelated prop, wrong event, missing action, missing evidence, "
        "generic portrait, close-up portrait, split screen, comic panel, collage, multiple scenes, "
        "cropped face, cropped body, cropped feet, cropped paws, cut off body, "
        "underexposed protagonist, unreadable face, blurry, deformed, bad anatomy, watermark, text"
    )


class SDXLButterflyCrossAttentionGenerator:
    """V31.2 Lite quality-stable SDXL wrapper.

    Removed from V31:
    - previous-frame img2img route
    - multi-reference collage
    - default refiner loading
    - long free-form frame_director prompt as the actual SDXL prompt

    Kept:
    - English caption contract
    - protagonist/source reference only
    - short caption-first SDXL prompt
    - token audit logging
    - multiple candidates with small prompt variants
    """

    def __init__(
        self,
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        device="cuda",
        width=1024,
        height=1024,
        num_inference_steps=36,
        guidance_scale=7.5,
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
        ip_adapter_scale=0.26,
        use_butterfly_adapter=False,
        quality_model_preset="sdxl_base",
        use_refiner=False,
        refiner_model_id=None,
        refiner_strength=0.80,
        aesthetic_score=6.0,
        negative_aesthetic_score=2.5,
        use_previous_frame_img2img=False,
        previous_frame_strength=0.40,
    ):
        from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

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
        self.refiner_error = "disabled in V31.2 Lite for stability"
        self.use_previous_frame_img2img = False
        self.story_img2img_error = "disabled in V31.2 Lite for identity/shape stability"
        self.previous_frame_strength = float(previous_frame_strength)

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

        try:
            self.pipe.vae.enable_slicing()
        except Exception:
            pass

        self.ip_adapter_loaded = False
        self.ip_adapter_error = ""
        self.ip_adapter_scale = float(ip_adapter_scale)
        if use_ip_adapter:
            try:
                self.pipe.load_ip_adapter(
                    ip_adapter_repo,
                    subfolder=ip_adapter_subfolder,
                    weight_name=ip_adapter_weight_name,
                )
                self.pipe.set_ip_adapter_scale(self.ip_adapter_scale)
                self.ip_adapter_loaded = True
            except Exception as e:
                self.ip_adapter_loaded = False
                self.ip_adapter_error = str(e)[:500]

        self.use_butterfly_adapter = False
        self.adapter_stack = None

    def _subject_reference_image(self, packet):
        refs = getattr(packet, "reference_images", {}) or {}
        meta = getattr(packet, "control_metadata", {}) or {}
        for key in ["subject", "source_reference_image_path", "input_reference_image_path"]:
            p = refs.get(key) if key in refs else meta.get(key, "")
            if _path_exists(p):
                return _load_reference_image(str(p), size=(1024, 1024)), str(p)
        src = meta.get("source_reference_image_path", "")
        if _path_exists(src):
            return _load_reference_image(str(src), size=(1024, 1024)), str(src)
        return None, ""

    def _token_count(self, tokenizer, text: str):
        try:
            return len(tokenizer(text, truncation=False, add_special_tokens=True).input_ids)
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

    def _fit_segments(self, tokenizer, segments: List[str], reserve: int = 2):
        max_len = int(getattr(tokenizer, "model_max_length", 77) or 77)
        if max_len > 1000:
            max_len = 77
        kept, dropped = [], []
        for seg in segments:
            seg = _clean(seg)
            if not seg:
                continue
            trial = ". ".join(kept + [seg])
            n = self._token_count(tokenizer, trial)
            if n is None or n <= max_len - reserve or not kept:
                kept.append(seg)
            else:
                dropped.append(seg)
        prompt = ". ".join(kept)
        return prompt, {"kept_segments": kept, "dropped_segments": dropped, "dropped_count": len(dropped), "max_length": max_len, "final_tokens": self._token_count(tokenizer, prompt)}

    def _build_prompt_pair(self, spec: FrameVisualSpec, mode: str):
        caption = _clip_words(spec.story_sentence, 24)
        action = _clip_words(spec.primary_action or spec.visible_event, 10)
        evidence = _clip_words(spec.visible_cause, 10)
        protagonist = _clip_words(spec.protagonist or "protagonist", 6)
        identity = _clip_words(spec.subject_identity, 14)
        required = _join_required_objects(spec.required_objects, 5, 4)
        location = _clip_words(spec.location, 8)
        emotion = _clip_words(spec.emotion, 4)
        lighting = _lighting_hint(spec)

        variant = {
            "caption": "match the exact caption",
            "evidence": "make the evidence object visible",
            "identity": "same protagonist identity as reference",
            "composition": "full body, uncropped, readable pose",
        }.get(mode, "match the exact caption")

        segments_1 = [
            "professional full-color storybook illustration",
            f"exact frame caption: {caption}",
            f"exactly one protagonist: {protagonist}",
            f"same identity as reference: {identity}",
            f"visible action: {action}",
            f"visible evidence: {evidence}",
            f"required objects: {required}" if required else "",
            f"location: {location}" if location else "",
            f"emotion: {emotion}" if emotion else "",
            "single coherent scene",
            "medium-wide shot, uncropped face and limbs",
            lighting,
            variant,
            "no humans, no extra animals, no duplicate protagonist",
        ]

        segments_2 = [
            f"caption: {caption}",
            f"one protagonist: {protagonist}",
            f"action: {action}",
            f"evidence: {evidence}",
            f"objects: {required}" if required else "",
            f"place: {location}" if location else "",
            "no extra subjects",
        ]

        tok1 = getattr(self.pipe, "tokenizer", None)
        tok2 = getattr(self.pipe, "tokenizer_2", None) or tok1
        if tok1 is not None:
            prompt, fit1 = self._fit_segments(tok1, segments_1)
        else:
            prompt, fit1 = ". ".join([s for s in segments_1 if s]), {}
        if tok2 is not None:
            prompt_2, fit2 = self._fit_segments(tok2, segments_2)
        else:
            prompt_2, fit2 = ". ".join([s for s in segments_2 if s]), {}

        negative_prompt = _critical_negative()
        audit = self._token_report(prompt, prompt_2, negative_prompt)
        audit["prompt_fit_tokenizer_1"] = fit1
        audit["prompt_fit_tokenizer_2"] = fit2
        audit["prompt_strategy"] = "V31_2_lite_caption_first_token_guard"
        return prompt, prompt_2, negative_prompt, audit

    def _apply_ip_adapter(self, reference_image):
        if self.ip_adapter_loaded and reference_image is not None:
            try:
                self.pipe.set_ip_adapter_scale(self.ip_adapter_scale)
            except Exception:
                pass
            return True
        return False

    @torch.no_grad()
    def generate_from_packet(
        self,
        packet: VisualControlPacket,
        frame_id: int,
        out_dir: Path,
        num_candidates: int = 4,
    ) -> List[CandidateImage]:
        out_dir.mkdir(parents=True, exist_ok=True)

        spec = _spec_from_packet(packet)
        reference_image, reference_path = self._subject_reference_image(packet)
        modes = ["caption", "evidence", "identity", "composition"]

        results: List[CandidateImage] = []
        for cid in range(num_candidates):
            mode = modes[cid % len(modes)]
            prompt, prompt_2, negative_prompt, token_report = self._build_prompt_pair(spec, mode)
            sd = self.seed + int(frame_id) * 1000 + cid
            gen = torch.Generator(device=self.device).manual_seed(sd) if self.device.startswith("cuda") else torch.Generator().manual_seed(sd)

            kwargs = dict(
                prompt=prompt,
                prompt_2=prompt_2,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt,
                width=self.width,
                height=self.height,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=gen,
            )
            used_reference = False
            if self._apply_ip_adapter(reference_image):
                kwargs["ip_adapter_image"] = reference_image
                used_reference = True

            image = self.pipe(**kwargs).images[0]
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
                        "generator_version": "V31.2-Lite",
                        "seed": sd,
                        "prompt_variant_mode": mode,
                        "generation_route": "caption_text2img_subject_reference_only",
                        "caption_locked_generation": True,
                        "english_caption_contract": True,
                        "story_faithful_generation": True,
                        "removed_v31_components": [
                            "previous_frame_img2img",
                            "multi_reference_collage",
                            "default_refiner",
                            "long_freeform_prompt"
                        ],
                        "quality_model_preset": self.quality_model_preset,
                        "model_id": self.model_id,
                        "subject_reference_path": reference_path,
                        "ip_adapter_loaded": self.ip_adapter_loaded,
                        "ip_adapter_used": used_reference,
                        "ip_adapter_scale": self.ip_adapter_scale,
                        "ip_adapter_error": self.ip_adapter_error,
                        "frame_visual_spec": spec.to_dict(),
                        "prompt_2": prompt_2,
                        "negative_prompt": negative_prompt,
                        "token_report": token_report,
                    },
                )
            )
        return results
