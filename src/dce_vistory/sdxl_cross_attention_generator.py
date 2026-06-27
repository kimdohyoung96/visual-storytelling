from __future__ import annotations

from pathlib import Path
from typing import Any, List
import torch
from PIL import Image

from .adapters_pytorch import ButterflyAdapterStack
from .latent_schema import VisualControlPacket
from .schema import CandidateImage
from .frame_director import FrameVisualSpec, prompt_from_spec, negative_from_spec


def _path_exists(x: Any) -> bool:
    try:
        return bool(x) and Path(str(x)).exists()
    except Exception:
        return False


def _load_rgb_image(path: str, size=(1024, 1024)):
    img = Image.open(path).convert("RGB")
    img.thumbnail(size)
    return img

def _combine_reference_images(paths: List[str], size=(1024, 1024)):
    valid = [str(p) for p in paths if _path_exists(p)]
    if not valid:
        return None, ''
    imgs = [_load_rgb_image(p, size=(size[0] // max(1, min(3, len(valid))), size[1])) for p in valid[:3]]
    if len(imgs) == 1:
        return imgs[0], valid[0]
    cols = len(imgs)
    w = max(1, size[0] // cols)
    canvas = Image.new('RGB', size, 'white')
    for i, img in enumerate(imgs):
        local = img.copy()
        local.thumbnail((w - 8, size[1] - 8))
        x0 = i * w + (w - local.width) // 2
        y0 = (size[1] - local.height) // 2
        canvas.paste(local, (x0, y0))
    return canvas, ' | '.join(valid[:3])


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


class SDXLButterflyCrossAttentionGenerator:
    """
    Sentence-locked SDXL generator.

    This version sends direct prompt strings to SDXL. It disables the untrained
    ButterflyAdapterStack by default because random/untrained adapter tokens can weaken
    the exact sentence-to-image mapping.
    """

    def __init__(
        self,
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        device="cuda",
        width=1024,
        height=1024,
        num_inference_steps=40,
        guidance_scale=8.5,
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
        ip_adapter_scale=0.28,
        use_butterfly_adapter=False,
    ):
        from diffusers import StableDiffusionXLPipeline

        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.seed = int(seed)
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.use_butterfly_adapter = bool(use_butterfly_adapter and adapter_ckpt)

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            use_safetensors=True,
        )
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

        self.adapter_stack = None
        if self.use_butterfly_adapter:
            self.adapter_stack = ButterflyAdapterStack(
                character_tokens,
                world_tokens,
                emotion_tokens,
                event_tokens,
                evidence_tokens,
            ).to(device)
            if self.dtype == torch.float16:
                self.adapter_stack = self.adapter_stack.half()
            self.adapter_stack.load_state_dict(torch.load(adapter_ckpt, map_location=device), strict=False)
            self.adapter_stack.eval()

    def _reference_image_for_packet(self, packet):
        refs = getattr(packet, "reference_images", {}) or {}
        paths = []
        if refs.get('subject'):
            paths.append(refs.get('subject'))
        mem = refs.get('memory_sequence', []) or []
        if isinstance(mem, str):
            mem = [mem]
        paths.extend(mem[:2])
        if not paths:
            meta = packet.control_metadata or {}
            src = meta.get('source_reference_image_path', '')
            if _path_exists(src):
                paths.append(src)
        img, desc = _combine_reference_images(paths, size=(1024, 1024))
        if img is not None:
            return img, desc
        return None, ''

    @torch.no_grad()
    def generate_from_packet(
        self,
        packet: VisualControlPacket,
        frame_id: int,
        out_dir: Path,
        num_candidates: int = 1,
    ) -> List[CandidateImage]:
        out_dir.mkdir(parents=True, exist_ok=True)

        spec = _spec_from_packet(packet)
        modes = ["sentence_locked", "object_locked", "continuity_locked", "background_locked", "emotion_locked"]
        reference_image, reference_path = self._reference_image_for_packet(packet)

        res = []
        for cid in range(num_candidates):
            mode = modes[cid % len(modes)]
            prompt = prompt_from_spec(spec, mode)
            negative_prompt = negative_from_spec(spec)

            sd = self.seed + int(frame_id) * 1000 + cid
            gen = torch.Generator(device=self.device).manual_seed(sd) if self.device.startswith("cuda") else torch.Generator().manual_seed(sd)

            kwargs = dict(
                prompt=prompt,
                prompt_2=prompt,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt,
                width=self.width,
                height=self.height,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=gen,
            )

            used_reference = False
            if self.ip_adapter_loaded and reference_image is not None:
                try:
                    self.pipe.set_ip_adapter_scale(self.ip_adapter_scale)
                except Exception:
                    pass
                kwargs["ip_adapter_image"] = reference_image
                used_reference = True

            img = self.pipe(**kwargs).images[0]
            path = out_dir / f"frame_{int(frame_id):03d}_cand_{cid:02d}.png"
            img.save(path)

            res.append(
                CandidateImage(
                    frame_id=int(frame_id),
                    candidate_id=cid,
                    image_path=str(path),
                    prompt=prompt,
                    scores={
                        "image_quality": 0.0,
                        "identity_consistency": 0.0,
                        "reference_subject_similarity": 0.0,
                        "emotion_visibility": 0.0,
                        "emotion_cause_visibility": 0.0,
                        "event_grounding": 0.0,
                        "evidence_visibility": 0.0,
                        "event_emotion_causal_consistency": 0.0,
                        "scene_alignment": 0.0,
                        "event_alignment": 0.0,
                        "story_alignment": 0.0,
                        "colorfulness": 0.0,
                        "continuity": 0.0,
                        "overall": 0.0,
                    },
                    notes={
                        "seed": sd,
                        "prompt_variant_mode": mode,
                        "sentence_locked_generation": True,
                        "untrained_butterfly_adapter_disabled": not self.use_butterfly_adapter,
                        "reference_image_path": reference_path,
                        "ip_adapter_loaded": self.ip_adapter_loaded,
                        "ip_adapter_used": used_reference,
                        "ip_adapter_scale": self.ip_adapter_scale,
                        "ip_adapter_error": self.ip_adapter_error,
                        "frame_visual_spec": spec.to_dict(),
                        "negative_prompt": negative_prompt,
                    },
                )
            )
        return res
