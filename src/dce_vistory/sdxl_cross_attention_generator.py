from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch

from .adapters_pytorch import ButterflyAdapterStack
from .latent_schema import VisualControlPacket
from .schema import CandidateImage


class SDXLButterflyCrossAttentionGenerator:
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: str = "cuda",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 40,
        guidance_scale: float = 8.0,
        seed: int = 42,
        adapter_ckpt: str | None = None,
        enable_cpu_offload: bool = False,
        character_tokens: int = 8,
        world_tokens: int = 8,
        emotion_tokens: int = 8,
        event_tokens: int = 8,
    ):
        from diffusers import StableDiffusionXLPipeline

        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.seed = int(seed)

        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.dtype = dtype

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
        )

        if enable_cpu_offload and device.startswith("cuda"):
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device)

        try:
            self.pipe.vae.to(torch.float32)
        except Exception:
            pass

        try:
            self.pipe.vae.enable_slicing()
        except Exception:
            try:
                self.pipe.enable_vae_slicing()
            except Exception:
                pass

        self.adapter_stack = ButterflyAdapterStack(
            character_tokens=character_tokens,
            world_tokens=world_tokens,
            emotion_tokens=emotion_tokens,
            event_tokens=event_tokens,
        ).to(device)

        if dtype == torch.float16:
            self.adapter_stack = self.adapter_stack.half()

        if adapter_ckpt:
            self.adapter_stack.load_state_dict(torch.load(adapter_ckpt, map_location=device), strict=False)

        self.adapter_stack.eval()

    def _encode_prompt(self, prompt: str, negative_prompt: str | None = None, do_cfg: bool = True):
        result = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=negative_prompt if negative_prompt is not None else "",
        )
        if len(result) != 4:
            raise RuntimeError(f"Unexpected encode_prompt output length: {len(result)}")
        return result

    def _encode_control(self, text: str) -> torch.Tensor:
        prompt_embeds, _, _, _ = self._encode_prompt(text, negative_prompt="", do_cfg=False)
        return prompt_embeds

    def _control_texts_from_packet(self, packet: VisualControlPacket) -> Dict[str, str]:
        meta = packet.control_metadata or {}
        return {
            "character_adapter": meta.get("character_text", json.dumps(meta.get("character", {}), ensure_ascii=False)),
            "world_adapter": meta.get("world_text", json.dumps(meta.get("world", {}), ensure_ascii=False)),
            "emotion_adapter": meta.get("emotion_text", json.dumps(meta.get("emotion", {}), ensure_ascii=False)),
            "event_adapter": meta.get("event_text", packet.positive_prompt[:1200]),
        }

    @torch.no_grad()
    def generate_from_packet(
        self,
        packet: VisualControlPacket,
        frame_id: int,
        out_dir: Path,
        num_candidates: int = 1,
    ) -> List[CandidateImage]:
        out_dir.mkdir(parents=True, exist_ok=True)
        candidates: List[CandidateImage] = []

        strong_positive = (
            packet.positive_prompt
            + "\n\nIMPORTANT VISUAL REQUIREMENTS:\n"
            + "- Render this as a FULL-COLOR cinematic storybook illustration.\n"
            + "- Never produce monochrome, grayscale, black-and-white, pencil-only, or sketch-only output.\n"
            + "- Use rich natural colors and emotionally meaningful color palette.\n"
            + "- The protagonist emotion must be immediately readable from face, eyes, body posture, lighting, color, and environment.\n"
            + "- The scene must clearly show the current story event and the visual reason for the protagonist emotion.\n"
            + "- Background, weather, time of day, and atmosphere must match the story frame.\n"
            + "- Avoid plain portrait-only composition unless the frame explicitly requires close-up emotional reaction.\n"
        )

        strong_negative = (
            packet.negative_prompt
            + ", monochrome, grayscale, black and white, pencil sketch, charcoal sketch, line art only, "
            + "emotionless face, flat expression, stiff pose, flat lighting, colorless image, empty background, portrait only"
        )

        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self._encode_prompt(
            strong_positive,
            negative_prompt=strong_negative,
            do_cfg=True,
        )

        control_texts = self._control_texts_from_packet(packet)
        control_embeds = {name: self._encode_control(text) for name, text in control_texts.items()}

        prompt_embeds = prompt_embeds.to(self.device, self.dtype)
        negative_prompt_embeds = negative_prompt_embeds.to(self.device, self.dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(self.device, self.dtype)
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(self.device, self.dtype)

        augmented_prompt_embeds, adapter_tokens = self.adapter_stack(
            prompt_embeds,
            control_embeds=control_embeds,
            adapter_weights=packet.adapter_weights or {},
        )

        if adapter_tokens.shape[1] > 0:
            negative_prompt_embeds = torch.cat(
                [negative_prompt_embeds, torch.zeros_like(adapter_tokens)],
                dim=1,
            )

        for cid in range(num_candidates):
            seed = self.seed + int(frame_id) * 1000 + cid
            generator = (
                torch.Generator(device=self.device).manual_seed(seed)
                if self.device.startswith("cuda")
                else torch.Generator().manual_seed(seed)
            )

            gs = self.guidance_scale
            if "intensity 5/5" in strong_positive.lower() or "emotion intensity: 5" in strong_positive.lower():
                gs += 0.5

            image = self.pipe(
                prompt_embeds=augmented_prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                width=self.width,
                height=self.height,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=gs,
                generator=generator,
            ).images[0]

            path = out_dir / f"frame_{int(frame_id):03d}_cand_{cid:02d}.png"
            image.save(path)

            candidates.append(
                CandidateImage(
                    frame_id=int(frame_id),
                    candidate_id=cid,
                    image_path=str(path),
                    prompt=strong_positive,
                    scores={
                        "image_quality": 0.0,
                        "identity_consistency": 0.0,
                        "emotion_visibility": 0.0,
                        "scene_alignment": 0.0,
                        "event_alignment": 0.0,
                        "colorfulness": 0.0,
                        "overall": 0.0,
                    },
                    notes={
                        "seed": seed,
                        "adapter_weights": packet.adapter_weights,
                        "cross_attention_adapter_tokens": int(adapter_tokens.shape[1]),
                        "negative_prompt": strong_negative,
                        "full_color_enforced": True,
                    },
                )
            )

        return candidates
