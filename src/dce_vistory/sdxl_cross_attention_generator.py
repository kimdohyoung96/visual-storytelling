from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import torch

from .adapters_pytorch import ButterflyAdapterStack
from .latent_schema import VisualControlPacket
from .schema import CandidateImage


def _clean_text(x: str | None) -> str:
    if not x:
        return ""
    x = str(x).replace("\n", " ")
    x = re.sub(r"\s+", " ", x).strip()
    return x


def _limit_words(x: str | None, max_words: int = 55) -> str:
    x = _clean_text(x)
    words = x.split()
    if len(words) <= max_words:
        return x
    return " ".join(words[:max_words])


def _compact_prompt_from_packet(packet: VisualControlPacket) -> str:
    """
    CLIP 77-token 제한 때문에 긴 story prompt를 그대로 넣지 않는다.
    긴 정보는 adapter branch의 control text로 보내고,
    base prompt는 이미지 생성에 가장 중요한 내용만 앞쪽에 압축한다.
    """
    meta = packet.control_metadata or {}

    character_text = _limit_words(
        meta.get("character_text", json.dumps(meta.get("character", {}), ensure_ascii=False)),
        18,
    )
    emotion_text = _limit_words(
        meta.get("emotion_text", json.dumps(meta.get("emotion", {}), ensure_ascii=False)),
        24,
    )
    world_text = _limit_words(
        meta.get("world_text", json.dumps(meta.get("world", {}), ensure_ascii=False)),
        20,
    )
    event_text = _limit_words(meta.get("event_text", ""), 24)

    prompt = (
        "full-color cinematic storybook illustration, rich natural colors, "
        "expressive face, expressive body language, clear emotional storytelling, "
        f"{emotion_text}, {event_text}, {world_text}, {character_text}, "
        "detailed background, cinematic lighting, sharp focus"
    )
    return _limit_words(prompt, 68)


def _compact_negative_prompt(packet: VisualControlPacket) -> str:
    base = (
        "monochrome, grayscale, black and white, pencil sketch, charcoal sketch, line art only, "
        "colorless image, emotionless face, weak expression, stiff pose, empty background, "
        "low quality, blurry, bad anatomy, distorted face, watermark, text"
    )
    extra = _limit_words(getattr(packet, "negative_prompt", ""), 25)
    return _limit_words(base + ", " + extra, 70)


class SDXLButterflyCrossAttentionGenerator:
    """
    Fixes:
    1. VAE dtype mismatch:
       Do NOT force self.pipe.vae.to(torch.float32) while latents are fp16.
       This avoids:
       RuntimeError: Input type Half and bias type float should be the same

    2. CLIP 77-token warning:
       Use compact base prompt. Send compact branch texts to adapter modules.
    """

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
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            use_safetensors=True,
        )

        if enable_cpu_offload and device.startswith("cuda"):
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device)

        # 중요:
        # self.pipe.vae.to(torch.float32)를 호출하지 않는다.
        # fp16 latent와 fp32 VAE bias가 섞여 decode 단계에서 터질 수 있다.

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

        if self.dtype == torch.float16:
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
        text = _limit_words(text, 70)
        prompt_embeds, _, _, _ = self._encode_prompt(text, negative_prompt="", do_cfg=False)
        return prompt_embeds

    def _control_texts_from_packet(self, packet: VisualControlPacket) -> Dict[str, str]:
        meta = packet.control_metadata or {}
        return {
            "character_adapter": _limit_words(
                meta.get("character_text", json.dumps(meta.get("character", {}), ensure_ascii=False)),
                70,
            ),
            "world_adapter": _limit_words(
                meta.get("world_text", json.dumps(meta.get("world", {}), ensure_ascii=False)),
                70,
            ),
            "emotion_adapter": _limit_words(
                meta.get("emotion_text", json.dumps(meta.get("emotion", {}), ensure_ascii=False)),
                70,
            ),
            "event_adapter": _limit_words(
                meta.get("event_text", getattr(packet, "positive_prompt", "")),
                70,
            ),
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

        compact_positive = _compact_prompt_from_packet(packet)
        compact_negative = _compact_negative_prompt(packet)

        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self._encode_prompt(
            compact_positive,
            negative_prompt=compact_negative,
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

            image = self.pipe(
                prompt_embeds=augmented_prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                width=self.width,
                height=self.height,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=generator,
            ).images[0]

            path = out_dir / f"frame_{int(frame_id):03d}_cand_{cid:02d}.png"
            image.save(path)

            candidates.append(
                CandidateImage(
                    frame_id=int(frame_id),
                    candidate_id=cid,
                    image_path=str(path),
                    prompt=compact_positive,
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
                        "negative_prompt": compact_negative,
                        "full_color_enforced": True,
                        "prompt_compacted_for_clip_77_tokens": True,
                        "control_texts": control_texts,
                    },
                )
            )

        return candidates
