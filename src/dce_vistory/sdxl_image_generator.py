# src/dce_vistory/sdxl_image_generator.py
# Optional SDXL img2img-compatible generator wrapper.
# Use this if your current generator only accepts text prompts.
# For better consistency, replace the body of generate_frame with your IP-Adapter call if available.

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from PIL import Image


def _load_image(path: Optional[str], size: tuple[int, int]) -> Optional[Image.Image]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return Image.open(p).convert("RGB").resize(size)


class SDXLSequentialImageGenerator:
    def __init__(self, pipe: Any):
        self.pipe = pipe

    def generate_frame(self, packet: Dict[str, Any]) -> str:
        output_path = Path(packet["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        width = int(packet.get("width", 1024))
        height = int(packet.get("height", 1024))
        seed = int(packet.get("seed", 42))
        generator = torch.Generator(device=self.pipe.device).manual_seed(seed)

        prev = _load_image(packet.get("previous_frame_path"), (width, height))
        init = prev or _load_image(packet.get("initial_image_path"), (width, height))

        common_kwargs = dict(
            prompt=packet["positive_prompt"],
            negative_prompt=packet["negative_prompt"],
            width=width,
            height=height,
            guidance_scale=float(packet.get("guidance_scale", 6.0)),
            num_inference_steps=int(packet.get("num_inference_steps", 30)),
            generator=generator,
        )

        # If this is an Img2Img pipeline, use previous frame as visual continuity reference.
        # If this is a Text2Img pipeline, it will fall back to prompt-only generation.
        try:
            if init is not None:
                image = self.pipe(
                    image=init,
                    strength=float(packet.get("reference_strength", 0.45)),
                    **common_kwargs,
                ).images[0]
            else:
                image = self.pipe(**common_kwargs).images[0]
        except TypeError:
            image = self.pipe(**common_kwargs).images[0]

        image.save(output_path)
        return str(output_path)
