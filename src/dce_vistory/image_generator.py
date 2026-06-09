from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw

from .prompts import QUALITY_SUFFIX, NEGATIVE_PROMPT
from .schema import CandidateImage


class BaseImageGenerator:
    def generate(self, prompt: str, frame_id: int, out_dir: Path, num_candidates: int = 1) -> List[CandidateImage]:
        raise NotImplementedError


class PlaceholderImageGenerator(BaseImageGenerator):
    def __init__(self, width: int = 1024, height: int = 1024):
        self.width = width
        self.height = height

    def generate(self, prompt: str, frame_id: int, out_dir: Path, num_candidates: int = 1) -> List[CandidateImage]:
        out_dir.mkdir(parents=True, exist_ok=True)
        candidates = []
        for cid in range(num_candidates):
            img = Image.new("RGB", (self.width, self.height), color=(246, 246, 246))
            draw = ImageDraw.Draw(img)
            draw.text((40, 40), f"DCE-ViStory Frame {frame_id} Candidate {cid}", fill=(0, 0, 0))
            draw.text((40, 100), self._wrap(prompt, width=75)[:1800], fill=(0, 0, 0))
            path = out_dir / f"frame_{frame_id:03d}_cand_{cid:02d}.png"
            img.save(path)
            candidates.append(CandidateImage(frame_id=frame_id, candidate_id=cid, image_path=str(path), prompt=prompt, scores={"image_quality": 0.5, "identity_consistency": 0.5, "emotion_alignment": 0.5, "overall": 0.5}, notes={"generator": "placeholder"}))
        return candidates

    @staticmethod
    def _wrap(text: str, width: int = 80) -> str:
        words = text.split()
        lines, cur = [], []
        for w in words:
            if sum(len(x) + 1 for x in cur) + len(w) > width:
                lines.append(" ".join(cur))
                cur = [w]
            else:
                cur.append(w)
        if cur:
            lines.append(" ".join(cur))
        return "\n".join(lines)


class SDXLImageGenerator(BaseImageGenerator):
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: str = "cuda",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 40,
        guidance_scale: float = 7.0,
        negative_prompt: str = NEGATIVE_PROMPT,
        quality_suffix: str = QUALITY_SUFFIX,
        seed: int = 42,
        enable_cpu_offload: bool = False,
    ):
        import torch
        from diffusers import StableDiffusionXLPipeline

        self.torch = torch
        self.device = device
        self.width = width
        self.height = height
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.negative_prompt = negative_prompt
        self.quality_suffix = quality_suffix
        self.seed = seed

        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=dtype, use_safetensors=True)

        if enable_cpu_offload and device.startswith("cuda"):
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device)

        try:
            self.pipe.enable_vae_slicing()
        except Exception:
            pass

    def _compose_prompt(self, prompt: str) -> str:
        prompt = prompt.strip()
        if self.quality_suffix.lower() not in prompt.lower():
            prompt = prompt + "\n\n" + self.quality_suffix
        return prompt

    def generate(self, prompt: str, frame_id: int, out_dir: Path, num_candidates: int = 1) -> List[CandidateImage]:
        out_dir.mkdir(parents=True, exist_ok=True)
        candidates = []
        full_prompt = self._compose_prompt(prompt)

        for cid in range(num_candidates):
            seed = self.seed + frame_id * 1000 + cid
            generator = self.torch.Generator(device=self.device).manual_seed(seed) if self.device.startswith("cuda") else self.torch.Generator().manual_seed(seed)

            image = self.pipe(
                prompt=full_prompt,
                negative_prompt=self.negative_prompt,
                width=self.width,
                height=self.height,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=generator,
            ).images[0]

            path = out_dir / f"frame_{frame_id:03d}_cand_{cid:02d}.png"
            image.save(path)
            candidates.append(CandidateImage(frame_id=frame_id, candidate_id=cid, image_path=str(path), prompt=full_prompt, scores={"image_quality": 0.0, "identity_consistency": 0.0, "emotion_alignment": 0.0, "overall": 0.0}, notes={"seed": seed, "negative_prompt": self.negative_prompt}))
        return candidates


def build_image_generator(cfg: Dict[str, Any]) -> BaseImageGenerator:
    provider = cfg.get("provider", "placeholder")

    if provider == "placeholder":
        return PlaceholderImageGenerator(width=int(cfg.get("width", 1024)), height=int(cfg.get("height", 1024)))

    if provider == "sdxl":
        return SDXLImageGenerator(
            model_id=cfg.get("model_id", "stabilityai/stable-diffusion-xl-base-1.0"),
            device=cfg.get("device", "cuda"),
            width=int(cfg.get("width", 1024)),
            height=int(cfg.get("height", 1024)),
            num_inference_steps=int(cfg.get("num_inference_steps", 40)),
            guidance_scale=float(cfg.get("guidance_scale", 7.0)),
            negative_prompt=cfg.get("negative_prompt", NEGATIVE_PROMPT),
            quality_suffix=cfg.get("quality_suffix", QUALITY_SUFFIX),
            seed=int(cfg.get("seed", 42)),
            enable_cpu_offload=bool(cfg.get("enable_cpu_offload", False)),
        )

    raise ValueError(f"Unknown image generator provider: {provider}")
