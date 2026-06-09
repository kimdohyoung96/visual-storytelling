from pathlib import Path
from typing import Any, Dict, Optional

from .llm import BaseLLM
from .prompts import SYSTEM_NARRATIVE, image_understanding_prompt
from .schema import ImageUnderstanding
from .utils import extract_json


class ImageUnderstandingModule:
    def __init__(self, provider: str, llm: BaseLLM, caption_model: str = 'Salesforce/blip-image-captioning-base'):
        self.provider = provider
        self.llm = llm
        self.caption_model = caption_model
        self._local_captioner = None

    def analyze(self, image_path: str | None, sample: Dict[str, Any]) -> Optional[ImageUnderstanding]:
        if not image_path:
            return None
        image_path = str(image_path)
        if not Path(image_path).exists():
            raise FileNotFoundError(f'Input image not found: {image_path}')
        if self.provider == 'local_caption':
            return self._local_caption(image_path)
        text = self.llm.generate(SYSTEM_NARRATIVE, image_understanding_prompt(image_path, sample), temperature=0.2, max_tokens=1200)
        data = extract_json(text)
        return ImageUnderstanding(caption=data.get('caption', ''), characters=data.get('characters', []), setting=data.get('setting', ''), objects=data.get('objects', []), mood=data.get('mood', ''), inferred_plot_hint=data.get('inferred_plot_hint', ''))

    def _local_caption(self, image_path: str) -> ImageUnderstanding:
        try:
            from PIL import Image
            from transformers import pipeline
            if self._local_captioner is None:
                self._local_captioner = pipeline('image-to-text', model=self.caption_model)
            result = self._local_captioner(Image.open(image_path).convert('RGB'))
            caption = result[0]['generated_text'] if result else 'No caption'
        except Exception:
            caption = 'An image with characters and setting relevant to the story.'
        return ImageUnderstanding(caption=caption, characters=[], setting='', objects=[], mood='', inferred_plot_hint='')
