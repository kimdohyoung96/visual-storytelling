import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import yaml
from PIL import Image, ImageDraw

_ENV_PATTERN = re.compile(r"\$\{([^:}]+):?([^}]*)\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def repl(match):
            key, default = match.group(1), match.group(2)
            return os.environ.get(key, default)
        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return _expand_env(yaml.safe_load(f))


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def extract_json(text: str) -> Any:
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?', '', text).strip()
        text = re.sub(r'```$', '', text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    starts = [i for i, ch in enumerate(text) if ch in '{[']
    for start in starts:
        for end in range(len(text), start, -1):
            chunk = text[start:end]
            try:
                return json.loads(chunk)
            except json.JSONDecodeError:
                continue
    raise ValueError(f'Could not parse JSON from response:\n{text[:1200]}')


def image_to_data_url(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    mime = 'image/png'
    if suffix in ['.jpg', '.jpeg']:
        mime = 'image/jpeg'
    elif suffix == '.webp':
        mime = 'image/webp'
    data = base64.b64encode(path.read_bytes()).decode('utf-8')
    return f'data:{mime};base64,{data}'


def make_contact_sheet(image_paths: List[str | Path], out_path: str | Path, cols: int = 3, thumb_size=(384, 384)) -> str:
    image_paths = [Path(p) for p in image_paths if p and Path(p).exists()]
    if not image_paths:
        raise ValueError('No images found for contact sheet.')
    rows = (len(image_paths) + cols - 1) // cols
    canvas = Image.new('RGB', (cols * thumb_size[0], rows * thumb_size[1]), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for idx, path in enumerate(image_paths):
        img = Image.open(path).convert('RGB')
        img.thumbnail(thumb_size)
        x = (idx % cols) * thumb_size[0]
        y = (idx // cols) * thumb_size[1]
        canvas.paste(img, (x, y))
        draw.text((x + 10, y + 10), f'Frame {idx+1}', fill=(0, 0, 0))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return str(out_path)
