
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List
import json
import math
import re

import numpy as np
from PIL import Image, ImageFilter, ImageStat, ImageDraw

from .prompts import SYSTEM_NARRATIVE, SYSTEM_VLM, eval_questions_prompt
from .utils import extract_json


def _safe_asdict(obj: Any):
    if is_dataclass(obj):
        d = asdict(obj)
        d.update({k: v for k, v in getattr(obj, '__dict__', {}).items() if k not in d})
        return _safe_asdict(d)
    if isinstance(obj, dict):
        return {str(k): _safe_asdict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_asdict(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def _compact_storyboard(storyboard):
    return [{
        'frame_id': getattr(f, 'frame_id', i + 1),
        'story_sentence': getattr(f, 'story_sentence', ''),
        'event': getattr(f, 'event', ''),
        'event_grounding': getattr(f, 'event_grounding', ''),
        'evidence_objects': getattr(f, 'evidence_objects', []),
        'emotion_evidence': getattr(f, 'emotion_evidence', []),
        'emotion': getattr(f, 'emotion', ''),
        'must_show': getattr(f, 'must_show', []),
        'scene_location': getattr(f, 'scene_location', ''),
        'weather': getattr(f, 'weather', ''),
    } for i, f in enumerate(storyboard)]


def _tokens(x):
    if isinstance(x, (list, tuple)):
        x = ' '.join(str(v) for v in x)
    s = re.sub(r'[^A-Za-z0-9가-힣 ]+', ' ', str(x or '').lower())
    return {t for t in s.split() if len(t) >= 2}


def _overlap(a, b):
    A = _tokens(a); B = _tokens(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def image_quality_proxy(path: str) -> float:
    try:
        img = Image.open(path).convert('RGB')
        gray = img.convert('L').resize((256, 256))
        edges = gray.filter(ImageFilter.FIND_EDGES)
        sharpness = min(1.0, ImageStat.Stat(edges).mean[0] / 45.0)
        stat = ImageStat.Stat(gray)
        contrast = min(1.0, stat.stddev[0] / 60.0)
        brightness = stat.mean[0] / 255.0
        brightness_score = 1.0 - min(1.0, abs(brightness - 0.55) / 0.55)
        return round(float(0.45 * sharpness + 0.35 * contrast + 0.20 * brightness_score), 4)
    except Exception:
        return 0.5


def colorfulness_score(path: str) -> float:
    try:
        img = np.array(Image.open(path).convert('RGB')).astype(np.float32)
        r = img[:, :, 0]; g = img[:, :, 1]; b = img[:, :, 2]
        rg = np.abs(r - g); yb = np.abs(0.5 * (r + g) - b)
        colorfulness = math.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2) + 0.3 * math.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
        return round(float(min(1.0, colorfulness / 60.0)), 4)
    except Exception:
        return 0.3

def gray_background_penalty(path: str) -> float:
    try:
        img = np.array(Image.open(path).convert('RGB').resize((256, 256))).astype(np.float32) / 255.0
        maxc = img.max(axis=2)
        minc = img.min(axis=2)
        sat = (maxc - minc).mean()
        rgb_mean = img.mean(axis=(0, 1))
        channel_gap = float(max(abs(float(rgb_mean[0] - rgb_mean[1])), abs(float(rgb_mean[1] - rgb_mean[2])), abs(float(rgb_mean[0] - rgb_mean[2]))))
        penalty = 0.0
        if sat < 0.12:
            penalty += min(0.35, (0.12 - sat) * 3.0)
        if channel_gap < 0.035:
            penalty += 0.12
        return round(float(min(0.5, penalty)), 4)
    except Exception:
        return 0.0





def background_presence_score(path: str) -> float:
    try:
        img = np.array(Image.open(path).convert('RGB').resize((256, 256))).astype(np.float32) / 255.0
        gray = img.mean(axis=2)
        outer = np.concatenate([
            gray[:48, :].ravel(), gray[-48:, :].ravel(), gray[:, :48].ravel(), gray[:, -48:].ravel()
        ])
        sat = (img.max(axis=2) - img.min(axis=2))
        outer_sat = np.concatenate([
            sat[:48, :].ravel(), sat[-48:, :].ravel(), sat[:, :48].ravel(), sat[:, -48:].ravel()
        ])
        detail = float(np.std(outer))
        color = float(np.mean(outer_sat))
        score = min(1.0, 1.6 * detail + 1.2 * color)
        return round(float(max(0.0, score)), 4)
    except Exception:
        return 0.35


def blank_background_penalty(path: str) -> float:
    try:
        score = background_presence_score(path)
        return round(float(max(0.0, min(0.55, 0.55 - score))), 4)
    except Exception:
        return 0.0


def _global_rgb_hist(path: str) -> np.ndarray | None:
    try:
        img = Image.open(path).convert('RGB').resize((128, 128))
        arr = np.array(img).astype(np.float32)
        hists = []
        for ch in range(3):
            hist, _ = np.histogram(arr[:, :, ch], bins=16, range=(0, 255), density=False)
            hist = hist.astype(np.float32)
            hist = hist / max(1e-6, float(hist.sum()))
            hists.append(hist)
        return np.concatenate(hists)
    except Exception:
        return None


def frame_repeat_penalty(image_path: str, previous_path: str) -> float:
    if not previous_path or not Path(str(previous_path)).exists():
        return 0.0
    try:
        a = _global_rgb_hist(image_path)
        b = _global_rgb_hist(previous_path)
        if a is None or b is None:
            return 0.0
        sim = float(np.minimum(a, b).sum() / max(1e-6, b.sum()))
        if sim < 0.90:
            return 0.0
        return round(float(min(0.30, (sim - 0.90) * 2.0)), 4)
    except Exception:
        return 0.0


def storytelling_progression_score(caption: str, previous_caption: str, repeat_penalty: float) -> float:
    if not previous_caption:
        return 0.55
    overlap = _overlap(caption, previous_caption)
    score = max(0.0, 0.85 - 0.35 * overlap - 1.2 * repeat_penalty)
    return round(float(min(1.0, score)), 4)


def crop_penalty_proxy(path: str) -> float:
    """Lightweight penalty for portrait-like/cropped outputs.

    This is not a detector. It looks for high edge activity touching the border, which often
    appears when the face/body/paws are cut by the frame. It is used only as a small ranking term.
    """
    try:
        img = Image.open(path).convert('L').resize((256, 256))
        edges = np.array(img.filter(ImageFilter.FIND_EDGES)).astype(np.float32) / 255.0
        border = np.concatenate([edges[:16, :].ravel(), edges[-16:, :].ravel(), edges[:, :16].ravel(), edges[:, -16:].ravel()])
        center = edges[48:208, 48:208].ravel()
        ratio = float(border.mean() / (center.mean() + 1e-6))
        if ratio <= 0.65:
            return 0.0
        return round(float(min(0.35, (ratio - 0.65) * 0.20)), 4)
    except Exception:
        return 0.0


def _center_crop_rgb_hist(path: str) -> np.ndarray | None:
    try:
        img = Image.open(path).convert('RGB').resize((256, 256))
        w, h = img.size
        crop = img.crop((w // 4, h // 4, 3 * w // 4, 3 * h // 4))
        arr = np.array(crop).astype(np.float32)
        hists = []
        for ch in range(3):
            hist, _ = np.histogram(arr[:, :, ch], bins=16, range=(0, 255), density=False)
            hist = hist.astype(np.float32)
            hist = hist / max(1e-6, float(hist.sum()))
            hists.append(hist)
        return np.concatenate(hists)
    except Exception:
        return None


def reference_subject_similarity(image_path: str, reference_path: str) -> float:
    if not reference_path:
        return 0.0
    try:
        if not Path(str(reference_path)).exists():
            return 0.0
        a = _center_crop_rgb_hist(image_path)
        b = _center_crop_rgb_hist(reference_path)
        if a is None or b is None:
            return 0.0
        # Histogram intersection, robust enough for species/fur/color drift without needing CLIP/DINO.
        sim = float(np.minimum(a, b).sum() / max(1e-6, b.sum()))
        return round(float(max(0.0, min(1.0, sim))), 4)
    except Exception:
        return 0.0


def _reference_path_from_candidate(cand) -> str:
    try:
        return str((getattr(cand, 'notes', {}) or {}).get('subject_reference_path', '') or '')
    except Exception:
        return ''

def generic_caption_penalty(caption: str) -> float:
    cap = str(caption or '').lower()
    penalty = 0.0
    generic_terms = ['drawing of a bear', 'black and white drawing', 'illustration of a bear', 'cartoon bear', 'polar bear sitting', 'bear sitting on top', 'a bear standing']
    if any(t in cap for t in generic_terms):
        penalty += 0.25
    if 'black and white' in cap or 'grayscale' in cap:
        penalty += 0.20
    if 'sitting on top of a green circle' in cap:
        penalty += 0.35
    return min(0.6, penalty)


def _make_contact_sheet_local(image_paths: List[str], out_path: Path, cols: int = 3, thumb_size=(384, 384)) -> str:
    valid = [Path(str(p)) for p in image_paths if p and Path(str(p)).exists()]
    if not valid:
        raise ValueError('No valid image paths for contact sheet.')
    rows = (len(valid) + cols - 1) // cols
    header_h = 54
    canvas = Image.new('RGB', (cols * thumb_size[0], rows * thumb_size[1] + header_h), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), 'DCEE-CausalVerse Contact Sheet', fill=(0, 0, 0))
    for idx, p in enumerate(valid):
        img = Image.open(p).convert('RGB')
        img.thumbnail((thumb_size[0] - 20, thumb_size[1] - 44))
        x0 = (idx % cols) * thumb_size[0]
        y0 = header_h + (idx // cols) * thumb_size[1]
        draw.rectangle([x0, y0, x0 + thumb_size[0] - 1, y0 + thumb_size[1] - 1], outline=(180, 180, 180))
        draw.text((x0 + 10, y0 + 10), f'Frame {idx + 1}', fill=(0, 0, 0))
        canvas.paste(img, (x0 + (thumb_size[0] - img.width) // 2, y0 + 34 + (thumb_size[1] - 44 - img.height) // 2))
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True); canvas.save(out_path)
    return str(out_path)


def _frame_tifa_questions(frame: Any) -> List[str]:
    return [
        f"Does the image match the exact story sentence `{getattr(frame, 'story_sentence', '')}`?",
        f"Is the main visible action `{getattr(frame, 'event', '')}` clearly shown?",
        f"Can you see evidence objects {getattr(frame, 'evidence_objects', [])} or equivalent visual clues?",
        f"Can you understand the cause `{getattr(frame, 'event_grounding', '')}` from the image?",
        f"Is the protagonist emotion `{getattr(frame, 'emotion', '')}` clearly visible in face/body/scene?",
        f"Do the background/weather/location match `{getattr(frame, 'scene_location', '')}` / `{getattr(frame, 'weather', '')}`?",
        'Does the protagonist remain the same person as in previous frames?',
    ]


def _scene_grounding_penalty(caption: str, frame: Any) -> float:
    cap = str(caption or '').lower()
    sent = str(getattr(frame, 'story_sentence', '') or '').lower()
    penalty = 0.0
    scene_terms = []
    for key in ['forest', 'woods', 'tree', 'branch', 'bush', 'root', 'roots', 'slope', 'hill', 'lake', 'water', 'shore', 'jar', 'honey']:
        if key in sent:
            scene_terms.append(key)
    if scene_terms:
        hits = sum(1 for key in set(scene_terms) if key in cap)
        if hits == 0:
            penalty += 0.45
        elif hits < max(1, len(set(scene_terms)) // 2):
            penalty += 0.20
    if re.search(r'plain background|blank background|isolated|sticker|icon|mascot', cap):
        penalty += 0.18
    return min(0.75, penalty)


def _wrong_subject_or_object_penalty(caption: str, frame: Any) -> float:
    cap = str(caption or '').lower()
    sent = str(getattr(frame, 'story_sentence', '') or '').lower()
    penalty = 0.0
    # protagonist should remain a white bear
    if 'white bear' in sent or 'bear' in sent:
        if re.search(r'\bfox\b|\bred fox\b|\bsquirrel\b|\bdeer\b|\brabbit\b|\bwolf\b', cap):
            penalty += 0.55
        if re.search(r'\bbrown bear\b|\bgrizzly\b|\bpanda\b', cap):
            penalty += 0.35
        if 'bear' not in cap and 'white bear' not in cap and 'polar bear' not in cap:
            penalty += 0.28
    if 'jar' in sent or 'honey' in sent:
        if re.search(r'\bbook\b|\breading\b|\btruck\b|\bcar\b|\bvehicle\b', cap):
            penalty += 0.45
    if 'lake' in sent or 'water' in sent:
        if re.search(r'\broom\b|\bindoor\b|\blibrary\b|\bclassroom\b', cap):
            penalty += 0.25
    return min(0.95, penalty)


def _protagonist_alignment_score(caption: str, frame: Any) -> float:
    cap = str(caption or '').lower()
    sent = str(getattr(frame, 'story_sentence', '') or '').lower()
    if 'white bear' in sent or 'bear' in sent:
        if 'white bear' in cap or 'polar bear' in cap:
            return 0.95
        if 'bear' in cap:
            return 0.70
        return 0.10
    return 0.50


def _bad_extra_subject_penalty(caption: str) -> float:
    cap = str(caption or '').lower()
    penalty = 0.0
    if re.search(r'\bperson\b|\bman\b|\bwoman\b|\bchild\b|\bboy\b|\bgirl\b|\bhuman\b|\bpeople\b|\bcrowd\b', cap):
        penalty += 0.50
    if re.search(r'\btwo\b|\bthree\b|\bgroup\b|\bcrowd\b|\bmultiple\b|\bseveral\b|\bpair\b', cap):
        penalty += 0.30
    if re.search(r'next to each other|standing next to each other|with another|and a bear|two bears|pair of bears|two polar bears|two pandas', cap):
        penalty += 0.48
    if re.search(r'\bportrait\b|\bclose[- ]?up\b|sticker|icon|mascot', cap):
        penalty += 0.22
    if re.search(r'cropped|cut off|head only|face only', cap):
        penalty += 0.25
    return min(0.95, penalty)

def _required_object_coverage(caption: str, required: Any) -> float:
    cap = str(caption or '').lower()
    objs = [str(x).lower() for x in (required or []) if str(x).strip()]
    keep = []
    for o in objs:
        if o in {'protagonist', 'white bear', 'bear', 'panda'}:
            continue
        keep.append(o)
    if not keep:
        return 0.50
    hits = 0
    for obj in keep[:6]:
        alts = {obj}
        if 'honey' in obj or 'jar' in obj:
            alts |= {'honey', 'jar'}
        if 'riverbank' in obj or obj == 'river':
            alts |= {'river', 'water', 'shore', 'bank', 'stream'}
        if 'lake' in obj:
            alts |= {'lake', 'water', 'shore'}
        if 'forest' in obj or 'tree' in obj or 'underbrush' in obj:
            alts |= {'forest', 'trees', 'woods', 'bush', 'underbrush'}
        if any(a in cap for a in alts):
            hits += 1
    return hits / max(1, min(len(keep), 4))


def _clean_phrase(text: Any) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'[^A-Za-z0-9가-힣 ]+', ' ', str(text or '').lower())).strip()


def _unique_list(items: List[str], limit: int | None = None) -> List[str]:
    out = []
    for item in items:
        s = _clean_phrase(item)
        if s and s not in out:
            out.append(s)
    return out if limit is None else out[:limit]


def _critical_visual_nouns_from_frame(frame: Any) -> List[str]:
    text = ' '.join([
        str(getattr(frame, 'story_sentence', '') or ''),
        str(getattr(frame, 'image_caption_en', '') or ''),
        str(getattr(frame, 'scene_location', '') or ''),
        ' '.join(str(x) for x in (getattr(frame, 'must_show', []) or [])),
        ' '.join(str(x) for x in (getattr(frame, 'environment_details', []) or [])),
    ]).lower()
    out: List[str] = []
    rules = [
        (['lost jar', 'honey jar', 'jar'], 'lost jar'),
        (['honey'], 'honey'),
        (['tangled roots', 'tree roots', 'roots', 'root'], 'tangled roots'),
        (['steep slope', 'slope', 'incline', 'hill', 'hillside'], 'steep slope'),
        (['serene lake', 'lake', 'lakeshore', 'shore', 'shoreline'], 'serene lake'),
        (['dense forest', 'forest', 'woods', 'trees', 'underbrush'], 'dense forest'),
        (['trail', 'path'], 'forest path'),
    ]
    for keys, label in rules:
        if any(k in text for k in keys):
            out.append(label)
    for item in list(getattr(frame, 'must_show', []) or []) + [getattr(frame, 'scene_location', '')] + list(getattr(frame, 'environment_details', []) or []):
        s = _clean_phrase(item)
        if not s:
            continue
        if s in {'protagonist', 'bear', 'white bear', 'panda', 'subject'}:
            continue
        out.append(s)
    return _unique_list(out, 8)


def _phrase_variants(phrase: str) -> set[str]:
    p = _clean_phrase(phrase)
    alts = {p}
    if 'jar' in p:
        alts |= {'jar', 'lost jar', 'honey jar', 'honey pot', 'pot', 'container'}
    if 'honey' in p:
        alts |= {'honey'}
    if 'root' in p:
        alts |= {'root', 'roots', 'tree root', 'tree roots', 'tangled roots'}
    if 'slope' in p or 'hill' in p or 'incline' in p:
        alts |= {'slope', 'steep slope', 'hill', 'hillside', 'incline'}
    if 'lake' in p or 'shore' in p:
        alts |= {'lake', 'lakeshore', 'shore', 'shoreline', 'water', 'pond'}
    if 'forest' in p or 'woods' in p or 'tree' in p or 'underbrush' in p:
        alts |= {'forest', 'dense forest', 'woods', 'trees', 'underbrush'}
    if 'path' in p or 'trail' in p:
        alts |= {'path', 'trail'}
    return {x for x in alts if x}


def critical_noun_coverage_score(caption: str, frame: Any) -> float:
    cap = _clean_phrase(caption)
    nouns = _critical_visual_nouns_from_frame(frame)
    if not nouns:
        return _required_object_coverage(caption, getattr(frame, 'must_show', []))
    hits = 0
    for noun in nouns[:6]:
        if any(v in cap for v in _phrase_variants(noun)):
            hits += 1
    return round(float(hits / max(1, min(len(nouns), 6))), 4)


def missing_critical_noun_penalty(caption: str, frame: Any) -> float:
    cov = critical_noun_coverage_score(caption, frame)
    nouns = _critical_visual_nouns_from_frame(frame)
    if not nouns:
        return 0.0
    penalty = (1.0 - cov) * min(0.75, 0.18 * min(len(nouns), 4))
    return round(float(max(0.0, penalty)), 4)


def story_progression_consistency_score(current_caption: str, target_caption: str, previous_caption: str, next_caption: str) -> float:
    curr = _overlap(current_caption, target_caption)
    prev = _overlap(current_caption, previous_caption) if previous_caption else 0.0
    nxt = _overlap(current_caption, next_caption) if next_caption else 0.0
    score = 0.30 + 0.70 * curr
    if previous_caption:
        score += 0.10 if curr >= prev else -0.20 * (prev - curr)
    if next_caption:
        score += 0.10 if curr >= nxt else -0.18 * (nxt - curr)
    return round(float(max(0.0, min(1.0, score))), 4)


def story_context_alignment_score(current_caption: str, target_caption: str, previous_summary: str, previous_local_caption: str) -> float:
    curr = _overlap(current_caption, target_caption)
    prev_story = _overlap(current_caption, previous_summary) if previous_summary else 0.0
    prev_local = _overlap(current_caption, previous_local_caption) if previous_local_caption else 0.0
    score = 0.40 + 0.60 * curr
    if previous_summary:
        score += 0.08 if curr >= prev_story else -0.10 * (prev_story - curr)
    if previous_local_caption:
        score += 0.08 if curr >= prev_local else -0.10 * (prev_local - curr)
    return round(float(max(0.0, min(1.0, score))), 4)


def _protagonist_identity_match(frame: Any, caption: str) -> float:
    cap = str(caption or '').lower()
    protagonist = f"{getattr(frame, 'character_identity', '')} {getattr(frame, 'story_sentence', '')}".lower()
    score = 0.0
    if 'white bear' in protagonist:
        if 'white bear' in cap:
            score += 1.0
        elif 'bear' in cap:
            score += 0.7
        if 'white' in cap:
            score += 0.2
    elif 'panda' in protagonist:
        if 'panda' in cap:
            score += 1.0
        elif 'bear' in cap:
            score += 0.5
    else:
        key = str(getattr(frame, 'character_identity', '') or getattr(frame, 'protagonist', '') or '').lower().strip()
        if key and key in cap:
            score += 1.0
    return min(1.0, score if score > 0 else 0.35)


def _caption_alignment(frame: Any, caption: str) -> float:
    return max(
        _overlap(caption, getattr(frame, 'image_caption_en', '')),
        _overlap(caption, getattr(frame, 'image_sentence', '')),
        _overlap(caption, getattr(frame, 'story_sentence', '')),
        _overlap(caption, getattr(frame, 'caption', '')),
        0.55 * _required_object_coverage(caption, getattr(frame, 'must_show', []))
    )


def _scene_alignment(frame: Any, caption: str) -> float:
    return max(
        _overlap(caption, getattr(frame, 'scene_location', '')),
        _overlap(caption, getattr(frame, 'weather', '')),
        _required_object_coverage(caption, getattr(frame, 'environment_details', [])),
    )


class DCEQAEvaluator:
    def __init__(self, llm, vlm, use_vlm: bool = True, save_contact_sheet: bool = True, use_local_caption_scorer: bool = True):
        self.llm = llm
        self.vlm = vlm
        self.use_vlm = use_vlm
        self.save_contact_sheet = save_contact_sheet
        self.use_local_caption_scorer = use_local_caption_scorer
        self.caption_processor = None
        self.caption_model = None
        self.caption_device = 'cpu'
        if use_local_caption_scorer:
            try:
                import torch
                from transformers import BlipProcessor, BlipForConditionalGeneration
                self.caption_processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
                self.caption_model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
                self.caption_device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.caption_model.to(self.caption_device)
                self.caption_model.eval()
            except Exception:
                self.caption_processor = None
                self.caption_model = None

    def generate_questions(self, dce_plan, emotion_arc, storyboard) -> Dict[str, Any]:
        out = {
            'global_questions': [
                'Does the generated sequence follow the planned DCEE event chain?',
                'Is each frame grounded in its story sentence rather than a generic portrait?',
                'Are event, evidence, emotional cause, and target emotion visible?',
                'Are character identity and world state consistent across frames?'
            ],
            'frame_questions': {str(getattr(f, 'frame_id', i + 1)): _frame_tifa_questions(f) for i, f in enumerate(storyboard)},
            'ending_questions': [
                f"Does the final image reach the target ending emotion `{getattr(dce_plan, 'target_ending_emotion', '')}`?",
                'Does the final frame feel like a story ending rather than just another intermediate scene?'
            ]
        }
        try:
            llm_extra = extract_json(self.llm.generate(SYSTEM_NARRATIVE, eval_questions_prompt(asdict(dce_plan), asdict(emotion_arc), _compact_storyboard(storyboard)), temperature=0.0, max_tokens=800))
            if isinstance(llm_extra, dict):
                out['llm_generated_questions'] = llm_extra
        except Exception:
            pass
        return out

    def _generate_local_caption(self, image_path: str) -> str:
        if self.caption_model is None or self.caption_processor is None:
            return ''
        try:
            import torch
            img = Image.open(image_path).convert('RGB')
            inputs = self.caption_processor(images=img, return_tensors='pt')
            inputs = {k: v.to(self.caption_device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.caption_model.generate(**inputs, max_new_tokens=35, num_beams=3)
            txt = self.caption_processor.decode(out[0], skip_special_tokens=True)
            return str(txt or '').strip()
        except Exception:
            return ''

    def _local_caption_eval(self, frame, cand) -> Dict[str, Any]:
        txt = self._generate_local_caption(cand.image_path)
        if not txt:
            return {}
        story_align = _caption_alignment(frame, txt)
        event_align = max(_overlap(txt, getattr(frame, 'event', '')), _overlap(txt, getattr(frame, 'event_grounding', '')))
        evidence = max(_overlap(txt, getattr(frame, 'evidence_objects', [])), _overlap(txt, getattr(frame, 'must_show', [])), _required_object_coverage(txt, getattr(frame, 'must_show', [])))
        scene_align = _scene_alignment(frame, txt)
        emotion_vis = max(_overlap(txt, getattr(frame, 'emotion', '')), _overlap(txt, getattr(frame, 'emotion_evidence', [])))
        id_match = _protagonist_identity_match(frame, txt)
        return {
            'local_caption': txt,
            'story_alignment_local': story_align,
            'event_alignment_local': event_align,
            'evidence_visibility_local': evidence,
            'scene_alignment_local': scene_align,
            'emotion_visibility_local': emotion_vis,
            'identity_consistency_local': id_match,
            'bad_extra_subject_penalty': _bad_extra_subject_penalty(txt),
            'generic_caption_penalty': generic_caption_penalty(txt),
            'scene_grounding_penalty': _scene_grounding_penalty(txt, frame),
        }

    def _vlm_frame_eval(self, frame, cand) -> Dict[str, Any]:
        if not self.use_vlm:
            return {}
        prompt = f"""
You are evaluating one visual storytelling frame.
Judge the image only.
Exact frame caption: {getattr(frame, 'image_caption_en', '') or getattr(frame, 'image_sentence', '') or getattr(frame, 'story_sentence', '')}
Planned story sentence: {getattr(frame, 'story_sentence', '')}
Planned event: {getattr(frame, 'event', '')}
Event grounding: {getattr(frame, 'event_grounding', '')}
Evidence objects: {getattr(frame, 'evidence_objects', []) or getattr(frame, 'must_show', [])}
Emotion evidence: {getattr(frame, 'emotion_evidence', [])}
Target emotion: {getattr(frame, 'emotion', '')}
Location/weather: {getattr(frame, 'scene_location', '')}, {getattr(frame, 'weather', '')}, {getattr(frame, 'atmosphere', '')}
Reject unrelated humans, extra animals, duplicate protagonist, generic portraits, and images that ignore the caption.
Return JSON only with keys:
answers, qa_score, story_alignment, event_alignment, event_grounding, evidence_visibility,
emotion_visibility, emotion_cause_visibility, scene_alignment, continuity,
identity_consistency, colorfulness, extra_subject, duplicate_protagonist, caption_mismatch, reason.
All scores must be 0 to 1.
""".strip()
        try:
            return extract_json(self.vlm.generate_with_images(SYSTEM_VLM, prompt, [cand.image_path], temperature=0.0, max_tokens=700))
        except Exception as e:
            return {'vlm_error': str(e)[:300]}

    def rank_frame_candidates(self, frame, dce_plan, candidates, is_ending: bool = False):
        ranked = []
        for c in candidates:
            scores = {
                'image_quality': image_quality_proxy(c.image_path),
                'colorfulness': colorfulness_score(c.image_path),
                'identity_consistency': 0.30,
                'reference_subject_similarity': 0.0,
                'subject_visibility': 0.0,
                'crop_penalty': crop_penalty_proxy(c.image_path),
                'story_alignment': 0.10,
                'event_alignment': 0.10,
                'event_grounding': 0.10,
                'evidence_visibility': 0.10,
                'emotion_visibility': 0.10,
                'emotion_cause_visibility': 0.10,
                'scene_alignment': 0.10,
                'continuity': 0.20,
                'qa_score': 0.20,
                'bad_extra_subject_penalty': 0.0,
                'generic_caption_penalty': 0.0,
                'gray_background_penalty': gray_background_penalty(c.image_path),
                'background_presence': background_presence_score(c.image_path),
                'blank_background_penalty': blank_background_penalty(c.image_path),
                'storytelling_progression': 0.45,
                'progression_consistency': 0.45,
                'critical_noun_coverage': 0.20,
                'missing_critical_noun_penalty': 0.0,
                'story_context_alignment': 0.45,
                'protagonist_alignment': 0.20,
                'wrong_subject_or_object_penalty': 0.0,
                'scene_grounding_penalty': 0.0,
                'static_repeat_penalty': 0.0,
            }

            local_scores = self._local_caption_eval(frame, c)
            if local_scores:
                c.notes.update({k: v for k, v in local_scores.items() if 'error' in k or k == 'local_caption'})
                scores['story_alignment'] = max(scores['story_alignment'], float(local_scores.get('story_alignment_local', 0.0)))
                scores['event_alignment'] = max(scores['event_alignment'], float(local_scores.get('event_alignment_local', 0.0)))
                scores['event_grounding'] = max(scores['event_grounding'], float(local_scores.get('event_alignment_local', 0.0)), 0.75 * float(local_scores.get('evidence_visibility_local', 0.0)))
                scores['evidence_visibility'] = max(scores['evidence_visibility'], float(local_scores.get('evidence_visibility_local', 0.0)))
                scores['emotion_visibility'] = max(scores['emotion_visibility'], float(local_scores.get('emotion_visibility_local', 0.0)))
                scores['scene_alignment'] = max(scores['scene_alignment'], float(local_scores.get('scene_alignment_local', 0.0)))
                scores['identity_consistency'] = max(scores['identity_consistency'], float(local_scores.get('identity_consistency_local', 0.0)))
                scores['bad_extra_subject_penalty'] = max(scores.get('bad_extra_subject_penalty', 0.0), float(local_scores.get('bad_extra_subject_penalty', 0.0)))
                scores['generic_caption_penalty'] = max(scores.get('generic_caption_penalty', 0.0), float(local_scores.get('generic_caption_penalty', 0.0)))
                scores['scene_grounding_penalty'] = max(scores.get('scene_grounding_penalty', 0.0), float(local_scores.get('scene_grounding_penalty', 0.0)))
                scores['protagonist_alignment'] = max(scores.get('protagonist_alignment', 0.0), _protagonist_alignment_score(str(local_scores.get('local_caption', '')), frame))
                scores['wrong_subject_or_object_penalty'] = max(scores.get('wrong_subject_or_object_penalty', 0.0), _wrong_subject_or_object_penalty(str(local_scores.get('local_caption', '')), frame))
                # continuity proxy: if caption includes both protagonist and background/object cues, reward
                scores['continuity'] = max(scores['continuity'], 0.5 * scores['identity_consistency'] + 0.5 * scores['scene_alignment'])

            ref_sim = reference_subject_similarity(c.image_path, _reference_path_from_candidate(c))
            if ref_sim > 0.0:
                scores['reference_subject_similarity'] = ref_sim
                # Reward reference similarity, but do not let it hide caption/story failures.
                scores['identity_consistency'] = max(scores['identity_consistency'], 0.45 + 0.45 * ref_sim)

            prev_path = str((getattr(c, 'notes', {}) or {}).get('previous_frame_image_path', '') or '')
            prev_caption = str((getattr(c, 'notes', {}) or {}).get('previous_frame_caption', '') or '')
            next_caption = str((getattr(c, 'notes', {}) or {}).get('next_frame_caption', '') or '')
            previous_summary = str((getattr(c, 'notes', {}) or {}).get('previous_story_summary', '') or '')
            previous_local_caption = str((getattr(c, 'notes', {}) or {}).get('previous_frame_local_caption', '') or '')
            repeat_pen = frame_repeat_penalty(c.image_path, prev_path)
            scores['static_repeat_penalty'] = repeat_pen
            current_caption = str((getattr(c, 'notes', {}) or {}).get('local_caption', '') or getattr(frame, 'story_sentence', '') or '')
            scores['protagonist_alignment'] = max(scores.get('protagonist_alignment', 0.0), _protagonist_alignment_score(current_caption, frame))
            scores['wrong_subject_or_object_penalty'] = max(scores.get('wrong_subject_or_object_penalty', 0.0), _wrong_subject_or_object_penalty(current_caption, frame))
            scores['critical_noun_coverage'] = max(scores['critical_noun_coverage'], critical_noun_coverage_score(current_caption, frame))
            scores['missing_critical_noun_penalty'] = max(scores['missing_critical_noun_penalty'], missing_critical_noun_penalty(current_caption, frame))
            scores['progression_consistency'] = max(scores['progression_consistency'], story_progression_consistency_score(current_caption, getattr(frame, 'story_sentence', '') or '', prev_caption, next_caption))
            scores['story_context_alignment'] = max(scores['story_context_alignment'], story_context_alignment_score(current_caption, getattr(frame, 'story_sentence', '') or '', previous_summary, previous_local_caption))
            scores['storytelling_progression'] = max(scores['storytelling_progression'], storytelling_progression_score(current_caption, prev_caption, repeat_pen), scores['progression_consistency'])
            scores['continuity'] = max(scores['continuity'], 0.36 * scores['identity_consistency'] + 0.18 * scores['scene_alignment'] + 0.16 * scores['storytelling_progression'] + 0.15 * scores['critical_noun_coverage'] + 0.15 * scores['story_context_alignment'])

            vlm_scores = self._vlm_frame_eval(frame, c)
            if vlm_scores and 'vlm_error' not in vlm_scores:
                for key in list(scores.keys()):
                    if key in vlm_scores:
                        try:
                            scores[key] = max(scores[key], float(vlm_scores[key]))
                        except Exception:
                            pass
                if bool(vlm_scores.get('extra_subject', False)) or bool(vlm_scores.get('duplicate_protagonist', False)):
                    scores['bad_extra_subject_penalty'] = max(scores.get('bad_extra_subject_penalty', 0.0), 0.75)
                    scores['identity_consistency'] = min(scores.get('identity_consistency', 0.35), 0.20)
                    scores['story_alignment'] = min(scores.get('story_alignment', 0.20), 0.25)
                if bool(vlm_scores.get('caption_mismatch', False)):
                    scores['story_alignment'] = min(scores.get('story_alignment', 0.20), 0.25)
                c.notes['tifa_answers'] = vlm_scores.get('answers', [])
                c.notes['vlm_reason'] = vlm_scores.get('reason', '')
            elif vlm_scores and 'vlm_error' in vlm_scores:
                c.notes['vlm_error'] = vlm_scores['vlm_error']

            overall = (
                0.03 * scores['image_quality'] +
                0.05 * scores['colorfulness'] +
                0.18 * scores['identity_consistency'] +
                0.05 * scores.get('reference_subject_similarity', 0.0) +
                0.10 * scores.get('protagonist_alignment', 0.0) +
                0.23 * scores['story_alignment'] +
                0.10 * scores['event_alignment'] +
                0.11 * scores['event_grounding'] +
                0.10 * scores['evidence_visibility'] +
                0.05 * scores['emotion_visibility'] +
                0.01 * scores['emotion_cause_visibility'] +
                0.12 * scores['scene_alignment'] +
                0.08 * scores['background_presence'] +
                0.09 * scores['critical_noun_coverage'] +
                0.07 * scores['story_context_alignment'] +
                0.07 * scores['storytelling_progression'] +
                0.06 * scores['progression_consistency'] +
                0.07 * scores['continuity'] -
                1.20 * scores.get('bad_extra_subject_penalty', 0.0) -
                1.10 * scores.get('wrong_subject_or_object_penalty', 0.0) -
                1.05 * scores.get('generic_caption_penalty', 0.0) -
                1.15 * scores.get('scene_grounding_penalty', 0.0) -
                1.10 * scores.get('gray_background_penalty', 0.0) -
                1.05 * scores.get('blank_background_penalty', 0.0) -
                1.20 * scores.get('missing_critical_noun_penalty', 0.0) -
                0.90 * scores.get('crop_penalty', 0.0) -
                0.70 * scores.get('static_repeat_penalty', 0.0)
            )
            if is_ending:
                overall += 0.03 * scores['emotion_visibility']
            scores['overall'] = round(float(overall), 4)
            c.scores.update(scores)
            c.notes['v36_selection_reason'] = {
                'caption_is_primary_contract': True,
                'identity_lock_is_primary_contract': True,
                'scene_contract_is_primary_contract': True,
                'background_and_scene_must_match_caption': True,
                'identity_and_object_grounding': True,
                'story_event_evidence_first': True,
                'single_protagonist_priority': True,
                'visual_storytelling_progression_required': True,
                'nonempty_background_required': True,
                'critical_visual_nouns_required': True,
                'previous_and_next_frame_progression_required': True,
                'previous_story_and_previous_frame_context_required': True,
            }
            ranked.append(c)
        return sorted(ranked, key=lambda x: x.scores.get('overall', 0.0), reverse=True)

    def rerank_ending_candidates(self, final_frame, dce_plan, candidates):
        return self.rank_frame_candidates(final_frame, dce_plan, candidates, is_ending=True)

    def evaluate_sequence(self, dce_plan, emotion_arc, storyboard, images, questions, out_dir=None) -> Dict[str, Any]:
        if not images:
            return {'warning': 'No images'}
        n = max(1, len(images))
        keys = ['image_quality', 'colorfulness', 'identity_consistency', 'reference_subject_similarity', 'protagonist_alignment', 'crop_penalty', 'story_alignment', 'story_context_alignment', 'event_alignment', 'event_grounding', 'evidence_visibility', 'emotion_visibility', 'emotion_cause_visibility', 'scene_alignment', 'background_presence', 'critical_noun_coverage', 'storytelling_progression', 'progression_consistency', 'continuity', 'scene_grounding_penalty', 'qa_score', 'overall']
        avg = {k: round(sum(float(getattr(img, 'scores', {}).get(k, 0.0)) for img in images) / n, 4) for k in keys}
        result = {'num_frames': len(images), 'averages': avg, 'selected_image_paths': [getattr(x, 'image_path', '') for x in images], 'questions': questions}
        if self.save_contact_sheet and out_dir:
            try:
                result['contact_sheet_path'] = _make_contact_sheet_local([getattr(x, 'image_path', '') for x in images], Path(out_dir) / 'contact_sheet.png')
            except Exception as e:
                result['contact_sheet_error'] = str(e)
        return result
