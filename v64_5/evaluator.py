
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




def _contains_any(text: str, options: List[str]) -> bool:
    s = str(text or '').lower()
    return any(str(o or '').lower() in s for o in options if str(o or '').strip())


def _story_word_targets(frame: Any) -> Dict[str, List[List[str]]]:
    # each inner list is an OR-set for one required concept.
    # V52: derive targets from the raw story sentence when generated FrameVisualSpec fields
    # are absent on the storyboard frame object. This prevents false 1.0 coverage.
    def add_unique(dst, group):
        group = [str(x or '').lower().strip() for x in group if str(x or '').strip()]
        if group and group not in dst:
            dst.append(group)

    action_groups: List[List[str]] = []
    scene_groups: List[List[str]] = []
    object_groups: List[List[str]] = []
    emotion_groups: List[List[str]] = []
    color_groups: List[List[str]] = []

    story = str(getattr(frame, 'story_sentence', '') or getattr(frame, 'image_caption_en', '') or getattr(frame, 'caption', '') or '').lower()
    protagonist = str(getattr(frame, 'character_identity', '') or getattr(frame, 'protagonist', '') or story).lower()

    # fallback from story sentence
    if 'white bear' in protagonist or 'white bear' in story:
        add_unique(color_groups, ['white bear', 'polar bear', 'white fur'])
    if any(k in story for k in ['enter', 'enters']):
        add_unique(action_groups, ['enter', 'entering', 'walk', 'walking'])
    if any(k in story for k in ['search', 'searches', 'searching']):
        add_unique(action_groups, ['search', 'searching', 'look', 'looking'])
    if any(k in story for k in ['pushes', 'pushing', 'push through']):
        add_unique(action_groups, ['push', 'pushing', 'pushes'])
    if any(k in story for k in ['follow', 'follows', 'scent']):
        add_unique(action_groups, ['follow', 'following', 'scent', 'trail'])
    if any(k in story for k in ['sees', 'spots', 'seeing', 'spotting']):
        add_unique(action_groups, ['see', 'sees', 'spot', 'spots', 'looking'])
    if any(k in story for k in ['savor', 'savors', 'taste', 'tastes']):
        add_unique(action_groups, ['savor', 'taste', 'eating', 'holding'])
    if any(k in story for k in ['lick', 'licks', 'licking']):
        add_unique(action_groups, ['lick', 'licking', 'tongue'])
    if any(k in story for k in ['forest', 'foliage', 'underbrush', 'trees']):
        add_unique(scene_groups, ['forest', 'foliage', 'underbrush', 'trees', 'woods'])
    if any(k in story for k in ['lake', 'water', 'shore', "water's edge", 'water edge']):
        add_unique(scene_groups, ['lake', 'water', 'shore', 'water edge'])
    if 'honey' in story or 'jar' in story:
        add_unique(object_groups, ['honey jar', 'jar', 'honey pot'])
        if any(k in story for k in ['savor', 'savors', 'lick', 'licks', 'taste', 'tastes']):
            add_unique(object_groups, ['honey', 'golden honey'])
    for k, opts in [
        ('yearning', ['yearning', 'longing', 'determined']),
        ('determination', ['determined', 'focused']),
        ('determined', ['determined', 'focused']),
        ('frustration', ['frustrated', 'frustration', 'tense']),
        ('frustrated', ['frustrated', 'tense']),
        ('hopeful', ['hopeful', 'hope']),
        ('excited', ['excited', 'relieved', 'happy']),
        ('relieved', ['relieved', 'happy']),
        ('content', ['content', 'happy', 'smile']),
        ('fulfilled', ['fulfilled', 'content', 'happy']),
        ('happiness', ['happy', 'happiness', 'joyful', 'smile']),
        ('joy', ['joyful', 'happy', 'smile']),
    ]:
        if k in story:
            add_unique(emotion_groups, opts)

    def wrap(items):
        out=[]
        for x in items or []:
            x=str(x or '').strip().lower()
            if not x:
                continue
            if x == 'white fur':
                add_unique(out, ['white bear','polar bear','white fur'])
            elif x == 'white bear':
                add_unique(out, ['white bear','polar bear'])
            elif x == 'pushing through foliage':
                add_unique(out, ['push','pushing','foliage','bush','underbrush'])
            elif x == 'emerging from foliage':
                add_unique(out, ['emerge','emerging','foliage','forest'])
            elif x == 'arriving at the lake':
                add_unique(out, ['arrive','arriving','lake','shore'])
            elif x == 'looking at the honey jar':
                add_unique(out, ['look','looking','spot','see','jar'])
            elif x == 'searching for the honey jar':
                add_unique(out, ['search','looking','jar'])
            elif x == 'savoring honey from the jar':
                add_unique(out, ['honey','jar','eat','eating','savor','taste','lick'])
            elif x == 'smiling with joy':
                add_unique(out, ['smile','smiling','joy'])
            elif x == 'serene lake':
                add_unique(out, ['lake','water','shore'])
            elif x == 'lakeshore':
                add_unique(out, ['lake','shore','water edge'])
            elif x == 'water edge':
                add_unique(out, ['water edge','shore','edge'])
            elif x == 'dense foliage':
                add_unique(out, ['foliage','forest','leaves'])
            elif x == 'dense underbrush':
                add_unique(out, ['underbrush','bush','bushes'])
            elif x == 'honey jar':
                add_unique(out, ['jar','honey jar'])
            elif x == 'honey':
                add_unique(out, ['honey'])
            elif x == 'joyful smile':
                add_unique(out, ['smile','happy','joyful'])
            elif x == 'happy expression':
                add_unique(out, ['happy','smiling'])
            elif x == 'anxious expression':
                add_unique(out, ['anxious','worried'])
            elif x == 'worried face':
                add_unique(out, ['worried','concerned'])
            elif x == 'hopeful expression':
                add_unique(out, ['hopeful','alert'])
            elif x == 'relieved expression':
                add_unique(out, ['relieved'])
            elif x == 'determined expression':
                add_unique(out, ['determined','focused'])
            elif x == 'focused face':
                add_unique(out, ['focused','determined'])
            elif x == 'content smile':
                add_unique(out, ['content','smile'])
            else:
                toks=[t for t in re.sub(r'[^a-z0-9 ]+',' ',x).split() if len(t)>=3]
                if toks:
                    add_unique(out, toks)
        return out

    for g in wrap(getattr(frame,'story_action_keywords',[]) or []): add_unique(action_groups, g)
    for g in wrap(getattr(frame,'story_scene_keywords',[]) or []): add_unique(scene_groups, g)
    for g in wrap(getattr(frame,'story_object_keywords',[]) or []): add_unique(object_groups, g)
    for g in wrap(getattr(frame,'story_emotion_keywords',[]) or []): add_unique(emotion_groups, g)
    for g in wrap(getattr(frame,'story_color_keywords',[]) or []): add_unique(color_groups, g)

    return {
        'action': action_groups,
        'scene': scene_groups,
        'object': object_groups,
        'emotion': emotion_groups,
        'color': color_groups,
    }


def _phrase_group_hit(caption: str, token_group: List[str]) -> float:
    s = str(caption or '').lower()
    if not s or not token_group:
        return 0.0
    # OR if any full phrase directly occurs
    for token in token_group:
        if token in s:
            return 1.0
    # otherwise partial token overlap
    toks = _tokens(s)
    needed = {t for phrase in token_group for t in re.sub(r'[^a-z0-9 ]+',' ',phrase).split() if len(t) >= 3}
    if not needed:
        return 0.0
    hit = len(toks & needed) / max(1, len(needed))
    return min(1.0, hit)


def story_word_coverage_details(caption: str, frame: Any) -> Dict[str, Any]:
    targets = _story_word_targets(frame)
    out = {}
    for cat, groups in targets.items():
        if not groups:
            out[cat] = 1.0
            continue
        vals = [_phrase_group_hit(caption, g) for g in groups]
        out[cat] = round(sum(vals) / max(1, len(vals)), 4)
    out['overall'] = round((0.30*out.get('action',1.0) + 0.20*out.get('scene',1.0) + 0.20*out.get('object',1.0) + 0.15*out.get('emotion',1.0) + 0.15*out.get('color',1.0)), 4)
    return out

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


def recent_frame_repeat_penalty(image_path: str, previous_paths: List[str]) -> float:
    paths = []
    for p in previous_paths or []:
        try:
            p = str(p or '')
        except Exception:
            p = ''
        if p and Path(p).exists() and p not in paths:
            paths.append(p)
    if not paths:
        return 0.0
    vals = [frame_repeat_penalty(image_path, p) for p in paths]
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return 0.0
    return round(float(max(vals)), 4)

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
            penalty += 0.60
        if re.search(r'\bbrown bear\b|\bgrizzly\b|\bpanda\b|\bcub\b|\borange bear\b|\btan bear\b|\bgolden bear\b', cap):
            penalty += 0.52
        if 'bear' not in cap and 'white bear' not in cap and 'polar bear' not in cap:
            penalty += 0.30
    if 'jar' in sent or 'honey' in sent:
        if re.search(r'\bbook\b|\breading\b|\btruck\b|\bcar\b|\bvehicle\b|\bmouse\b|\bbird\b|\bfish\b', cap):
            penalty += 0.55
    if 'lake' in sent or 'water' in sent:
        if re.search(r'\broom\b|\bindoor\b|\blibrary\b|\bclassroom\b', cap):
            penalty += 0.25
    return min(0.95, penalty)


def _frame_state_penalty(caption: str, frame: Any) -> float:
    cap = str(caption or '').lower()
    sent = str(getattr(frame, 'story_sentence', '') or '').lower()
    penalty = 0.0
    if ('enter' in sent or 'enters' in sent) and not re.search(r'walk|walking|enter|entering|path|trail', cap):
        penalty += 0.22
    if ('search' in sent or 'searches' in sent or 'searching' in sent) and not re.search(r'search|searching|looking|looks|sniff|sniffing|scan', cap):
        penalty += 0.24
    if ('underbrush' in sent or 'bush' in sent or 'bushes' in sent) and not re.search(r'underbrush|bush|bushes|shrubs|foliage', cap):
        penalty += 0.24
    if ('towering trees' in sent or 'trees' in sent or 'forest' in sent) and not re.search(r'forest|trees|woods', cap):
        penalty += 0.16
    if ('pause' in sent or 'pauses' in sent) and re.search(r'running|jumping|sprinting', cap):
        penalty += 0.12
    if ('root' in sent or 'stumble' in sent or 'stumbles' in sent) and not re.search(r'root|roots|stumble|stumbling|trip|tripping', cap):
        penalty += 0.30
    if ('lake' in sent or 'water edge' in sent or "water's edge" in sent or 'shore' in sent or 'water' in sent) and not re.search(r'lake|water|shore|shoreline|edge', cap):
        penalty += 0.34
    if ('jar' in sent or 'honey jar' in sent or 'honey' in sent) and not re.search(r'jar|honey|pot|container', cap):
        penalty += 0.40
    if ('retriev' in sent or 'pick' in sent or 'hold' in sent) and not re.search(r'hold|holding|pick|picked|retrieve|retrieving|carry|carrying', cap):
        penalty += 0.26
    if ('savor' in sent or 'savors' in sent or 'eat' in sent or 'taste' in sent or 'lick' in sent) and not re.search(r'eat|eating|lick|licking|taste|tasting|savor|holding.*jar|honey', cap):
        penalty += 0.30
    if ('looking around' in sent or 'search' in sent or 'searches' in sent) and re.search(r'sitting|resting|posing|portrait', cap):
        penalty += 0.18
    if ('white bear' in sent or 'bear' in sent) and re.search(r'fox|red fox|brown bear|grizzly|panda|cub', cap):
        penalty += 0.42
    if ('lake' not in sent and 'water' not in sent) and re.search(r'lake|shore|shoreline|water edge', cap):
        penalty += 0.12
    return min(0.98, penalty)


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



def signature_item_coverage_score(caption: str, frame: Any) -> float:
    cap = _clean_phrase(caption)
    raw_items = list(getattr(frame, 'signature_items', []) or getattr(frame, 'must_show', []) or [])
    story = f"{getattr(frame, 'story_sentence', '')} {getattr(frame, 'image_sentence', '')} {getattr(frame, 'image_caption_en', '')}".lower()
    items = []
    for x in raw_items:
        x = str(x or '').strip().lower()
        if x and x not in items:
            items.append(x)
    if ('honey jar' in story or ('honey' in story and 'jar' in story) or 'honey' in story) and 'honey jar' not in items:
        items.append('honey jar')
    if not items:
        return 1.0
    hits = 0
    total = 0
    for item in items[:4]:
        total += 1
        variants = list(_phrase_variants(item))
        if item == 'honey jar':
            variants = list(dict.fromkeys(variants + ['honey jar', 'jar', 'honey pot', 'pot of honey', 'golden honey', 'jar of honey', 'open honey jar', 'glass honey jar', 'honey container']))
        variants = [str(v) for v in variants if str(v).strip()]
        if any(v in cap for v in variants):
            hits += 1
    return round(float(hits / max(1, total)), 4)

def emotion_face_visibility_score(caption: str, frame: Any) -> float:
    cap = _clean_phrase(caption)
    story = f"{getattr(frame, 'story_sentence', '')} {getattr(frame, 'image_sentence', '')} {getattr(frame, 'image_caption_en', '')}".lower()
    emotion = str(getattr(frame, 'emotion', '') or '').lower()
    want = []
    if any(k in story or k in emotion for k in ['anx', 'worried', 'concerned']):
        want += ['anxious', 'worried', 'concerned', 'uneasy']
    if any(k in story or k in emotion for k in ['frustrat', 'tense']):
        want += ['frustrated', 'tense', 'grimace', 'strained', 'struggling']
    if any(k in story or k in emotion for k in ['hope', 'hopeful']):
        want += ['hopeful', 'alert', 'hope', 'looking ahead', 'attentive']
    if any(k in story or k in emotion for k in ['relief', 'relieved']):
        want += ['relieved', 'relief', 'soft smile', 'relaxed', 'eased']
    if any(k in story or k in emotion for k in ['joy', 'joyful', 'happy', 'content']):
        want += ['joyful', 'happy', 'content', 'smile', 'smiling', 'delighted', 'peaceful', 'satisfied']
    if not want:
        want = ['expression', 'face', 'smile', 'worried']
    emotion_hit = 1.0 if any(w in cap for w in want) else 0.0
    face_hit = 1.0 if any(w in cap for w in ['face', 'facial', 'expression', 'eyes', 'smile', 'mouth', 'muzzle', 'brow', 'gaze']) else 0.0
    return round(float(0.65 * emotion_hit + 0.35 * face_hit), 4)

def protagonist_color_consistency_score(caption: str, frame: Any) -> float:
    cap = _clean_phrase(caption)
    story = f"{getattr(frame, 'story_sentence', '')} {getattr(frame, 'character_identity', '')} {getattr(frame, 'protagonist', '')}".lower()
    if 'white bear' in story:
        if 'white bear' in cap or ('white' in cap and 'bear' in cap):
            return 1.0
        if 'bear' in cap:
            return 0.45
        return 0.0
    return 0.6


def missing_signature_object_penalty(caption: str, frame: Any) -> float:
    cap = _clean_phrase(caption)
    story = f"{getattr(frame, 'story_sentence', '')} {getattr(frame, 'image_sentence', '')} {getattr(frame, 'image_caption_en', '')}".lower()
    if ('honey jar' in story or ('honey' in story and 'jar' in story) or 'honey' in story):
        if any(v in cap for v in ['honey jar', 'jar', 'honey pot', 'pot of honey', 'honey']):
            return 0.0
        return 0.28
    return 0.0

def severe_identity_object_penalty(caption: str, frame: Any) -> float:
    cap = _clean_phrase(caption)
    story = f"{getattr(frame, 'story_sentence', '')} {getattr(frame, 'character_identity', '')} {getattr(frame, 'protagonist', '')}".lower()
    penalty = 0.0
    if 'white bear' in story:
        if any(x in cap for x in ['fox', 'squirrel', 'raccoon', 'teddy bear', 'plush', 'small animal', 'girl', 'boy', 'woman', 'man', 'child', 'human']):
            penalty += 0.85
        elif 'bear' in cap and not ('white bear' in cap or 'polar bear' in cap or 'white' in cap):
            penalty += 0.30
        elif 'bear' not in cap and 'animal' in cap:
            penalty += 0.55
    if ('honey' in story or 'jar' in story) and not any(x in cap for x in ['honey jar', 'jar', 'honey pot', 'pot of honey', 'honey']):
        penalty += 0.30
    return round(float(min(1.0, penalty)), 4)


def protagonist_species_score(caption: str, frame: Any) -> float:
    cap = _clean_phrase(caption)
    story = f"{getattr(frame, 'story_sentence', '')} {getattr(frame, 'character_identity', '')} {getattr(frame, 'protagonist', '')}".lower()
    if 'white bear' in story or 'bear' in story:
        if 'white bear' in cap or 'polar bear' in cap or ('white' in cap and 'bear' in cap):
            return 1.0
        if 'bear' in cap:
            return 0.55
        if any(x in cap for x in ['rabbit', 'bunny', 'hare', 'raccoon', 'fox', 'squirrel', 'deer', 'wolf', 'dog', 'cat']):
            return 0.0
        return 0.12
    return 0.6


def forbidden_species_penalty(caption: str, frame: Any) -> float:
    cap = _clean_phrase(caption)
    story = f"{getattr(frame, 'story_sentence', '')} {getattr(frame, 'character_identity', '')} {getattr(frame, 'protagonist', '')}".lower()
    penalty = 0.0
    if 'white bear' in story or 'bear' in story:
        if any(x in cap for x in ['rabbit', 'bunny', 'hare', 'raccoon', 'fox', 'squirrel', 'deer', 'wolf']):
            penalty += 0.95
        if any(x in cap for x in ['brown bear', 'grizzly', 'panda', 'cub', 'orange bear', 'tan bear', 'golden bear', 'teddy bear', 'plush']):
            penalty += 0.80
        if any(x in cap for x in ['girl', 'boy', 'woman', 'man', 'child', 'human', 'person']):
            penalty += 0.55
        if 'bear' not in cap and not any(x in cap for x in ['rabbit', 'bunny', 'hare', 'raccoon', 'fox', 'squirrel', 'deer', 'wolf']):
            penalty += 0.35
    return round(float(min(1.0, penalty)), 4)


def protagonist_age_consistency_score(caption: str, frame: Any) -> float:
    cap = _clean_phrase(caption)
    anchor_age = _clean_phrase(getattr(frame, 'anchor_age_stage', '') or 'adult')
    story = f"{getattr(frame, 'story_sentence', '')} {getattr(frame, 'character_identity', '')} {getattr(frame, 'protagonist', '')}".lower()
    if 'bear' not in story and 'white bear' not in story:
        return 0.6
    if anchor_age in {'adult', 'grown', 'mature'}:
        if any(x in cap for x in ['adult', 'large', 'full grown', 'full-grown', 'grown']):
            return 1.0
        if any(x in cap for x in ['cub', 'baby', 'young', 'small bear', 'little bear']):
            return 0.0
        if 'bear' in cap or 'white bear' in cap or 'polar bear' in cap:
            return 0.78
        return 0.2
    if anchor_age in {'cub', 'baby', 'young', 'juvenile'}:
        if any(x in cap for x in ['cub', 'baby', 'young', 'small bear', 'little bear']):
            return 1.0
        if any(x in cap for x in ['adult', 'large', 'full grown', 'full-grown', 'grown']):
            return 0.0
        if 'bear' in cap:
            return 0.55
        return 0.2
    return 0.7


def protagonist_age_penalty(caption: str, frame: Any) -> float:
    cap = _clean_phrase(caption)
    anchor_age = _clean_phrase(getattr(frame, 'anchor_age_stage', '') or 'adult')
    if anchor_age in {'adult', 'grown', 'mature'} and any(x in cap for x in ['cub', 'baby', 'young', 'little bear', 'small bear']):
        return 0.85
    if anchor_age in {'cub', 'baby', 'young', 'juvenile'} and any(x in cap for x in ['adult', 'large', 'full grown', 'full-grown', 'grown']):
        return 0.80
    return 0.0


def protagonist_white_bear_exact_score(caption: str, frame: Any) -> float:
    cap = _clean_phrase(caption)
    story = f"{getattr(frame, 'story_sentence', '')} {getattr(frame, 'character_identity', '')} {getattr(frame, 'protagonist', '')}".lower()
    if 'white bear' in story:
        if 'white bear' in cap or 'polar bear' in cap or ('white' in cap and 'bear' in cap):
            return 1.0
        if 'bear' in cap and not any(x in cap for x in ['brown', 'orange', 'golden', 'tan', 'red', 'panda']):
            return 0.68
        return 0.0
    return 0.6


def sentence_complete_score(score_map: Dict[str, float]) -> float:
    parts = [
        float(score_map.get('story_word_coverage', 0.0)),
        float(score_map.get('action_word_coverage', 0.0)),
        float(score_map.get('scene_word_coverage', 0.0)),
        float(score_map.get('object_word_coverage', 0.0)),
        float(score_map.get('emotion_word_coverage', 0.0)),
        float(score_map.get('color_word_coverage', 0.0)),
        float(score_map.get('critical_noun_coverage', 0.0)),
        float(score_map.get('protagonist_color_consistency', 0.0)),
    ]
    weights = [0.18, 0.18, 0.14, 0.18, 0.10, 0.08, 0.08, 0.06]
    val = sum(w * p for w, p in zip(weights, parts))
    return round(float(max(0.0, min(1.0, val))), 4)


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
        word_cov = story_word_coverage_details(txt, frame)
        return {
            'local_caption': txt,
            'story_alignment_local': story_align,
            'event_alignment_local': event_align,
            'evidence_visibility_local': evidence,
            'scene_alignment_local': scene_align,
            'emotion_visibility_local': emotion_vis,
            'emotion_face_visibility_local': emotion_face_visibility_score(txt, frame),
            'signature_item_coverage_local': signature_item_coverage_score(txt, frame),
            'identity_consistency_local': id_match,
            'bad_extra_subject_penalty': _bad_extra_subject_penalty(txt),
            'generic_caption_penalty': generic_caption_penalty(txt),
            'scene_grounding_penalty': _scene_grounding_penalty(txt, frame),
            'story_word_coverage': float(word_cov.get('overall', 0.0)),
            'action_word_coverage': float(word_cov.get('action', 0.0)),
            'scene_word_coverage': float(word_cov.get('scene', 0.0)),
            'object_word_coverage': float(word_cov.get('object', 0.0)),
            'emotion_word_coverage': float(word_cov.get('emotion', 0.0)),
            'color_word_coverage': float(word_cov.get('color', 0.0)),
            'story_word_details': word_cov,
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
                'emotion_face_visibility': 0.10,
                'signature_item_coverage': 0.10,
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
                'frame_state_penalty': 0.0,
                'scene_grounding_penalty': 0.0,
                'static_repeat_penalty': 0.0,
                'recent_repeat_penalty': 0.0,
                'story_word_coverage': 0.0,
                'action_word_coverage': 0.0,
                'scene_word_coverage': 0.0,
                'object_word_coverage': 0.0,
                'emotion_word_coverage': 0.0,
                'color_word_coverage': 0.0,
                'story_word_penalty': 0.0,
                'protagonist_color_consistency': 0.0,
                'protagonist_white_bear_exact_score': 0.0,
                'protagonist_species_score': 0.0,
                'protagonist_age_consistency_score': 0.0,
                'protagonist_age_penalty': 0.0,
                'forbidden_species_penalty': 0.0,
                'missing_signature_object_penalty': 0.0,
                'sentence_complete_score': 0.0,
                'severe_identity_object_penalty': 0.0,
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
                scores['frame_state_penalty'] = max(scores.get('frame_state_penalty', 0.0), _frame_state_penalty(str(local_scores.get('local_caption', '')), frame))
                scores['story_word_coverage'] = max(scores.get('story_word_coverage', 0.0), float(local_scores.get('story_word_coverage', 0.0)))
                scores['action_word_coverage'] = max(scores.get('action_word_coverage', 0.0), float(local_scores.get('action_word_coverage', 0.0)))
                scores['scene_word_coverage'] = max(scores.get('scene_word_coverage', 0.0), float(local_scores.get('scene_word_coverage', 0.0)))
                scores['object_word_coverage'] = max(scores.get('object_word_coverage', 0.0), float(local_scores.get('object_word_coverage', 0.0)))
                scores['emotion_word_coverage'] = max(scores.get('emotion_word_coverage', 0.0), float(local_scores.get('emotion_word_coverage', 0.0)))
                scores['color_word_coverage'] = max(scores.get('color_word_coverage', 0.0), float(local_scores.get('color_word_coverage', 0.0)))
                local_caption_txt = str(local_scores.get('local_caption', ''))
                scores['protagonist_color_consistency'] = max(scores.get('protagonist_color_consistency', 0.0), protagonist_color_consistency_score(local_caption_txt, frame))
                scores['protagonist_white_bear_exact_score'] = max(scores.get('protagonist_white_bear_exact_score', 0.0), protagonist_white_bear_exact_score(local_caption_txt, frame))
                scores['protagonist_species_score'] = max(scores.get('protagonist_species_score', 0.0), protagonist_species_score(local_caption_txt, frame))
                scores['protagonist_age_consistency_score'] = max(scores.get('protagonist_age_consistency_score', 0.0), protagonist_age_consistency_score(local_caption_txt, frame))
                scores['protagonist_age_penalty'] = max(scores.get('protagonist_age_penalty', 0.0), protagonist_age_penalty(local_caption_txt, frame))
                scores['forbidden_species_penalty'] = max(scores.get('forbidden_species_penalty', 0.0), forbidden_species_penalty(local_caption_txt, frame))
                scores['missing_signature_object_penalty'] = max(scores.get('missing_signature_object_penalty', 0.0), missing_signature_object_penalty(local_caption_txt, frame))
                scores['severe_identity_object_penalty'] = max(scores.get('severe_identity_object_penalty', 0.0), severe_identity_object_penalty(local_caption_txt, frame))
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
            previous_two = list((getattr(c, 'notes', {}) or {}).get('previous_two_selected_image_paths', []) or [])
            repeat_pen = frame_repeat_penalty(c.image_path, prev_path)
            scores['static_repeat_penalty'] = repeat_pen
            scores['recent_repeat_penalty'] = max(scores.get('recent_repeat_penalty', 0.0), recent_frame_repeat_penalty(c.image_path, previous_two))
            current_caption = str((getattr(c, 'notes', {}) or {}).get('local_caption', '') or getattr(frame, 'story_sentence', '') or '')
            scores['protagonist_alignment'] = max(scores.get('protagonist_alignment', 0.0), _protagonist_alignment_score(current_caption, frame))
            scores['wrong_subject_or_object_penalty'] = max(scores.get('wrong_subject_or_object_penalty', 0.0), _wrong_subject_or_object_penalty(current_caption, frame))
            scores['frame_state_penalty'] = max(scores.get('frame_state_penalty', 0.0), _frame_state_penalty(current_caption, frame))
            word_details = story_word_coverage_details(str((getattr(c, 'notes', {}) or {}).get('local_caption', '') or ''), frame)
            scores['story_word_coverage'] = max(scores.get('story_word_coverage', 0.0), float(word_details.get('overall', 0.0)))
            scores['action_word_coverage'] = max(scores.get('action_word_coverage', 0.0), float(word_details.get('action', 0.0)))
            scores['scene_word_coverage'] = max(scores.get('scene_word_coverage', 0.0), float(word_details.get('scene', 0.0)))
            scores['object_word_coverage'] = max(scores.get('object_word_coverage', 0.0), float(word_details.get('object', 0.0)))
            scores['emotion_word_coverage'] = max(scores.get('emotion_word_coverage', 0.0), float(word_details.get('emotion', 0.0)))
            scores['color_word_coverage'] = max(scores.get('color_word_coverage', 0.0), float(word_details.get('color', 0.0)))
            scores['signature_item_coverage'] = max(scores.get('signature_item_coverage', 0.0), signature_item_coverage_score(current_caption, frame))
            scores['emotion_face_visibility'] = max(scores.get('emotion_face_visibility', 0.0), emotion_face_visibility_score(current_caption, frame))
            scores['story_word_penalty'] = max(scores.get('story_word_penalty', 0.0), round(1.0 - float(word_details.get('overall', 0.0)), 4))
            c.notes['story_word_details'] = word_details
            scores['critical_noun_coverage'] = max(scores['critical_noun_coverage'], critical_noun_coverage_score(current_caption, frame))
            scores['missing_critical_noun_penalty'] = max(scores['missing_critical_noun_penalty'], missing_critical_noun_penalty(current_caption, frame))
            scores['progression_consistency'] = max(scores['progression_consistency'], story_progression_consistency_score(current_caption, getattr(frame, 'story_sentence', '') or '', prev_caption, next_caption))
            scores['story_context_alignment'] = max(scores['story_context_alignment'], story_context_alignment_score(current_caption, getattr(frame, 'story_sentence', '') or '', previous_summary, previous_local_caption))
            scores['storytelling_progression'] = max(scores['storytelling_progression'], storytelling_progression_score(current_caption, prev_caption, repeat_pen), scores['progression_consistency'])
            scores['protagonist_color_consistency'] = max(scores.get('protagonist_color_consistency', 0.0), protagonist_color_consistency_score(current_caption, frame))
            scores['protagonist_white_bear_exact_score'] = max(scores.get('protagonist_white_bear_exact_score', 0.0), protagonist_white_bear_exact_score(current_caption, frame))
            scores['protagonist_species_score'] = max(scores.get('protagonist_species_score', 0.0), protagonist_species_score(current_caption, frame))
            scores['protagonist_age_consistency_score'] = max(scores.get('protagonist_age_consistency_score', 0.0), protagonist_age_consistency_score(current_caption, frame))
            scores['protagonist_age_penalty'] = max(scores.get('protagonist_age_penalty', 0.0), protagonist_age_penalty(current_caption, frame))
            scores['forbidden_species_penalty'] = max(scores.get('forbidden_species_penalty', 0.0), forbidden_species_penalty(current_caption, frame))
            scores['missing_signature_object_penalty'] = max(scores.get('missing_signature_object_penalty', 0.0), missing_signature_object_penalty(current_caption, frame))
            scores['severe_identity_object_penalty'] = max(scores.get('severe_identity_object_penalty', 0.0), severe_identity_object_penalty(current_caption, frame))
            scores['sentence_complete_score'] = max(scores.get('sentence_complete_score', 0.0), sentence_complete_score(scores))
            scores['continuity'] = max(scores['continuity'], 0.24 * scores['identity_consistency'] + 0.16 * scores.get('reference_subject_similarity', 0.0) + 0.10 * scores['scene_alignment'] + 0.09 * scores['storytelling_progression'] + 0.09 * scores['critical_noun_coverage'] + 0.09 * scores['story_context_alignment'] + 0.10 * scores.get('protagonist_color_consistency', 0.0) + 0.08 * scores.get('protagonist_age_consistency_score', 0.0) + 0.10 * scores.get('signature_item_coverage', 0.0) + 0.10 * scores.get('emotion_face_visibility', 0.0))

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
                0.04 * scores['colorfulness'] +
                0.22 * scores['identity_consistency'] +
                0.10 * scores.get('reference_subject_similarity', 0.0) +
                0.10 * scores.get('protagonist_alignment', 0.0) +
                0.23 * scores['story_alignment'] +
                0.10 * scores['event_alignment'] +
                0.11 * scores['event_grounding'] +
                0.10 * scores['evidence_visibility'] +
                0.06 * scores['emotion_visibility'] +
                0.04 * scores.get('emotion_face_visibility', 0.0) +
                0.01 * scores['emotion_cause_visibility'] +
                0.12 * scores['scene_alignment'] +
                0.08 * scores['background_presence'] +
                0.10 * scores['critical_noun_coverage'] +
                0.11 * scores.get('story_word_coverage', 0.0) +
                0.06 * scores.get('action_word_coverage', 0.0) +
                0.07 * scores.get('emotion_word_coverage', 0.0) +
                0.09 * scores.get('signature_item_coverage', 0.0) +
                0.08 * scores.get('color_word_coverage', 0.0) +
                0.14 * scores.get('protagonist_color_consistency', 0.0) +
                0.18 * scores.get('protagonist_white_bear_exact_score', 0.0) +
                0.20 * scores.get('protagonist_species_score', 0.0) +
                0.16 * scores.get('protagonist_age_consistency_score', 0.0) +
                0.15 * scores.get('sentence_complete_score', 0.0) +
                0.08 * scores['story_context_alignment'] +
                0.07 * scores['storytelling_progression'] +
                0.06 * scores['progression_consistency'] +
                0.10 * scores['continuity'] -
                0.95 * scores.get('recent_repeat_penalty', 0.0) -
                1.20 * scores.get('bad_extra_subject_penalty', 0.0) -
                1.25 * scores.get('wrong_subject_or_object_penalty', 0.0) -
                1.35 * scores.get('frame_state_penalty', 0.0) -
                1.05 * scores.get('generic_caption_penalty', 0.0) -
                1.20 * scores.get('scene_grounding_penalty', 0.0) -
                1.10 * scores.get('gray_background_penalty', 0.0) -
                1.05 * scores.get('blank_background_penalty', 0.0) -
                1.20 * scores.get('missing_critical_noun_penalty', 0.0) -
                1.70 * scores.get('missing_signature_object_penalty', 0.0) -
                1.90 * scores.get('severe_identity_object_penalty', 0.0) -
                1.85 * scores.get('protagonist_age_penalty', 0.0) -
                2.10 * scores.get('forbidden_species_penalty', 0.0) -
                1.15 * scores.get('story_word_penalty', 0.0) -
                0.90 * scores.get('crop_penalty', 0.0) -
                0.70 * scores.get('static_repeat_penalty', 0.0)
            )
            story_text = f"{getattr(frame, 'story_sentence', '')} {getattr(frame, 'image_caption_en', '')}".lower()
            frame_id = int(getattr(frame, 'frame_id', 1))
            if scores.get('protagonist_species_score', 0.0) < (0.92 if frame_id == 1 else 0.84):
                overall -= 1.40
            if scores.get('protagonist_white_bear_exact_score', 0.0) < (0.93 if frame_id == 1 else 0.86):
                overall -= 1.35
            if scores.get('protagonist_color_consistency', 0.0) < (0.95 if frame_id == 1 else 0.86):
                overall -= 1.30
            if scores.get('protagonist_age_consistency_score', 0.0) < (0.90 if frame_id == 1 else 0.82):
                overall -= 1.10
            if scores.get('protagonist_age_penalty', 0.0) > 0.10:
                overall -= 1.20
            if scores.get('reference_subject_similarity', 0.0) < 0.74 and frame_id > 1:
                overall -= 1.05
            if scores.get('identity_consistency', 0.0) < 0.78 and frame_id > 1:
                overall -= 1.05
            if scores.get('continuity', 0.0) < 0.76 and frame_id > 1:
                overall -= 0.95
            if scores.get('forbidden_species_penalty', 0.0) > 0.08:
                overall -= 1.35
            if ('honey' in story_text or 'jar' in story_text) and scores.get('object_word_coverage', 0.0) < (0.72 if frame_id == 1 else 0.62):
                overall -= 1.00
            if scores.get('sentence_complete_score', 0.0) < (0.82 if frame_id == 1 else 0.78):
                overall -= 1.00
            if ('honey' in story_text or 'jar' in story_text) and scores.get('signature_item_coverage', 0.0) < (0.90 if frame_id == 1 else 0.84):
                overall -= 1.15
            if scores.get('emotion_face_visibility', 0.0) < (0.62 if frame_id == 1 else 0.56):
                overall -= 0.85
            if is_ending:
                overall += 0.04 * scores['emotion_visibility'] + 0.04 * scores.get('emotion_face_visibility', 0.0)
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
                'word_level_story_verification_required': True,
            }
            ranked.append(c)
        return sorted(ranked, key=lambda x: x.scores.get('overall', 0.0), reverse=True)

    def rerank_ending_candidates(self, final_frame, dce_plan, candidates):
        return self.rank_frame_candidates(final_frame, dce_plan, candidates, is_ending=True)

    def evaluate_sequence(self, dce_plan, emotion_arc, storyboard, images, questions, out_dir=None) -> Dict[str, Any]:
        if not images:
            return {'warning': 'No images'}
        n = max(1, len(images))
        keys = ['image_quality', 'colorfulness', 'identity_consistency', 'reference_subject_similarity', 'protagonist_alignment', 'crop_penalty', 'story_alignment', 'story_context_alignment', 'event_alignment', 'event_grounding', 'evidence_visibility', 'emotion_visibility', 'emotion_cause_visibility', 'emotion_face_visibility', 'signature_item_coverage', 'scene_alignment', 'background_presence', 'critical_noun_coverage', 'storytelling_progression', 'progression_consistency', 'continuity', 'protagonist_color_consistency', 'protagonist_white_bear_exact_score', 'protagonist_species_score', 'protagonist_age_consistency_score', 'protagonist_age_penalty', 'forbidden_species_penalty', 'sentence_complete_score', 'scene_grounding_penalty', 'qa_score', 'overall']
        avg = {k: round(sum(float(getattr(img, 'scores', {}).get(k, 0.0)) for img in images) / n, 4) for k in keys}
        result = {'num_frames': len(images), 'averages': avg, 'selected_image_paths': [getattr(x, 'image_path', '') for x in images], 'questions': questions}
        if self.save_contact_sheet and out_dir:
            try:
                result['contact_sheet_path'] = _make_contact_sheet_local([getattr(x, 'image_path', '') for x in images], Path(out_dir) / 'contact_sheet.png')
            except Exception as e:
                result['contact_sheet_error'] = str(e)
        return result
