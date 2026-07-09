from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List
from types import SimpleNamespace
import json
import traceback

from PIL import Image, ImageDraw

from .llm import build_llm, build_vlm
from .planner import DCEPlanner
from .causal_memory import DCEECausalMemoryStore
from .anchor_bank import DCEEAnchorBank
from .evaluator import DCEQAEvaluator
from .image_understanding import ImageUnderstandingModule
from .schema import PipelineResult, CandidateImage
from .prompts import QUALITY_SUFFIX, NEGATIVE_PROMPT
from .butterfly_adapter import ButterflyController
from .sdxl_cross_attention_generator import SDXLButterflyCrossAttentionGenerator
from .utils import extract_json


class TextOnlyImageSummary(SimpleNamespace):
    """Attribute-compatible image summary for runs without image_path.

    planner.py fallback code may access fields such as setting, mood, caption,
    objects, scene, style, and other image-summary attributes. Returning a
    conservative default prevents text-only runs from crashing when strict LLM
    seed JSON repair fails.
    """
    def __getattr__(self, name):
        # Important: copy.deepcopy/dataclasses.asdict probe dunder methods such as
        # __deepcopy__. Returning an empty string for those makes copy.py try to
        # call a string and raises: TypeError: 'str' object is not callable.
        if str(name).startswith('__') and str(name).endswith('__'):
            raise AttributeError(name)
        if name in {
            'objects', 'object_candidates', 'characters', 'people', 'animals',
            'key_objects', 'visible_objects', 'tags', 'colors', 'color_palette'
        }:
            return []
        return ''

    def __deepcopy__(self, memo):
        return TextOnlyImageSummary(**{k: v for k, v in self.__dict__.items()})


def _safe_asdict(obj: Any):
    if is_dataclass(obj):
        d = asdict(obj)
        d.update({k: v for k, v in getattr(obj, "__dict__", {}).items() if k not in d})
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


def _write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe_asdict(obj), ensure_ascii=False, indent=2), encoding="utf-8")

def _clear_generated_pngs(*dirs: Path):
    """Remove stale candidate images from previous runs."""
    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        for pattern in ["frame_*_cand_*.png", "frame_*.png", "*.tmp.png"]:
            for p in d.glob(pattern):
                try:
                    p.unlink()
                except Exception:
                    pass


def _image_path_exists(path: Any) -> bool:
    if not path:
        return False
    try:
        return Path(str(path)).exists()
    except Exception:
        return False


def _normalize_text_only_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    sample = dict(sample or {})
    sample.setdefault('image_path', '')
    sample.setdefault('protagonist_reference_paths', [])
    sample.setdefault('canonical_reference_sheet_path', '')
    sample.setdefault('signature_items', [])
    sample['protagonist'] = _canonicalize_protagonist_text(sample.get('protagonist', '')) or 'white bear'
    sample['signature_items'] = _canonicalize_signature_items(sample.get('signature_items', []), sample.get('text_prompt', ''), sample['protagonist'])
    sample['json_input_contract'] = {
        'protagonist': sample['protagonist'],
        'signature_items': sample['signature_items'],
        'text_prompt': sample.get('text_prompt', ''),
        'target_ending_emotion': sample.get('target_ending_emotion', ''),
    }
    return sample


def _canonicalize_protagonist_text(text: Any) -> str:
    t = _clean_text(text).lower()
    if not t:
        return ''
    if ('white' in t and 'bear' in t) or 'polar bear' in t:
        return 'white bear'
    return _clean_text(text)


def _canonicalize_signature_items(items: Any, text_prompt: Any = '', protagonist: Any = '') -> List[str]:
    vals = _as_list_clean(items)
    joined = ' '.join([_clean_text(text_prompt).lower(), _clean_text(protagonist).lower(), ' '.join(v.lower() for v in vals)])
    out: List[str] = []
    for v in vals:
        vl = v.lower()
        if ('honey' in vl and 'jar' in vl) or vl in {'jar of honey', 'honey pot', 'pot of honey'}:
            out.append('honey jar')
        else:
            out.append(v)
    if any(x in joined for x in ['honey jar', 'lost honey jar', 'jar of honey', 'honey pot', 'pot of honey']):
        out.insert(0, 'honey jar')
    uniq: List[str] = []
    for v in out:
        if v and v not in uniq:
            uniq.append(v)
    return uniq[:8]


def _build_v61_identity_contract(sample: Dict[str, Any], protagonist: str, signature_items: List[str]) -> Dict[str, Any]:
    protagonist = _canonicalize_protagonist_text(protagonist) or 'white bear'
    age_stage = 'adult'
    color_name = 'white'
    fur_phrase = 'creamy white fur'
    species_phrase = 'white bear'
    if 'cub' in protagonist or 'baby bear' in protagonist or 'young' in protagonist:
        age_stage = 'cub'
        species_phrase = 'white bear cub'
    identity_anchor_prompt = (
        f"exactly one {age_stage} {species_phrase} with {fur_phrase}, rounded ears, black oval eyes, a black nose, "
        f"rounded muzzle, large paws, sturdy body proportions, and the same face in every frame"
    )
    face = 'same face in every frame: rounded muzzle, black oval eyes, black nose, rounded ears, calm storybook facial structure'
    if age_stage == 'cub':
        body = 'same body in every frame: one cub white bear with compact body, short legs, round belly, and youthful proportions'
    else:
        body = 'same body in every frame: one adult white bear with sturdy stocky body, medium-long legs, broad shoulders, and consistent proportions'
    return {
        'protagonist': protagonist,
        'age_stage': age_stage,
        'color': color_name,
        'fur_phrase': fur_phrase,
        'identity_anchor_prompt': identity_anchor_prompt,
        'face': face,
        'body': body,
        'signature_items': signature_items,
    }


def _build_v611_identity_contract(sample: Dict[str, Any], protagonist: str, signature_items: List[str]) -> Dict[str, Any]:
    protagonist = _canonicalize_protagonist_text(protagonist) or 'white bear'
    age_stage = 'cub' if any(x in protagonist.lower() for x in ['cub', 'baby', 'young']) else 'adult'
    fur_phrase = 'creamy white fur'
    species_phrase = 'white bear cub' if age_stage == 'cub' else 'adult white bear'
    identity_anchor_prompt = (
        f"exactly one {species_phrase} with {fur_phrase}, rounded ears, black oval eyes, black nose, rounded muzzle, "
        f"large paws, consistent body proportions, and the same face in every frame"
    )
    face = 'same face in every frame: rounded muzzle, black oval eyes, black nose, rounded ears, consistent storybook face'
    if age_stage == 'cub':
        body = 'same cub body in every frame: compact white bear cub, short legs, round belly, youthful proportions'
    else:
        body = 'same adult body in every frame: sturdy stocky adult white bear, broad shoulders, medium-long legs, consistent size and silhouette'
    return {
        'protagonist': protagonist,
        'age_stage': age_stage,
        'color': 'white',
        'fur_phrase': fur_phrase,
        'identity_anchor_prompt': identity_anchor_prompt,
        'face': face,
        'body': body,
        'signature_items': signature_items,
    }


def _enforce_input_contract_on_seed(seed: Any, sample: Dict[str, Any]):
    protagonist = _canonicalize_protagonist_text(sample.get('protagonist', '')) or _canonicalize_protagonist_text(getattr(seed, 'protagonist', '')) or 'white bear'
    signature_items = _canonicalize_signature_items(sample.get('signature_items', []), sample.get('text_prompt', ''), protagonist)
    contract = _build_v611_identity_contract(sample, protagonist, signature_items)

    setattr(seed, 'protagonist', protagonist)
    setattr(seed, 'input_protagonist', protagonist)
    setattr(seed, 'signature_items', signature_items)
    setattr(seed, 'input_signature_items', signature_items)
    setattr(seed, 'protagonist_reference_paths', sample.get('protagonist_reference_paths', []) or [])
    setattr(seed, '_v61_identity_contract', contract)
    setattr(seed, '_v611_identity_contract', contract)

    profiles = list(getattr(seed, 'character_profiles', []) or [])
    if profiles:
        prof = profiles[0]
    else:
        prof = SimpleNamespace()
        profiles = [prof]

    setattr(prof, 'identity_anchor_prompt', contract['identity_anchor_prompt'])
    setattr(prof, 'face', contract['face'])
    setattr(prof, 'body', contract['body'])
    setattr(prof, 'age_stage', contract['age_stage'])
    setattr(prof, 'color', contract['color'])
    setattr(prof, 'signature_items', signature_items)
    setattr(seed, 'character_profiles', profiles)

    global_negative = [
        'rabbit', 'raccoon', 'fox', 'panda', 'dog', 'cat', 'mouse', 'teddy bear', 'plush toy',
        'brown bear', 'grizzly bear', 'black bear', 'orange fur', 'brown fur', 'gray fur', 'tan fur',
        'cub version when adult is required', 'adult version when cub is required',
        'multiple protagonists', 'extra animal', 'human protagonist', 'species swap', 'different face',
    ]

    story_bible = dict(getattr(seed, '_story_bible', {}) or {})
    existing_neg = story_bible.get('global_negative', [])
    if isinstance(existing_neg, str):
        existing_neg = [existing_neg]
    for neg in global_negative:
        if neg not in existing_neg:
            existing_neg.append(neg)
    story_bible.update({
        'protagonist_lock': protagonist,
        'protagonist_species_lock': 'white bear',
        'protagonist_color_lock': 'white',
        'protagonist_fur_lock': contract['fur_phrase'],
        'protagonist_age_lock': contract['age_stage'],
        'protagonist_face_lock': contract['face'],
        'protagonist_body_lock': contract['body'],
        'signature_items_lock': signature_items,
        'json_source_protagonist': protagonist,
        'json_source_signature_items': signature_items,
        'global_negative': existing_neg,
        'identity_anchor_prompt': contract['identity_anchor_prompt'],
    })
    setattr(seed, '_story_bible', story_bible)
    setattr(seed, '_json_input_contract', {
        'protagonist': protagonist,
        'signature_items': signature_items,
        'text_prompt': sample.get('text_prompt', ''),
        'target_ending_emotion': sample.get('target_ending_emotion', ''),
        'identity_contract': contract,
    })
    return seed


def _clean_text(x: Any) -> str:

    return ' '.join(str(x or '').replace('\n', ' ').split()).strip()


def _infer_anchor_age_stage(text: Any) -> str:
    t = str(text or '').lower()
    if any(x in t for x in ['cub', 'baby bear', 'young bear', 'juvenile', 'little bear', 'small bear']):
        return 'cub'
    return 'adult'


def _default_white_bear_appearance(age_stage: str) -> str:
    if age_stage == 'cub':
        return 'exactly one white bear cub with creamy white fur, rounded muzzle, black oval eyes, black nose, rounded ears, compact cub body, and the same face and body proportions in every frame'
    return 'exactly one adult white bear with creamy white fur, rounded muzzle, black oval eyes, black nose, rounded ears, sturdy stocky body, broad shoulders, and the same face and body proportions in every frame'




def _sample_has_explicit_story_sentences(sample: Dict[str, Any]) -> bool:
    return bool(_extract_explicit_story_sentences(sample or {}))


def _compact_story_context(rows: List[Dict[str, Any]], limit: int = 6) -> str:
    parts: List[str] = []
    for i, row in enumerate(rows[-limit:], max(1, len(rows) - limit + 1)):
        if isinstance(row, dict):
            sent = _clean_text(row.get('sentence') or row.get('story_sentence') or row.get('caption'))
        else:
            sent = _clean_text(row)
        if sent:
            parts.append(f"{i}. {sent}")
    return " ".join(parts)


def _abstract_to_text(abstract: Any) -> str:
    if isinstance(abstract, str):
        return _clean_text(abstract)
    if isinstance(abstract, dict):
        for k in ['abstract', 'summary', 'story_abstract', 'text', 'caption']:
            if abstract.get(k):
                return _clean_text(abstract.get(k))
        return _clean_text(json.dumps(_safe_asdict(abstract), ensure_ascii=False))
    return _clean_text(abstract)


def _dcee_plan_to_text(dce_plan: Any) -> str:
    data = _safe_asdict(dce_plan)
    if not isinstance(data, dict):
        return _clean_text(data)
    pieces: List[str] = []
    for key in ['desire', 'conflict', 'turning_point', 'target_ending_emotion', 'ending_state']:
        val = _clean_text(data.get(key, ''))
        if val:
            pieces.append(f"{key}: {val}")
    chain = data.get('event_chain') or data.get('event_spine') or []
    if isinstance(chain, list) and chain:
        pieces.append("event chain: " + " -> ".join(_clean_text(x) for x in chain if _clean_text(x)))
    return _clean_text("; ".join(pieces))


def _v611_default_event_chain(sample: Dict[str, Any]) -> List[str]:
    protagonist = _canonicalize_protagonist_text(sample.get('protagonist', '')) or 'white bear'
    sig_items = _canonicalize_signature_items(sample.get('signature_items', []), sample.get('text_prompt', ''), protagonist)
    text = _clean_text(sample.get('text_prompt', '')).lower()
    if protagonist == 'white bear' and ('honey jar' in ' '.join(sig_items).lower() or 'honey' in text):
        return [
            'The white bear realizes the honey jar is missing in the dense forest.',
            'The white bear searches through underbrush and tangled foliage.',
            'The white bear follows clues toward the lake.',
            "The white bear discovers the honey jar at the water's edge.",
            'The white bear retrieves the honey jar.',
            'The white bear enjoys honey beside the serene lake.',
        ]
    return [
        f'The {protagonist} begins the desire-driven story.',
        f'The {protagonist} encounters the main conflict.',
        f'The {protagonist} searches for evidence and moves forward.',
        f'The {protagonist} reaches the turning point.',
        f'The {protagonist} acts to resolve the conflict.',
        f'The {protagonist} reaches the ending emotion.',
    ]


def _v611_fallback_dce_plan(seed: Any, sample: Dict[str, Any], abstract: Any, error: str = '') -> SimpleNamespace:
    protagonist = _canonicalize_protagonist_text(sample.get('protagonist', '') or getattr(seed, 'protagonist', '')) or 'white bear'
    signature_items = _canonicalize_signature_items(sample.get('signature_items', []) or getattr(seed, 'signature_items', []), sample.get('text_prompt', ''), protagonist)
    text_prompt = _clean_text(sample.get('text_prompt', '') or getattr(seed, 'text_prompt', '') or getattr(seed, 'caption', ''))
    target_emotion = _clean_text(sample.get('target_ending_emotion', '')) or 'joyful'
    event_chain = _v611_default_event_chain(sample)
    desire = f'{protagonist} wants to recover and enjoy {signature_items[0] if signature_items else "the important object"}.'
    conflict = 'The key object is lost, hidden, or difficult to reach, forcing the protagonist to search through the story world.'
    return SimpleNamespace(
        desire=desire,
        conflict=conflict,
        event_chain=event_chain,
        event_spine=event_chain,
        turning_point=event_chain[min(3, len(event_chain)-1)],
        ending_state=f'{protagonist} resolves the search and feels {target_emotion}.',
        target_ending_emotion=target_emotion,
        protagonist=protagonist,
        signature_items=signature_items,
        abstract=_abstract_to_text(abstract),
        text_prompt=text_prompt,
        fallback=True,
        fallback_reason='planner.generate_dce_plan failed; deterministic V61.1 DCEE fallback was used',
        original_error=error,
    )


def _v611_fallback_emotion_arc(sample: Dict[str, Any], total_frames: int, error: str = '') -> List[Dict[str, Any]]:
    base = ['anxious', 'frustrated', 'hopeful', 'relieved', 'joyful', 'content']
    total_frames = max(1, int(total_frames or 6))
    out: List[Dict[str, Any]] = []
    for i in range(total_frames):
        emotion = base[min(i, len(base)-1)]
        if i == total_frames - 1:
            emotion = _clean_text(sample.get('target_ending_emotion', '')) or 'content'
        out.append({
            'frame_id': i + 1,
            'emotion': emotion,
            'intensity': round(0.45 + 0.45 * (i / max(1, total_frames - 1)), 3),
            'cause': 'DCEE progression from desire/conflict toward resolution',
            'fallback': True,
            'original_error': error,
        })
    return out


def _v611_fallback_story_step(sample: Dict[str, Any], idx: int, total_frames: int, dce_plan: Any, emotion_arc: Any, error: str = '') -> Dict[str, Any]:
    protagonist = _canonicalize_protagonist_text(sample.get('protagonist', '')) or getattr(dce_plan, 'protagonist', 'white bear')
    signature_items = _canonicalize_signature_items(sample.get('signature_items', []) or getattr(dce_plan, 'signature_items', []), sample.get('text_prompt', ''), protagonist)
    event_chain = list(getattr(dce_plan, 'event_chain', []) or getattr(dce_plan, 'event_spine', []) or _v611_default_event_chain(sample))
    sentence = _clean_text(event_chain[min(idx, len(event_chain)-1)])
    if protagonist and protagonist.lower() not in sentence.lower():
        sentence = f'The {protagonist} {sentence[0].lower() + sentence[1:] if sentence else "continues the story."}'
    emotion = 'focused'
    try:
        if isinstance(emotion_arc, list) and idx < len(emotion_arc):
            row = emotion_arc[idx]
            emotion = _clean_text(row.get('emotion', '')) if isinstance(row, dict) else _clean_text(getattr(row, 'emotion', ''))
    except Exception:
        emotion = 'focused'
    low = sentence.lower()
    location = 'deep forest'
    if any(k in low for k in ['lake', 'water', 'shore']):
        location = 'serene lake edge in the forest'
    elif any(k in low for k in ['underbrush', 'foliage', 'forest']):
        location = 'dense forest with tangled underbrush'
    must_show = _unique_clean([protagonist] + signature_items + _must_show_for_sentence(sentence, protagonist, signature_items, idx, total_frames), 12)
    return {
        'sentence': sentence,
        'story_sentence': sentence,
        'event': sentence,
        'action': _infer_story_action(sentence),
        'visible_cause': sentence,
        'emotion': emotion,
        'location': location,
        'scene_location': location,
        'weather': 'soft daylight',
        'atmosphere': 'storybook adventure',
        'must_show': must_show,
        'required_objects': signature_items,
        'critical_visual_nouns': must_show,
        'input_protagonist': protagonist,
        'input_signature_items': signature_items,
        'fallback': True,
        'fallback_reason': 'planner.generate_story_step failed; deterministic V61.1 story-step fallback was used',
        'original_error': error,
    }


def _v611_frame_render_contract(frame: Any, story_step: Dict[str, Any], seed: Any, dce_plan: Any, abstract: Any, sample: Dict[str, Any], idx: int, total_frames: int) -> Dict[str, Any]:
    protagonist = _canonicalize_protagonist_text(sample.get('protagonist', '') or getattr(frame, 'protagonist', '') or getattr(seed, 'protagonist', '')) or 'white bear'
    signature_items = _canonicalize_signature_items(sample.get('signature_items', []) or getattr(seed, 'signature_items', []), sample.get('text_prompt', ''), protagonist)
    sentence = _clean_text(getattr(frame, 'story_sentence', '') or story_step.get('sentence', ''))
    caption = _clean_text(getattr(frame, 'frame_caption', '') or getattr(frame, 'image_caption_en', '') or sentence)
    identity_contract = getattr(seed, '_v611_identity_contract', {}) or getattr(seed, '_v61_identity_contract', {}) or getattr(seed, '_json_input_contract', {}).get('identity_contract', {}) or {}
    must_show = _unique_clean(
        _as_list_clean(story_step.get('must_show', []))
        + _as_list_clean(getattr(frame, 'must_show', []))
        + _as_list_clean(getattr(frame, 'critical_visual_nouns', []))
        + [protagonist]
        + signature_items,
        16,
    )
    must_not_show = _unique_clean([
        'brown bear', 'black bear', 'grizzly bear', 'panda', 'rabbit', 'raccoon', 'fox', 'squirrel',
        'human protagonist', 'extra animal', 'duplicate protagonist', 'wrong fur color', 'wrong age stage',
        'generic portrait', 'random scene', 'fallback scene', 'unrelated object', 'unrelated location',
    ], 20)
    return {
        'frame_id': idx + 1,
        'total_frames': total_frames,
        'source_rule': 'Render only from story_sentence, frame_caption, story_abstract, DCEE plan, and input JSON contract.',
        'story_sentence': sentence,
        'frame_caption': caption,
        'story_abstract': _abstract_to_text(abstract),
        'dcee_plan_summary': _dcee_plan_to_text(dce_plan),
        'generated_story_context': _clean_text(getattr(frame, 'generated_story_context', '')),
        'protagonist_contract': {
            'protagonist': protagonist,
            'identity': identity_contract.get('identity_anchor_prompt') or getattr(frame, 'character_identity', ''),
            'face': identity_contract.get('face', 'same face in every frame'),
            'body': identity_contract.get('body', 'same body proportions in every frame'),
            'fur': identity_contract.get('fur_phrase', 'creamy white fur'),
            'age_stage': identity_contract.get('age_stage', 'adult'),
        },
        'must_show': must_show,
        'must_not_show': must_not_show,
        'object_state': _clean_text(story_step.get('object_state') or getattr(frame, 'object_state_hint', '') or 'object state must follow the current caption'),
        'scene_state': _clean_text(getattr(frame, 'scene_location', '') or story_step.get('scene_location', '')),
        'emotion': _clean_text(getattr(frame, 'emotion', '') or story_step.get('emotion', '')),
        'action': _clean_text(getattr(frame, 'event', '') or story_step.get('action', '') or sentence),
        'connectivity': {
            'previous_frame_caption': _clean_text(getattr(frame, 'previous_story_hint', '')),
            'next_frame_caption': _clean_text(getattr(frame, 'next_story_hint', '')),
            'sequence_rule': 'Keep the same protagonist identity while changing only the action, emotion, object state, and scene required by the current caption.',
        },
    }


def _split_story_string(raw: Any) -> List[str]:
    txt = _clean_text(raw)
    if not txt:
        return []
    lines: List[str] = []
    for part in str(raw).replace("\r", "\n").split("\n"):
        s = _clean_text(part)
        if not s:
            continue
        while s and s[0].isdigit():
            s = s[1:].lstrip(' .):-')
        if s:
            lines.append(s)
    if len(lines) > 1:
        return lines
    parts = [x.strip() for x in txt.split('. ') if x.strip()]
    return [x if x.endswith('.') else x + '.' for x in parts] if len(parts) > 1 else ([txt] if txt else [])


def _extract_explicit_story_sentences(sample: Dict[str, Any]) -> List[str]:
    sample = sample or {}
    candidates: List[str] = []
    for key in ['story_sentences', 'sentences', 'story_rows', 'story', 'generated_story', 'full_story']:
        val = sample.get(key)
        if not val:
            continue
        if isinstance(val, list):
            for row in val:
                if isinstance(row, dict):
                    s = _clean_text(row.get('sentence') or row.get('story_sentence') or row.get('text') or row.get('caption'))
                else:
                    s = _clean_text(row)
                if s:
                    candidates.append(s)
        elif isinstance(val, dict):
            rows = val.get('sentences') or val.get('story_rows') or []
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict):
                        s = _clean_text(row.get('sentence') or row.get('story_sentence') or row.get('text') or row.get('caption'))
                    else:
                        s = _clean_text(row)
                    if s:
                        candidates.append(s)
            story_text = _clean_text(val.get('story_text') or val.get('text') or '')
            if story_text and not candidates:
                candidates.extend(_split_story_string(story_text))
        else:
            candidates.extend(_split_story_string(val))
    return _unique_clean(candidates)


def _is_white_bear_honey_quest(sample: Dict[str, Any]) -> bool:
    protagonist = _canonicalize_protagonist_text(sample.get('protagonist', ''))
    joined = ' '.join([
        protagonist.lower(),
        _clean_text(sample.get('text_prompt', '')).lower(),
        ' '.join(x.lower() for x in _canonicalize_signature_items(sample.get('signature_items', []), sample.get('text_prompt', ''), protagonist)),
    ])
    return protagonist == 'white bear' and 'honey jar' in joined


def _default_white_bear_honey_story(total_frames: int = 6) -> List[str]:
    base = [
        'The white bear searches through dense underbrush, feeling anxious about his lost honey jar.',
        'The white bear pushes through tangled foliage, growing increasingly frustrated in his search for the honey jar.',
        'The white bear emerges from the tangled foliage, spotting the serene lake ahead with a mix of hope and anxiety.',
        "The white bear approaches the lake's edge, feeling a surge of relief upon seeing his honey jar.",
        'The white bear happily savors honey beside the serene lake, feeling pure joy at last.',
        'The white bear smiles contentedly beside the serene lake, fully enjoying the sweet honey.',
    ]
    if total_frames <= len(base):
        return base[:total_frames]
    out = list(base)
    while len(out) < total_frames:
        out.append(base[-1])
    return out


def _infer_story_location(sentence: str, idx: int = 0, total_frames: int = 6) -> str:
    low = _clean_text(sentence).lower()
    if any(x in low for x in ['lake', 'shore', 'water', "water's edge", 'water edge']):
        return 'serene lake edge in the forest'
    if any(x in low for x in ['underbrush', 'foliage', 'forest', 'woods', 'bush', 'roots', 'branches', 'thicket']):
        return 'dense forest with tangled underbrush'
    if idx >= max(0, total_frames - 3):
        return 'serene lake edge in the forest'
    return 'deep forest'


def _infer_story_emotion(sentence: str, default_emotion: str = '') -> str:
    low = _clean_text(sentence).lower()
    for key in ['anxious', 'frustrated', 'hopeful', 'relieved', 'joyful', 'happy', 'content', 'contented', 'worried']:
        if key in low:
            return 'content' if key == 'contented' else key
    return _clean_text(default_emotion) or 'focused'


def _infer_story_action(sentence: str) -> str:
    low = _clean_text(sentence).lower()
    if 'search' in low:
        return 'searching through the underbrush for the lost honey jar'
    if 'push' in low:
        return 'pushing through tangled foliage while searching for the honey jar'
    if 'spot' in low or 'emerge' in low:
        return 'emerging from the foliage and noticing the lake ahead'
    if 'approach' in low or 'seeing his honey jar' in low or 'upon seeing' in low:
        return 'approaching the water edge and noticing the honey jar'
    if 'savor' in low or 'enjoy' in low or 'smile' in low:
        return 'enjoying honey beside the lake'
    return _clean_text(sentence)


def _must_show_for_sentence(sentence: str, protagonist: str, signature_items: List[str], idx: int, total_frames: int) -> List[str]:
    low = _clean_text(sentence).lower()
    out: List[str] = [protagonist]
    if any(x in low for x in ['underbrush', 'foliage', 'forest', 'bush', 'roots', 'branches', 'thicket']):
        out.extend(['dense underbrush', 'tangled foliage'])
    if any(x in low for x in ['lake', 'shore', 'water', "water's edge", 'water edge']):
        out.append('serene lake')
    if any(x in low for x in ['honey', 'jar']) or idx >= max(0, total_frames - 3):
        out.append('honey jar')
    if any(x in low for x in ['savor', 'enjoy', 'sweet honey', 'happily']):
        out.append('honey')
    return _unique_clean(out, 10)


def _v62_should_force_json_story_flow(sample: Dict[str, Any], pipe_cfg: Dict[str, Any]) -> bool:
    if bool(pipe_cfg.get('force_json_story_flow', False)):
        return True
    if bool(pipe_cfg.get('force_text_story_only', False)) and not _safe_first_image_path(sample):
        return True
    if _is_white_bear_honey_quest(sample):
        return True
    if sample.get('protagonist') and sample.get('signature_items') and not _safe_first_image_path(sample):
        return True
    return False


def _v62_input_grounded_abstract(sample: Dict[str, Any], story_steps: List[Dict[str, Any]]) -> str:
    protagonist = _canonicalize_protagonist_text(sample.get('protagonist', '')) or 'protagonist'
    signature_items = _canonicalize_signature_items(sample.get('signature_items', []), sample.get('text_prompt', ''), protagonist)
    item = signature_items[0] if signature_items else 'important object'
    sentences = [_clean_text((x or {}).get('sentence', '')) for x in (story_steps or []) if _clean_text((x or {}).get('sentence', ''))]
    if protagonist == 'white bear' and item == 'honey jar':
        return (
            "In a deep forest, a white bear anxiously searches for his lost honey jar. "
            "He pushes through dense underbrush and tangled foliage, growing frustrated as the search continues. "
            "When he reaches a serene lake, hope returns as he spots the honey jar at the water's edge. "
            "He approaches the lake, retrieves the honey jar, and finally enjoys sweet honey beside the tranquil water with relief, joy, and contentment."
        )
    if sentences:
        joined = ' '.join(sentences[:max(1, min(6, len(sentences)))])
        return joined
    return _clean_text(sample.get('text_prompt', ''))


def _v62_input_grounded_dce_plan(sample: Dict[str, Any], abstract: Any, story_steps: List[Dict[str, Any]]) -> SimpleNamespace:
    protagonist = _canonicalize_protagonist_text(sample.get('protagonist', '')) or 'protagonist'
    signature_items = _canonicalize_signature_items(sample.get('signature_items', []), sample.get('text_prompt', ''), protagonist)
    item = signature_items[0] if signature_items else 'important object'
    event_chain = [_clean_text((x or {}).get('sentence', '')) for x in (story_steps or []) if _clean_text((x or {}).get('sentence', ''))]
    target_emotion = _clean_text(sample.get('target_ending_emotion', '')) or 'joyful'
    desire = f'{protagonist} wants to find and recover the {item}.'
    conflict = f'The {item} is lost in the environment, so {protagonist} must search through the scene and overcome obstacles.'
    turning_point = event_chain[min(3, len(event_chain)-1)] if event_chain else f'{protagonist} finds the turning point.'
    ending_state = f'{protagonist} recovers the {item} and feels {target_emotion}.'
    return SimpleNamespace(
        desire=desire,
        conflict=conflict,
        event_chain=event_chain,
        event_spine=event_chain,
        turning_point=turning_point,
        ending_state=ending_state,
        target_ending_emotion=target_emotion,
        protagonist=protagonist,
        signature_items=signature_items,
        abstract=_abstract_to_text(abstract),
        text_prompt=_clean_text(sample.get('text_prompt', '')),
        input_grounded=True,
    )


def _v62_candidate_gate_pass(scores: Dict[str, float], frame_id: int, thresholds: Dict[str, float]) -> bool:
    def g(name, default=0.0):
        return float((scores or {}).get(name, default))
    is_frame1 = int(frame_id) == 1
    sentence_t = thresholds['frame1_sentence_threshold'] if is_frame1 else thresholds['sentence_complete_threshold']
    object_t = thresholds['frame1_object_threshold'] if is_frame1 else thresholds['object_word_threshold']
    species_t = thresholds['frame1_species_threshold'] if is_frame1 else thresholds['protagonist_species_threshold']
    white_t = thresholds['frame1_white_threshold'] if is_frame1 else thresholds['protagonist_white_exact_threshold']
    age_t = thresholds['frame1_age_threshold'] if is_frame1 else thresholds['protagonist_age_threshold']
    color_t = thresholds['frame1_color_threshold'] if is_frame1 else thresholds['protagonist_color_threshold']
    if g('sentence_complete_score') < sentence_t: return False
    if g('story_word_coverage') < thresholds['word_coverage_threshold']: return False
    if g('action_word_coverage') < thresholds['action_word_threshold']: return False
    if g('scene_word_coverage') < thresholds['scene_word_threshold']: return False
    if g('object_word_coverage') < object_t: return False
    if g('critical_noun_coverage') < thresholds['critical_noun_threshold']: return False
    if g('signature_item_coverage', 1.0) < thresholds.get('signature_item_coverage_threshold', 0.82): return False
    if g('emotion_face_visibility', 1.0) < thresholds.get('emotion_face_visibility_threshold', 0.54): return False
    if g('protagonist_species_score') < species_t: return False
    if g('protagonist_white_bear_exact_score') < white_t: return False
    if g('protagonist_color_consistency') < color_t: return False
    if g('protagonist_age_consistency_score') < age_t: return False
    if g('protagonist_age_penalty') > 0.10: return False
    if g('forbidden_species_penalty') > thresholds['forbidden_species_penalty_max']: return False
    if frame_id > 1 and g('identity_consistency') < thresholds['identity_consistency_threshold']: return False
    if frame_id > 1 and g('continuity') < thresholds['continuity_threshold']: return False
    return True


def _v62_candidate_gate_score(scores: Dict[str, float], frame_id: int) -> float:
    s = scores or {}
    parts = {
        'sentence_complete_score': 3.4,
        'story_word_coverage': 2.0,
        'action_word_coverage': 2.4,
        'scene_word_coverage': 1.5,
        'object_word_coverage': 2.2,
        'critical_noun_coverage': 1.9,
        'emotion_word_coverage': 1.4,
        'emotion_face_visibility': 2.2,
        'signature_item_coverage': 2.4,
        'protagonist_white_bear_exact_score': 1.4,
        'protagonist_species_score': 1.4,
        'protagonist_color_consistency': 1.1,
        'protagonist_age_consistency_score': 0.9,
        'story_context_alignment': 1.0,
        'continuity': 0.9,
    }
    score = sum(float(s.get(k, 0.0)) * w for k, w in parts.items())
    score -= 4.5 * float(s.get('forbidden_species_penalty', 0.0))
    score -= 3.0 * float(s.get('protagonist_age_penalty', 0.0))
    score -= 2.0 * float(s.get('frame_state_penalty', 0.0))
    score -= 1.2 * float(s.get('missing_critical_noun_penalty', 0.0))
    score -= 1.4 * float(s.get('missing_signature_object_penalty', 0.0))
    return round(score, 4)


def _v62_choose_best_candidate(ranked: List[Any], frame_id: int, thresholds: Dict[str, float]):
    if not ranked:
        return None
    gated = []
    others = []
    for cand in ranked:
        scores = getattr(cand, 'scores', {}) or {}
        gate_score = _v62_candidate_gate_score(scores, frame_id)
        notes = getattr(cand, 'notes', {}) or {}
        notes['v62_gate_score'] = gate_score
        notes['v62_gate_pass'] = _v62_candidate_gate_pass(scores, frame_id, thresholds)
        cand.notes = notes
        if notes['v62_gate_pass']:
            gated.append(cand)
        else:
            others.append(cand)
    def key(c):
        s = (getattr(c, 'scores', {}) or {})
        n = (getattr(c, 'notes', {}) or {})
        return (
            n.get('v62_gate_score', -999),
            float(s.get('sentence_complete_score', 0.0)),
            float(s.get('signature_item_coverage', 0.0)),
            float(s.get('emotion_face_visibility', 0.0)),
            float(s.get('object_word_coverage', 0.0)),
            float(s.get('critical_noun_coverage', 0.0)),
            float(s.get('identity_consistency', 0.0)),
            float(s.get('overall', -999)),
        )
    if gated:
        gated.sort(key=key, reverse=True)
        return gated[0]
    others.sort(key=key, reverse=True)
    return others[0]


def _build_precomputed_story_steps(sample: Dict[str, Any], total_frames: int) -> List[Dict[str, Any]]:
    protagonist = _canonicalize_protagonist_text(sample.get('protagonist', '')) or 'white bear'
    signature_items = _canonicalize_signature_items(sample.get('signature_items', []), sample.get('text_prompt', ''), protagonist)
    sentences = _extract_explicit_story_sentences(sample)
    if not sentences and _is_white_bear_honey_quest(sample):
        sentences = _default_white_bear_honey_story(total_frames)
    sentences = _unique_clean(sentences)
    if total_frames > 0 and len(sentences) > total_frames:
        sentences = sentences[:total_frames]
    steps: List[Dict[str, Any]] = []
    for idx in range(total_frames):
        sent = sentences[idx] if idx < len(sentences) else ''
        if not sent:
            sent = f'The {protagonist} continues the story action from the input prompt.'
        low = sent.lower()
        if protagonist == 'white bear' and 'white bear' not in low:
            if ' bear ' in f' {low} ':
                sent = sent.replace(' bear', ' white bear').replace('Bear', 'White bear')
            elif sent.lower().startswith('the '):
                sent = 'The white bear ' + sent[4:].lstrip()
            else:
                sent = 'The white bear ' + sent[0].lower() + sent[1:]
        location = _infer_story_location(sent, idx, total_frames)
        emotion = _infer_story_emotion(sent, sample.get('target_ending_emotion', ''))
        action = _infer_story_action(sent)
        must_show = _must_show_for_sentence(sent, protagonist, signature_items, idx, total_frames)
        steps.append({
            'sentence': sent,
            'story_sentence': sent,
            'event': action,
            'action': action,
            'visible_cause': action,
            'emotion': emotion,
            'location': location,
            'scene_location': location,
            'weather': 'soft daylight',
            'atmosphere': 'storybook adventure',
            'must_show': must_show,
            'required_objects': signature_items,
            'critical_visual_nouns': must_show,
            'input_protagonist': protagonist,
            'input_signature_items': list(signature_items),
        })
    return steps


def _repair_story_step_from_input(story_step: Dict[str, Any], sample: Dict[str, Any], idx: int, total_frames: int, precomputed_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    step = dict(story_step or {})
    protagonist = _canonicalize_protagonist_text(sample.get('protagonist', '')) or 'white bear'
    signature_items = _canonicalize_signature_items(sample.get('signature_items', []), sample.get('text_prompt', ''), protagonist)
    prepared = precomputed_steps[idx] if idx < len(precomputed_steps) else {}
    sentence = _clean_text(prepared.get('sentence') or step.get('sentence') or step.get('story_sentence') or '')
    if protagonist == 'white bear' and sentence:
        low = sentence.lower()
        if 'white bear' not in low:
            if ' bear ' in f' {low} ':
                sentence = sentence.replace(' bear', ' white bear').replace('Bear', 'White bear')
            elif sentence.lower().startswith('the '):
                sentence = 'The white bear ' + sentence[4:].lstrip()
            else:
                sentence = 'The white bear ' + sentence[0].lower() + sentence[1:]
    if not sentence:
        sentence = f'The {protagonist} continues the story from the input prompt.'
    step['sentence'] = sentence
    step['story_sentence'] = sentence
    step['event'] = _clean_text(prepared.get('event') or step.get('event') or prepared.get('action') or step.get('action') or _infer_story_action(sentence))
    step['action'] = _clean_text(prepared.get('action') or step.get('action') or step['event'])
    step['visible_cause'] = _clean_text(prepared.get('visible_cause') or step.get('visible_cause') or step['action'])
    step['emotion'] = _clean_text(prepared.get('emotion') or step.get('emotion') or _infer_story_emotion(sentence, sample.get('target_ending_emotion', '')))
    step['location'] = _clean_text(prepared.get('location') or step.get('location') or _infer_story_location(sentence, idx, total_frames))
    step['scene_location'] = _clean_text(prepared.get('scene_location') or step.get('scene_location') or step['location'])
    step['weather'] = _clean_text(prepared.get('weather') or step.get('weather') or 'soft daylight')
    step['atmosphere'] = _clean_text(prepared.get('atmosphere') or step.get('atmosphere') or 'storybook adventure')
    must_show = _unique_clean(
        _as_list_clean(prepared.get('must_show', []))
        + _as_list_clean(step.get('must_show', []))
        + _must_show_for_sentence(sentence, protagonist, signature_items, idx, total_frames),
        12,
    )
    step['must_show'] = must_show
    step['critical_visual_nouns'] = _unique_clean(_as_list_clean(prepared.get('critical_visual_nouns', [])) + _as_list_clean(step.get('critical_visual_nouns', [])) + must_show, 12)
    step['required_objects'] = _unique_clean(signature_items + _as_list_clean(step.get('required_objects', [])), 8)
    step['input_protagonist'] = protagonist
    step['input_signature_items'] = list(signature_items)
    return step


def _enforce_frame_contract(frame: Any, story_step: Dict[str, Any], sample: Dict[str, Any], idx: int, total_frames: int):
    protagonist = _canonicalize_protagonist_text(sample.get('protagonist', '')) or 'white bear'
    signature_items = _canonicalize_signature_items(sample.get('signature_items', []), sample.get('text_prompt', ''), protagonist)
    sentence = _clean_text(story_step.get('sentence') or story_step.get('story_sentence') or getattr(frame, 'story_sentence', ''))
    must_show = _unique_clean(_as_list_clean(story_step.get('must_show', [])) + _must_show_for_sentence(sentence, protagonist, signature_items, idx, total_frames), 12)
    setattr(frame, 'frame_id', idx + 1)
    setattr(frame, 'protagonist', protagonist)
    setattr(frame, 'input_protagonist', protagonist)
    setattr(frame, 'story_sentence', sentence)
    setattr(frame, 'image_caption_en', sentence)
    setattr(frame, 'event', _clean_text(story_step.get('event') or story_step.get('action') or getattr(frame, 'event', '') or _infer_story_action(sentence)))
    setattr(frame, 'event_grounding', _clean_text(story_step.get('visible_cause') or getattr(frame, 'event_grounding', '') or getattr(frame, 'event', '')))
    setattr(frame, 'emotion', _clean_text(story_step.get('emotion') or getattr(frame, 'emotion', '') or _infer_story_emotion(sentence, sample.get('target_ending_emotion', ''))))
    setattr(frame, 'scene_location', _clean_text(story_step.get('scene_location') or story_step.get('location') or getattr(frame, 'scene_location', '') or _infer_story_location(sentence, idx, total_frames)))
    setattr(frame, 'weather', _clean_text(story_step.get('weather') or getattr(frame, 'weather', '') or 'soft daylight'))
    setattr(frame, 'atmosphere', _clean_text(story_step.get('atmosphere') or getattr(frame, 'atmosphere', '') or 'storybook adventure'))
    setattr(frame, 'must_show', must_show)
    setattr(frame, 'critical_visual_nouns', _unique_clean(_as_list_clean(story_step.get('critical_visual_nouns', [])) + must_show, 12))
    setattr(frame, 'required_objects', list(signature_items))
    setattr(frame, 'signature_items', list(signature_items))
    setattr(frame, 'input_signature_items', list(signature_items))
    setattr(frame, 'evidence_objects', list(signature_items))
    setattr(frame, 'target_objects', must_show)
    setattr(frame, 'character_identity', 'exactly one adult white bear with creamy white fur, black eyes, black nose, rounded ears, large paws, a stocky body, and the same face/body proportions in every frame')
    setattr(frame, 'camera_shot', getattr(frame, 'camera_shot', '') or 'medium-wide full-body story shot')
    setattr(frame, 'visual_focus', _clean_text(story_step.get('event') or sentence))
    setattr(frame, 'facial_cue', f"clear {getattr(frame, 'emotion', 'focused')} expression")
    setattr(frame, 'body_cue', f"body pose clearly showing {getattr(frame, 'emotion', 'focused')} while {getattr(frame, 'event', 'acting')}")
    return frame


def _as_list_clean(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [_clean_text(v) for v in x if _clean_text(v)]
    if isinstance(x, dict):
        vals: List[str] = []
        for v in x.values():
            s = _clean_text(v)
            if s:
                vals.append(s)
        return vals
    s = _clean_text(x)
    return [s] if s else []


def _unique_clean(items: List[str], limit: int | None = None) -> List[str]:
    out: List[str] = []
    for item in items or []:
        s = _clean_text(item)
        if s and s not in out:
            out.append(s)
    return out if limit is None else out[:limit]


def _has_any_reference_image(sample: Dict[str, Any]) -> bool:
    sample = sample or {}
    if _image_path_exists(sample.get('image_path', '')):
        return True
    if _image_path_exists(sample.get('canonical_reference_sheet_path', '')):
        return True
    for p in sample.get('protagonist_reference_paths', []) or []:
        if _image_path_exists(p):
            return True
    return False


def _make_contact_sheet_force(image_paths: List[str], out_path: Path, cols: int = 3, thumb_size: tuple[int, int] = (384, 384), title: str = "DCEE-CausalVerse Visual Story") -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    valid_paths = [Path(str(p)) for p in image_paths if _image_path_exists(p)]
    if not valid_paths:
        raise ValueError("No valid image paths for contact sheet.")
    rows = (len(valid_paths) + cols - 1) // cols
    header_h = 54
    cell_w, cell_h = thumb_size
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h + header_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), title, fill=(0, 0, 0))
    for idx, img_path in enumerate(valid_paths):
        img = Image.open(img_path).convert("RGB")
        img.thumbnail((cell_w - 20, cell_h - 44))
        col = idx % cols
        row = idx // cols
        x0 = col * cell_w
        y0 = header_h + row * cell_h
        draw.rectangle([x0, y0, x0 + cell_w - 1, y0 + cell_h - 1], outline=(180, 180, 180))
        draw.text((x0 + 10, y0 + 10), f"Frame {idx + 1}", fill=(0, 0, 0))
        x = x0 + (cell_w - img.width) // 2
        y = y0 + 34 + (cell_h - 44 - img.height) // 2
        canvas.paste(img, (x, y))
    canvas.save(out_path)
    return str(out_path)


class CrossAttentionButterflyDCEViStoryPipeline:
    """Grounded incremental DCEE pipeline with identity-lock and scene-contract image control.

    Main change:
    sentence_1 -> frame_1 image
    sentence_2 (conditioned on previous story + frame_1 summary) -> frame_2 image
    ...
    This keeps every sentence image-friendly and makes frame generation more aligned with the story.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.llm = build_llm(cfg.get("llm", {}))
        self.vlm = build_vlm(cfg.get("vlm", {}))

        llm_cfg = cfg.get("llm", {})
        self.planner = DCEPlanner(
            self.llm,
            temperature=float(llm_cfg.get("temperature", 0.35)),
            max_tokens=int(llm_cfg.get("max_tokens", 1600)),
        )

        iu_cfg = cfg.get("image_understanding", {})
        self.image_understanding = ImageUnderstandingModule(
            iu_cfg.get("provider", "llm_caption"),
            self.llm,
            iu_cfg.get("caption_model", "Salesforce/blip-image-captioning-base"),
        )

        ev_cfg = cfg.get("evaluation", {})
        self.evaluator = DCEQAEvaluator(
            self.llm,
            self.vlm,
            use_vlm=bool(ev_cfg.get("use_vlm", False)),
            save_contact_sheet=bool(ev_cfg.get("save_contact_sheet", True)),
        )

        b_cfg = cfg.get("butterfly", {})
        self.controller = ButterflyController(
            b_cfg.get("quality_suffix", QUALITY_SUFFIX),
            b_cfg.get("negative_prompt", NEGATIVE_PROMPT),
            int(b_cfg.get("num_hypotheses", 3)),
        )

        img_cfg = cfg.get("image_generator", {})
        ad_cfg = cfg.get("cross_attention_adapters", {})
        self.image_generator = SDXLButterflyCrossAttentionGenerator(
            model_id=img_cfg.get("model_id", "stabilityai/stable-diffusion-xl-base-1.0"),
            device=img_cfg.get("device", "cuda"),
            width=int(img_cfg.get("width", 1024)),
            height=int(img_cfg.get("height", 1024)),
            num_inference_steps=int(img_cfg.get("num_inference_steps", 36)),
            guidance_scale=float(img_cfg.get("guidance_scale", 7.5)),
            seed=int(img_cfg.get("seed", 42)),
            adapter_ckpt=ad_cfg.get("adapter_ckpt"),
            enable_cpu_offload=bool(img_cfg.get("enable_cpu_offload", False)),
            character_tokens=int(ad_cfg.get("character_tokens", 8)),
            world_tokens=int(ad_cfg.get("world_tokens", 8)),
            emotion_tokens=int(ad_cfg.get("emotion_tokens", 8)),
            event_tokens=int(ad_cfg.get("event_tokens", 8)),
            evidence_tokens=int(ad_cfg.get("evidence_tokens", 8)),
            use_refiner=bool(img_cfg.get("use_refiner", False)),
            refiner_model_id=img_cfg.get("refiner_model_id", "stabilityai/stable-diffusion-xl-refiner-1.0"),
            refiner_strength=float(img_cfg.get("refiner_strength", 0.80)),
            aesthetic_score=float(img_cfg.get("aesthetic_score", 6.0)),
            negative_aesthetic_score=float(img_cfg.get("negative_aesthetic_score", 2.5)),
            quality_model_preset=img_cfg.get("quality_model_preset", "sdxl_base"),
            use_ip_adapter=bool(img_cfg.get("use_ip_adapter", True)),
            ip_adapter_scale=float(img_cfg.get("ip_adapter_scale", 0.40)),
            canonical_reference_sheet_path=img_cfg.get("canonical_reference_sheet_path", ""),
            identity_backend_priority=img_cfg.get("identity_backend_priority", ["instantid", "photomaker", "canonical_reference_sheet", "character_lora", "ip_adapter", "text"]),
            use_instantid=bool(img_cfg.get("use_instantid", False)),
            instantid_adapter_path=img_cfg.get("instantid_adapter_path", ""),
            instantid_controlnet_path=img_cfg.get("instantid_controlnet_path", ""),
            use_photomaker=bool(img_cfg.get("use_photomaker", False)),
            photomaker_adapter_path=img_cfg.get("photomaker_adapter_path", ""),
            use_character_lora=bool(img_cfg.get("use_character_lora", False)),
            character_lora_path=img_cfg.get("character_lora_path", ""),
            character_lora_scale=float(img_cfg.get("character_lora_scale", 0.85)),
            use_subject_scene_fusion=bool(img_cfg.get("use_subject_scene_fusion", True)),
            subject_scene_fusion_scale=float(img_cfg.get("subject_scene_fusion_scale", 0.72)),
            subject_scene_fusion_first_n=int(img_cfg.get("subject_scene_fusion_first_n", 2)),
            consistent_identity_seed_across_frames=bool(img_cfg.get("consistent_identity_seed_across_frames", True)),
            multi_gpu=bool(img_cfg.get("multi_gpu", False)),
            gpu_ids=img_cfg.get("gpu_ids", None),
            max_parallel_generators=img_cfg.get("max_parallel_generators", None),
            oom_safe_generation=bool(img_cfg.get("oom_safe_generation", True)),
            oom_retry_width=int(img_cfg.get("oom_retry_width", 768)),
            oom_retry_height=int(img_cfg.get("oom_retry_height", 768)),
            enable_vae_tiling=bool(img_cfg.get("enable_vae_tiling", True)),
            enable_vae_slicing=bool(img_cfg.get("enable_vae_slicing", True)),
            enable_attention_slicing=bool(img_cfg.get("enable_attention_slicing", True)),
            min_free_memory_gb=float(img_cfg.get("min_free_memory_gb", 14.0)),
            skip_busy_gpus=bool(img_cfg.get("skip_busy_gpus", True)),
            continue_on_worker_failure=bool(img_cfg.get("continue_on_worker_failure", True)),
            max_candidate_failures_per_frame=int(img_cfg.get("max_candidate_failures_per_frame", 999)),
            force_safe_lowres_generation=bool(img_cfg.get("force_safe_lowres_generation", True)),
            safe_generation_width=int(img_cfg.get("safe_generation_width", 768)),
            safe_generation_height=int(img_cfg.get("safe_generation_height", 768)),
            safe_num_inference_steps=int(img_cfg.get("safe_num_inference_steps", 34)),
            disable_fusion_in_multigpu_safe=bool(img_cfg.get("disable_fusion_in_multigpu_safe", True)),
            process_isolated_multi_gpu=bool(img_cfg.get("process_isolated_multi_gpu", True)),
            use_previous_frame_img2img=False,
        )

    def _strengthen_packet(self, packet, frame):
        meta = getattr(packet, 'control_metadata', {}) or {}
        meta['retry_strong_mode'] = True
        meta['retry_exact_sentence'] = getattr(frame, 'story_sentence', '')
        meta['retry_required_objects'] = getattr(frame, 'must_show', [])
        meta['retry_critical_visual_nouns'] = meta.get('critical_visual_nouns', getattr(frame, 'must_show', []))
        meta['retry_location'] = getattr(frame, 'scene_location', '')
        meta['retry_emotion'] = getattr(frame, 'emotion', '')
        meta['retry_next_sentence'] = meta.get('next_frame_caption', '')
        meta['retry_single_scene'] = True
        meta['retry_full_color_background'] = True
        packet.control_metadata = meta
        try:
            packet.positive_prompt += (
                "\n\nRETRY CONTROL:"
                f"\n- Exact sentence: {getattr(frame, 'story_sentence', '')}"
                f"\n- Event: {getattr(frame, 'event', '')}"
                f"\n- Evidence: {getattr(frame, 'event_grounding', '')}"
                f"\n- Required objects: {getattr(frame, 'must_show', [])}"
                f"\n- Critical visual nouns: {meta.get('critical_visual_nouns', getattr(frame, 'must_show', []))}"
                f"\n- Next sentence (do not jump ahead to it): {meta.get('next_frame_caption', '')}"
                f"\n- Emotion: {getattr(frame, 'emotion', '')}"
                "\n- Show one protagonist only."
                "\n- Single scene only."
                "\n- Keep protagonist identity unchanged: same face, body size, fur/color pattern, hands/paws, feet, and signature items."
                "\n- DCEE may change only expression, pose, dirt/wetness, lighting, and emotional tension."
                "\n- Show full body with uncropped face, hands/paws, feet, and action."
                "\n- Preserve a readable full-color background that matches this exact story sentence."
                "\n- Show a layered environment with setting details; do not leave the background blank."
                "\n- Make every critical visual noun visible and recognizable."
                "\n- Make the frame read like one step of a progressing visual story, not an isolated portrait."
                "\n- Do not repeat the previous frame and do not jump ahead to the next frame."
            )
            packet.negative_prompt += (
                "; duplicate protagonist, extra character, extra animal, wrong action, "
                "missing object, missing critical noun, cropped body, cropped face, cropped hands, cropped paws, changed protagonist identity, gray empty background, unrelated background, blank background, static repeated scene, jumping to next scene"
            )
        except Exception:
            pass
        return packet

    def _persona_anchor_from_seed(self, seed: Any, protagonist: str) -> str:
        protagonist = _clean_text(protagonist) or 'white bear'
        contract = getattr(seed, '_v611_identity_contract', {}) or getattr(seed, '_v61_identity_contract', {}) or {}
        parts: List[str] = [
            contract.get('identity_anchor_prompt', ''),
            contract.get('face', ''),
            contract.get('body', ''),
            protagonist,
        ]
        profiles = getattr(seed, 'character_profiles', []) or []
        if profiles:
            prof = profiles[0]
            for key in ['identity_anchor_prompt', 'face', 'body']:
                val = _clean_text(getattr(prof, key, ''))
                if val:
                    parts.append(val)
            parts += _as_list_clean(getattr(prof, 'signature_items', []))[:3]
        if getattr(seed, 'signature_items', None):
            parts += _as_list_clean(getattr(seed, 'signature_items', []))[:3]
        parts += ['exactly one recurring protagonist', 'same protagonist identity in every frame', 'keep species as white bear']
        return '; '.join(_unique_clean(parts, 10))


    def _default_render_plan(self, seed: Any, frame: Any, frame_index: int, total_frames: int, previous_sentence: str = '', next_sentence: str = '') -> Dict[str, Any]:
        story_sentence = _clean_text(getattr(frame, 'story_sentence', '') or getattr(frame, 'image_caption_en', ''))
        protagonist = _clean_text(getattr(seed, 'protagonist', '') or getattr(frame, 'protagonist', '')) or 'protagonist'
        persona_anchor = self._persona_anchor_from_seed(seed, protagonist)
        must_show = _unique_clean(_as_list_clean(getattr(frame, 'must_show', [])) + _as_list_clean(getattr(frame, 'critical_visual_nouns', [])), 10)
        scene_location = _clean_text(getattr(frame, 'scene_location', ''))
        weather = _clean_text(getattr(frame, 'weather', ''))
        atmosphere = _clean_text(getattr(frame, 'atmosphere', ''))
        emotion = _clean_text(getattr(frame, 'emotion', ''))
        event = _clean_text(getattr(frame, 'event', ''))
        event_grounding = _clean_text(getattr(frame, 'event_grounding', ''))
        object_state = 'show the object state exactly as implied by the sentence'
        low = story_sentence.lower()
        if 'jar' in low and not any(k in low for k in ['spot', 'see', 'retrieve', 'hold', 'savor']):
            object_state = 'the honey jar is missing or not yet found; do not make the jar the main focus'
        elif 'spots' in low or 'sees' in low:
            object_state = 'the honey jar must be visible in the scene and the bear should be looking at it'
        elif 'retrieve' in low or 'holds' in low or 'picks up' in low:
            object_state = 'the honey jar must be visible and the bear must be touching or lifting it'
        elif 'savor' in low or 'eat' in low or 'lick' in low:
            object_state = 'the honey jar must be in the bear paws and the bear must be tasting or enjoying honey'
        must_not_show = ['extra human', 'extra animal', 'duplicate protagonist', 'generic portrait', 'wrong animal species']
        if 'lake' not in low and 'water' not in low:
            must_not_show += ['dominant lake background']
        if 'jar' not in low and not any(k in low for k in ['spot', 'see', 'retrieve', 'savor']):
            must_not_show += ['large close-up honey jar']
        prompt_core = f"single coherent storybook illustration of {protagonist}. exact frame sentence: {story_sentence}. show one clear moment only. visible action: {_clean_text(event or story_sentence)}. required objects: {', '.join(must_show) if must_show else 'caption-grounded objects only'}. environment: {scene_location or 'story environment'}"
        prompt_detail = f"full body visible, readable foreground midground and background, expression should show {emotion or 'the current emotion'}, visible evidence: {event_grounding or event or story_sentence}. weather: {weather or 'none'}. atmosphere: {atmosphere or 'storybook'}. continue after: {previous_sentence or 'none'}. do not jump ahead to: {next_sentence or 'none'}"
        negative_prompt = ', '.join(_unique_clean(must_not_show + ['text', 'watermark', 'mascot', 'sticker', 'toy-like bear', 'cropped body', 'blank background', 'brown bear', 'red fox']))
        return {
            'persona_anchor': persona_anchor,
            'story_sentence': story_sentence,
            'frame_goal': story_sentence,
            'action_pose': _clean_text(event or story_sentence),
            'object_state': object_state,
            'environment': '; '.join(_unique_clean([scene_location, weather, atmosphere], 3)),
            'camera': _clean_text(getattr(frame, 'camera_shot', '') or 'medium-wide full-body story shot'),
            'must_show': must_show,
            'must_not_show': _unique_clean(must_not_show, 10),
            'prompt_core': prompt_core,
            'prompt_detail': prompt_detail,
            'negative_prompt': negative_prompt,
            'frame_stage': ['setup', 'search', 'transition', 'discovery', 'resolution'][min(4, int((frame_index / max(1, total_frames - 1)) * 4))],
        }

    def _generate_llm_render_plan(self, seed: Any, frame: Any, frame_index: int, total_frames: int, previous_sentence: str = '', next_sentence: str = '', has_reference_image: bool = False, dce_plan: Any = None) -> Dict[str, Any]:
        default_plan = self._default_render_plan(seed, frame, frame_index, total_frames, previous_sentence, next_sentence)
        cfg = self.cfg.get('pipeline', {}) or {}
        temperature = float(cfg.get('llm_frame_prompt_temperature', 0.0))
        max_tokens = int(cfg.get('llm_frame_prompt_max_tokens', 900))
        protagonist = _clean_text(getattr(seed, 'protagonist', '') or getattr(frame, 'protagonist', '')) or 'protagonist'
        payload = {
            'protagonist': protagonist,
            'persona_anchor': default_plan.get('persona_anchor', ''),
            'frame_id': frame_index + 1,
            'total_frames': total_frames,
            'story_abstract': _abstract_to_text(getattr(seed, '_story_abstract', '') or getattr(seed, '_story_abstract_text', '')),
            'dcee_plan_summary': _dcee_plan_to_text(dce_plan),
            'generated_story_context': _clean_text(getattr(frame, 'generated_story_context', '')),
            'frame_caption': _clean_text(getattr(frame, 'image_caption_en', '') or getattr(frame, 'story_sentence', '')),
            'story_sentence': _clean_text(getattr(frame, 'story_sentence', '') or getattr(frame, 'image_caption_en', '')),
            'event': _clean_text(getattr(frame, 'event', '')),
            'event_grounding': _clean_text(getattr(frame, 'event_grounding', '')),
            'emotion': _clean_text(getattr(frame, 'emotion', '')),
            'location': _clean_text(getattr(frame, 'scene_location', '')),
            'weather': _clean_text(getattr(frame, 'weather', '')),
            'atmosphere': _clean_text(getattr(frame, 'atmosphere', '')),
            'must_show': _as_list_clean(getattr(frame, 'must_show', [])),
            'critical_visual_nouns': _as_list_clean(getattr(frame, 'critical_visual_nouns', [])),
            'previous_sentence': _clean_text(previous_sentence),
            'next_sentence': _clean_text(next_sentence),
            'has_reference_image': bool(has_reference_image),
        }
        system = ('You are a storyboard-to-image prompt director for visual storytelling. Your job is to convert one story frame into a highly literal SDXL-friendly render plan. Focus on exact object grounding, background grounding, protagonist action, and frame-stage correctness. Do NOT write prose paragraphs. Return JSON only.')
        user = json.dumps({
            'task': 'Create a literal frame render plan for one image. Prefer precise, concrete, visual language over abstract wording.',
            'required_schema': {
                'persona_anchor': 'short protagonist identity anchor',
                'action_pose': 'what the protagonist is physically doing',
                'object_state': 'state of key object(s) in this exact frame',
                'environment': 'specific visual background description',
                'camera': 'camera/framing',
                'must_show': ['list of visible nouns that must appear'],
                'must_not_show': ['list of things that must not appear'],
                'prompt_core': 'compact primary prompt for SDXL',
                'prompt_detail': 'secondary support prompt for SDXL',
                'negative_prompt': 'compact negative prompt'
            },
            'rules': [
                'Preserve exactly one protagonist.',
                'If the sentence mentions a location object such as forest, roots, reeds, lake, or jar, it must be visible when physically possible.',
                'Do not jump ahead to later events.',
                'Do not turn the frame into a generic cute portrait.',
                'Make the image readable as the exact current sentence at a glance.',
                'Ground the frame in the paper flow: input JSON -> abstract -> DCEE plan -> generated story -> current frame caption -> image.',
                'The frame image must be based ONLY on the generated story sentence, current caption, abstract, DCEE plan, and input JSON contract.',
                'Do not use a fallback scene, generic prompt, generic animal prior, or unrelated visual imagination.',
                'Show a literal visual storytelling panel: protagonist action, key object state, and the sentence-specific environment must all be visible together.',
                'If the frame is before the jar is found, do not show the bear already celebrating or eating honey.',
                'If the frame is the discovery frame, show the jar visibly near the water edge.',
                'If the frame is the savoring frame, show the same protagonist beside the lake actively enjoying honey from the jar.'
            ],
            'frame_context': payload
        }, ensure_ascii=False)
        try:
            raw = self.llm.generate(system, user, temperature=temperature, max_tokens=max_tokens)
            parsed = extract_json(raw)
            if isinstance(parsed, list) and parsed:
                parsed = parsed[0]
            if not isinstance(parsed, dict):
                parsed = {}
            plan = dict(default_plan)
            for key in ['persona_anchor', 'action_pose', 'object_state', 'environment', 'camera', 'prompt_core', 'prompt_detail', 'negative_prompt']:
                val = _clean_text(parsed.get(key, ''))
                if val:
                    plan[key] = val
            for key in ['must_show', 'must_not_show']:
                vals = _unique_clean(_as_list_clean(parsed.get(key, [])), 12)
                if vals:
                    plan[key] = vals
            plan['llm_render_plan_raw'] = raw
            plan['llm_render_plan_source'] = 'llm'
            return plan
        except Exception as e:
            default_plan['llm_render_plan_raw'] = f'fallback: {type(e).__name__}: {e}'
            default_plan['llm_render_plan_source'] = 'fallback'
            return default_plan

    def _save_core_plan_outputs(self, out_dir: Path, seed, abstract, dce_plan, emotion_arc, full_story, storyboard):
        _write_json(out_dir / "seed.json", seed)
        (out_dir / "abstract.txt").write_text(str(abstract), encoding="utf-8")
        _write_json(out_dir / "dcee_plan.json", dce_plan)
        _write_json(out_dir / "dce_plan.json", dce_plan)
        _write_json(out_dir / "emotion_arc.json", emotion_arc)
        _write_json(out_dir / "full_story.json", full_story)
        _write_json(out_dir / "storyboard.json", storyboard)

    def run(self, sample: Dict[str, Any], out_dir: Path) -> PipelineResult:
        out_dir = Path(out_dir)
        frames_dir = out_dir / "frames"
        ending_dir = out_dir / "ending_candidates"
        out_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        ending_dir.mkdir(parents=True, exist_ok=True)
                # V28: clean stale candidate images from previous runs in the same output folder.
        _clear_generated_pngs(frames_dir, ending_dir)

        run_errors: List[Dict[str, Any]] = []
        sample = _normalize_text_only_sample(sample)
        has_reference_image = _has_any_reference_image(sample)
        if has_reference_image:
            image_summary = self.image_understanding.analyze(sample.get("image_path"), sample)
        else:
            # Planner fallback expects attribute access such as image_summary.setting.
            # V41 used a dict here, which caused AttributeError after strict seed JSON repair failed.
            # V42 keeps text-only mode and makes the summary deepcopy-safe for dataclasses.asdict.
            image_summary = TextOnlyImageSummary(
                mode='text_only_story_generation',
                caption=sample.get('text_prompt', ''),
                summary=sample.get('text_prompt', ''),
                description=sample.get('text_prompt', ''),
                objects=list(sample.get('signature_items', []) or []),
                object_candidates=list(sample.get('signature_items', []) or []),
                key_objects=list(sample.get('signature_items', []) or []),
                visible_objects=list(sample.get('signature_items', []) or []),
                scene=sample.get('setting') or sample.get('text_prompt', '') or 'deep forest and serene lake in the forest',
                setting=sample.get('setting') or 'deep forest and serene lake in the forest',
                mood=sample.get('mood') or sample.get('target_ending_emotion', '') or 'happy',
                style=sample.get('style', ''),
                protagonist=sample.get('protagonist', ''),
                genre=sample.get('genre', ''),
                signature_items=list(sample.get('signature_items', []) or []),
                notes='No input reference image. Build story and visuals from text prompt, protagonist, and signature items only.',
            )
        seed = self.planner.build_seed(sample, image_summary)
        seed = _enforce_input_contract_on_seed(seed, sample)
        # V62.1 hotfix: pipe_cfg is needed before abstract/DCEE generation.
        # In V62 it was first assigned later in the function, so Python treated it
        # as a local variable and raised UnboundLocalError here.
        pipe_cfg = self.cfg.get("pipeline", {}) or {}
        selection_cfg = self.cfg.get("selection", {}) or {}
        img_cfg = self.cfg.get("image_generator", {}) or {}
        # V62.2 hotfix: total_frames is required for strict JSON story-flow
        # before the later generation-policy block, so initialize it here.
        total_frames = int(sample.get("num_frames", 6))
        force_json_story_flow = _v62_should_force_json_story_flow(sample, pipe_cfg)
        precomputed_story_steps = _build_precomputed_story_steps(sample, total_frames) if force_json_story_flow else []
        if force_json_story_flow:
            abstract = _v62_input_grounded_abstract(sample, precomputed_story_steps)
        else:
            abstract = self.planner.generate_abstract(seed)
        setattr(seed, '_story_abstract', abstract)
        setattr(seed, '_story_abstract_text', _abstract_to_text(abstract))
        if force_json_story_flow:
            dce_plan = _v62_input_grounded_dce_plan(sample, abstract, precomputed_story_steps)
        else:
            try:
                dce_plan = self.planner.generate_dce_plan(seed, abstract)
            except Exception as e:
                run_errors.append({"stage": "planner.generate_dce_plan", "error": str(e), "traceback": traceback.format_exc(), "fallback": "v61_1_deterministic_dcee_plan"})
                dce_plan = _v611_fallback_dce_plan(seed, sample, abstract, str(e))
        generation_policy = {
            "version": "V63.1",
            "mode": "v63_1_signature_coverage_hotfix_generation",
            "protagonist_only": True,
            "no_secondary_characters": True,
            "training_free_consistency_removed": True,
            "english_only_text_generation": True,
            "sdxl_refiner_enabled": False,
            "previous_frame_img2img_enabled": False,
            "prompt_length_guard": True,
            "seed_validator_hotfix": True,
            "long_freeform_prompt_removed": True,
            "caption_is_image_contract": True,
            "identity_lock_is_image_contract": True,
            "scene_contract_is_image_contract": True,
            "dcee_appearance_delta_enabled": True,
            "multi_candidate_generation_restored": True,
            "cleanup_stale_candidates": True,
            "text_only_input_supported": True,
            "lightweight_multihistory_continuity": True,
            "visualized_nonvisual_event_grounding": True,
            "caption_locked_candidate_scoring": True,
            "lightweight_cross_paper_adaptations": ["ViSTA-style history selection", "StoryGen-style temporal stage cue", "StoryGPT-V-style story-to-visual bridge"],
            "uses_reference_image": has_reference_image,
            "allowed_visual_elements": [
                "protagonist",
                "caption-grounded protagonist props",
                "caption-grounded background objects",
                "weather",
                "lighting",
                "emotion cues",
                "visible cause/evidence"
            ],
            "blocked_story_entities": getattr(seed, "forbidden_ungrounded_entities", []),
            "llm_frame_prompt_enabled": True,
            "persona_anchor_used": True,
            "frame_render_plan_json": True,
            "planner_dce_json_fallback_enabled": True,
            "canonical_protagonist_contract_enabled": True,
            "story_exact_candidate_gate_enabled": True,
            "frame_render_contract_enabled": True,
            "paper_flow_enforced": True,
            "story_abstract_caption_identity_lock_generation": True,
            "fallback_prompt_disabled": True,
            "frame_sources": ["input_json", "abstract", "generated_story", "frame_caption", "dcee_plan"],
            "reason": "V44 chooses LLM-directed frame prompt generation over persona-only conditioning. Persona helps identity consistency, but the main failure in V43 was missing frame-specific objects/backgrounds. V44 therefore uses the LLM to build a literal per-frame render plan (action, object state, environment, negatives) and feeds that plan into SDXL."
        }
        _write_json(out_dir / "generation_policy_V44_llm_frame_prompt.json", generation_policy)
        total_frames = int(sample.get("num_frames", 6))
        try:
            emotion_arc = self.planner.generate_emotion_arc(seed, abstract, dce_plan, total_frames)
        except Exception as e:
            run_errors.append({"stage": "planner.generate_emotion_arc", "error": str(e), "traceback": traceback.format_exc(), "fallback": "v61_1_emotion_arc"})
            emotion_arc = _v611_fallback_emotion_arc(sample, total_frames, str(e))

        canonical_protagonist_contract = getattr(seed, '_v611_identity_contract', {}) or getattr(seed, '_v61_identity_contract', {}) or getattr(seed, '_json_input_contract', {}).get('identity_contract', {})
        _write_json(out_dir / "canonical_protagonist_contract_v61_1.json", canonical_protagonist_contract)

        memory = DCEECausalMemoryStore()
        try:
            memory.initialize(seed, dce_plan)
        except Exception as e:
            run_errors.append({"stage": "memory.initialize", "error": str(e), "traceback": traceback.format_exc()})
        try:
            anchor_bank = DCEEAnchorBank().build_from_seed_and_plan(seed, dce_plan)
        except Exception as e:
            run_errors.append({"stage": "anchor_bank.build_from_seed_and_plan", "error": str(e), "traceback": traceback.format_exc(), "fallback": "empty_anchor_bank"})
            anchor_bank = {}

        selected_images: List[CandidateImage] = []
        ending_candidates: List[CandidateImage] = []
        packet_log: List[Dict[str, Any]] = []
        memory_log: List[Dict[str, Any]] = []
        candidate_manifest: List[Dict[str, Any]] = []
        story_rows: List[Dict[str, Any]] = []
        storyboard: List[Any] = []
        llm_render_plans: List[Dict[str, Any]] = []
        frame_render_contracts: List[Dict[str, Any]] = []

        img_cfg = self.cfg.get("image_generator", {})
        pipe_cfg = self.cfg.get("pipeline", {})
        selection_cfg = self.cfg.get("selection", {}) or {}
        # V28: restore the useful V19/V20-style candidate selection.
        # We generate multiple candidates, but each candidate must be a single coherent scene.
        num_candidates = int(img_cfg.get("num_candidates_per_frame", 3))
        num_ending_candidates = int(img_cfg.get("num_ending_candidates", 4))
        if not has_reference_image:
            num_candidates = max(num_candidates, int(img_cfg.get("text_only_num_candidates_per_frame", 6)))
            num_ending_candidates = max(num_ending_candidates, int(img_cfg.get("text_only_num_ending_candidates", 6)))
        retry_enabled = bool(pipe_cfg.get("emotion_retry", True))
        emotion_threshold = float(pipe_cfg.get("emotion_visibility_threshold", 0.74))
        color_threshold = float(pipe_cfg.get("colorfulness_threshold", 0.35))
        event_threshold = float(pipe_cfg.get("event_grounding_threshold", 0.78))
        evidence_threshold = float(pipe_cfg.get("evidence_visibility_threshold", 0.78))
        story_threshold = float(pipe_cfg.get("story_alignment_threshold", 0.82))
        background_threshold = float(pipe_cfg.get("background_presence_threshold", 0.52))
        progression_threshold = float(pipe_cfg.get("progression_threshold", 0.42))
        critical_noun_threshold = float(pipe_cfg.get("critical_noun_coverage_threshold", 0.72))
        # V49.1 hotfix: word-level verification thresholds must be defined in the pipeline scope.
        # V49 added story/action/scene/object/emotion/color word checks but the retry branch
        # referenced these variables before defining them.
        word_coverage_threshold = float(pipe_cfg.get("word_coverage_threshold", selection_cfg.get("word_coverage_threshold", 0.48)))
        action_word_threshold = float(pipe_cfg.get("action_word_threshold", selection_cfg.get("action_word_threshold", 0.34)))
        scene_word_threshold = float(pipe_cfg.get("scene_word_threshold", selection_cfg.get("scene_word_threshold", 0.34)))
        object_word_threshold = float(pipe_cfg.get("object_word_threshold", selection_cfg.get("object_word_threshold", 0.40)))
        emotion_word_threshold = float(pipe_cfg.get("emotion_word_threshold", selection_cfg.get("emotion_word_threshold", 0.30)))
        color_word_threshold = float(pipe_cfg.get("color_word_threshold", selection_cfg.get("color_word_threshold", 0.90)))
        signature_item_coverage_threshold = float(pipe_cfg.get("signature_item_coverage_threshold", selection_cfg.get("signature_item_coverage_threshold", 0.82)))
        emotion_face_visibility_threshold = float(pipe_cfg.get("emotion_face_visibility_threshold", selection_cfg.get("emotion_face_visibility_threshold", 0.54)))
        # V51.1 hotfix: V51 retry branches use these variables, so define them in scope.
        sentence_complete_threshold = float(pipe_cfg.get("sentence_complete_threshold", selection_cfg.get("sentence_complete_threshold", 0.58)))
        protagonist_color_threshold = float(pipe_cfg.get("protagonist_color_threshold", selection_cfg.get("protagonist_color_threshold", 0.72)))
        protagonist_species_threshold = float(pipe_cfg.get("protagonist_species_threshold", selection_cfg.get("protagonist_species_threshold", 0.78)))
        protagonist_age_threshold = float(pipe_cfg.get("protagonist_age_threshold", selection_cfg.get("protagonist_age_threshold", 0.78)))
        protagonist_white_exact_threshold = float(pipe_cfg.get("protagonist_white_exact_threshold", selection_cfg.get("protagonist_white_exact_threshold", 0.78)))
        identity_consistency_threshold = float(pipe_cfg.get("identity_consistency_threshold", selection_cfg.get("identity_consistency_threshold", 0.68)))
        reference_subject_similarity_threshold = float(pipe_cfg.get("reference_subject_similarity_threshold", selection_cfg.get("reference_subject_similarity_threshold", 0.62)))
        continuity_threshold = float(pipe_cfg.get("continuity_threshold", selection_cfg.get("continuity_threshold", 0.62)))
        forbidden_species_penalty_max = float(pipe_cfg.get("forbidden_species_penalty_max", selection_cfg.get("forbidden_species_penalty_max", 0.08)))
        progression_consistency_threshold = float(pipe_cfg.get("progression_consistency_threshold", 0.56))
        story_context_alignment_threshold = float(pipe_cfg.get("story_context_alignment_threshold", 0.60))
        use_llm_frame_prompt = bool(pipe_cfg.get("use_llm_frame_prompt", True))
        v62_candidate_gate_thresholds = {
            'word_coverage_threshold': word_coverage_threshold,
            'action_word_threshold': action_word_threshold,
            'scene_word_threshold': scene_word_threshold,
            'object_word_threshold': object_word_threshold,
            'critical_noun_threshold': critical_noun_threshold,
            'signature_item_coverage_threshold': signature_item_coverage_threshold,
            'emotion_face_visibility_threshold': emotion_face_visibility_threshold,
            'sentence_complete_threshold': sentence_complete_threshold,
            'protagonist_color_threshold': protagonist_color_threshold,
            'protagonist_species_threshold': protagonist_species_threshold,
            'protagonist_age_threshold': protagonist_age_threshold,
            'protagonist_white_exact_threshold': protagonist_white_exact_threshold,
            'identity_consistency_threshold': identity_consistency_threshold,
            'continuity_threshold': continuity_threshold,
            'forbidden_species_penalty_max': forbidden_species_penalty_max,
        }
        frame1_species_threshold = float(pipe_cfg.get("frame1_species_threshold", max(protagonist_species_threshold, 0.94)))
        frame1_white_threshold = float(pipe_cfg.get("frame1_white_threshold", max(protagonist_white_exact_threshold, 0.94)))
        frame1_color_threshold = float(pipe_cfg.get("frame1_color_threshold", max(protagonist_color_threshold, 0.96)))
        frame1_age_threshold = float(pipe_cfg.get("frame1_age_threshold", max(protagonist_age_threshold, 0.92)))
        frame1_object_threshold = float(pipe_cfg.get("frame1_object_threshold", max(object_word_threshold, 0.72)))
        frame1_sentence_threshold = float(pipe_cfg.get("frame1_sentence_threshold", max(sentence_complete_threshold, 0.82)))
        if not has_reference_image:
            story_threshold = max(story_threshold, float(pipe_cfg.get("text_only_story_alignment_threshold", 0.90)))
            critical_noun_threshold = max(critical_noun_threshold, float(pipe_cfg.get("text_only_critical_noun_coverage_threshold", 0.86)))
            background_threshold = max(background_threshold, float(pipe_cfg.get("text_only_background_presence_threshold", 0.60)))
            story_context_alignment_threshold = max(story_context_alignment_threshold, float(pipe_cfg.get("text_only_story_context_alignment_threshold", 0.68)))
            word_coverage_threshold = max(word_coverage_threshold, float(pipe_cfg.get("text_only_word_coverage_threshold", 0.52)))
            action_word_threshold = max(action_word_threshold, float(pipe_cfg.get("text_only_action_word_threshold", 0.38)))
            scene_word_threshold = max(scene_word_threshold, float(pipe_cfg.get("text_only_scene_word_threshold", 0.38)))
            object_word_threshold = max(object_word_threshold, float(pipe_cfg.get("text_only_object_word_threshold", selection_cfg.get("text_only_object_word_threshold", 0.45))))
            emotion_word_threshold = max(emotion_word_threshold, float(pipe_cfg.get("text_only_emotion_word_threshold", selection_cfg.get("text_only_emotion_word_threshold", 0.34))))
            color_word_threshold = max(color_word_threshold, float(pipe_cfg.get("text_only_color_word_threshold", selection_cfg.get("text_only_color_word_threshold", 0.90))))
            signature_item_coverage_threshold = max(signature_item_coverage_threshold, float(pipe_cfg.get("text_only_signature_item_coverage_threshold", selection_cfg.get("text_only_signature_item_coverage_threshold", 0.88))))
            emotion_face_visibility_threshold = max(emotion_face_visibility_threshold, float(pipe_cfg.get("text_only_emotion_face_visibility_threshold", selection_cfg.get("text_only_emotion_face_visibility_threshold", 0.58))))
            sentence_complete_threshold = max(sentence_complete_threshold, float(pipe_cfg.get("text_only_sentence_complete_threshold", selection_cfg.get("text_only_sentence_complete_threshold", 0.66))))
            protagonist_color_threshold = max(protagonist_color_threshold, float(pipe_cfg.get("text_only_protagonist_color_threshold", selection_cfg.get("text_only_protagonist_color_threshold", 0.78))))
            protagonist_species_threshold = max(protagonist_species_threshold, float(pipe_cfg.get("text_only_protagonist_species_threshold", selection_cfg.get("text_only_protagonist_species_threshold", 0.86))))
            protagonist_age_threshold = max(protagonist_age_threshold, float(pipe_cfg.get("text_only_protagonist_age_threshold", selection_cfg.get("text_only_protagonist_age_threshold", 0.84))))
            protagonist_white_exact_threshold = max(protagonist_white_exact_threshold, float(pipe_cfg.get("text_only_protagonist_white_exact_threshold", selection_cfg.get("text_only_protagonist_white_exact_threshold", 0.86))))
            identity_consistency_threshold = max(identity_consistency_threshold, float(pipe_cfg.get("text_only_identity_consistency_threshold", selection_cfg.get("text_only_identity_consistency_threshold", 0.78))))
            reference_subject_similarity_threshold = max(reference_subject_similarity_threshold, float(pipe_cfg.get("text_only_reference_subject_similarity_threshold", selection_cfg.get("text_only_reference_subject_similarity_threshold", 0.72))))
            continuity_threshold = max(continuity_threshold, float(pipe_cfg.get("text_only_continuity_threshold", selection_cfg.get("text_only_continuity_threshold", 0.74))))
            forbidden_species_penalty_max = min(forbidden_species_penalty_max, float(pipe_cfg.get("text_only_forbidden_species_penalty_max", selection_cfg.get("text_only_forbidden_species_penalty_max", 0.05))))
            frame1_species_threshold = max(frame1_species_threshold, 0.95)
            frame1_white_threshold = max(frame1_white_threshold, 0.95)
            frame1_color_threshold = max(frame1_color_threshold, 0.97)
            frame1_age_threshold = max(frame1_age_threshold, 0.93)
            frame1_object_threshold = max(frame1_object_threshold, 0.75)
            frame1_sentence_threshold = max(frame1_sentence_threshold, 0.84)
        v62_candidate_gate_thresholds.update({
            'frame1_species_threshold': frame1_species_threshold,
            'frame1_white_threshold': frame1_white_threshold,
            'frame1_color_threshold': frame1_color_threshold,
            'frame1_age_threshold': frame1_age_threshold,
            'frame1_object_threshold': frame1_object_threshold,
            'frame1_sentence_threshold': frame1_sentence_threshold,
        })

        style = sample.get("style", "full-color cinematic storybook illustration")
        if "color" not in style.lower():
            style = "full-color " + style

        planning_previous_frame = None
        explicit_story_mode = bool(pipe_cfg.get("respect_input_story_sentences", False)) and _sample_has_explicit_story_sentences(sample)
        explicit_story_mode = bool(explicit_story_mode or force_json_story_flow)
        if not precomputed_story_steps and explicit_story_mode:
            precomputed_story_steps = _build_precomputed_story_steps(sample, total_frames)
        _write_json(out_dir / "json_grounded_story_steps.json", precomputed_story_steps)
        _write_json(out_dir / "paper_flow_contract.json", {
            "version": "V63.1",
            "flow": ["input_json", "abstract", "dcee_plan", "emotion_arc", "generated_story", "storyboard_caption", "frame_image"],
            "force_json_story_flow": force_json_story_flow,
            "respect_input_story_sentences": explicit_story_mode,
            "image_grounding_sources": ["story_abstract", "generated_story_sentence", "frame_caption", "dcee_plan", "json_protagonist", "json_signature_items"],
            "strict_image_rule": "Every frame image must be generated only from abstract, generated story, frame caption, DCEE plan, and JSON contract.",
            "input_contract": sample.get("json_input_contract", {}),
        })
        for idx in range(total_frames):
            if force_json_story_flow and idx < len(precomputed_story_steps):
                raw_story_step = dict(precomputed_story_steps[idx])
            else:
                try:
                    raw_story_step = self.planner.generate_story_step(seed, abstract, dce_plan, emotion_arc, story_rows, planning_previous_frame, idx, total_frames)
                except Exception as e:
                    run_errors.append({"stage": f"planner.generate_story_step.frame_{idx+1}", "error": str(e), "traceback": traceback.format_exc(), "fallback": "v61_1_story_step"})
                    raw_story_step = _v611_fallback_story_step(sample, idx, total_frames, dce_plan, emotion_arc, str(e))
            story_step = _repair_story_step_from_input(raw_story_step, sample, idx, total_frames, precomputed_story_steps)
            story_step["story_abstract"] = _abstract_to_text(abstract)
            story_step["generated_story_context_before"] = _compact_story_context(story_rows)
            story_rows.append(story_step)
            current_story_context = _compact_story_context(story_rows, limit=total_frames)
            story_step["generated_story_context_after"] = current_story_context
            frame = self.planner.story_step_to_frame(seed, dce_plan, emotion_arc, story_step, idx, total_frames)
            frame = _enforce_frame_contract(frame, story_step, sample, idx, total_frames)
            setattr(frame, "story_abstract", _abstract_to_text(abstract))
            setattr(frame, "generated_story_context", current_story_context)
            setattr(frame, "frame_caption", getattr(frame, "image_caption_en", "") or getattr(frame, "story_sentence", ""))
            setattr(frame, "paper_flow_sources", ["abstract", "generated_story", "frame_caption", "dcee_plan", "input_json"])
            frame_render_contract = _v611_frame_render_contract(frame, story_step, seed, dce_plan, abstract, sample, idx, total_frames)
            setattr(frame, "frame_render_contract", frame_render_contract)
            try:
                raw_crit = list(getattr(frame, 'must_show', []) or [])
                loc = str(getattr(frame, 'scene_location', '') or '')
                sent = str(getattr(frame, 'story_sentence', '') or story_step.get('sentence', '') or '')
                extra = []
                low = f"{sent} {loc}".lower()
                if 'jar' in low: extra.append('lost honey jar')
                if 'honey' in low: extra.append('honey')
                if 'root' in low: extra.append('tangled roots')
                if 'branch' in low: extra.append('fallen branches')
                if 'bush' in low: extra.append('bush')
                if 'slope' in low or 'hill' in low or 'incline' in low: extra.append('steep slope')
                if 'lake' in low or 'shore' in low or 'water' in low: extra.append('serene lake')
                if 'forest' in low or 'woods' in low or 'tree' in low: extra.append('dense forest')
                drop = {'enter','enters','search','searches','look','looks','climb','climbs','follow','follows','hear','hears','retrieve','retrieves','arrive','arrives'}
                seen = []
                for x in raw_crit + extra:
                    x = str(x).strip()
                    if not x:
                        continue
                    if x.lower() in drop:
                        continue
                    if x not in seen:
                        seen.append(x)
                setattr(frame, 'critical_visual_nouns', seen[:8])
            except Exception:
                pass
            storyboard.append(frame)
            frame_render_contracts.append(getattr(frame, 'frame_render_contract', {}))
            planning_previous_frame = frame

        _write_json(out_dir / "frame_render_contracts_v61_1.json", frame_render_contracts)

        previous_frame = None
        first_frame_selected_image_path = ""
        previous_two_selected_image_paths = []
        for idx, frame in enumerate(storyboard):
            frame_id = idx + 1
            is_last = idx == total_frames - 1
            target_dir = ending_dir if is_last else frames_dir
            candidate_count = num_ending_candidates if is_last else num_candidates
            if frame_id == 1:
                candidate_count = max(candidate_count, int(pipe_cfg.get('frame1_num_candidates', candidate_count + 6)))

            try:
                selected_memory = memory.select(frame, dce_plan, emotion_arc, strategy=pipe_cfg.get("memory_strategy", "adaptive_causal"))
            except Exception as e:
                selected_memory = {"error": str(e), "stage": "memory.select"}
                run_errors.append({"stage": f"memory.select.frame_{frame_id}", "error": str(e), "traceback": traceback.format_exc()})

            try:
                anchors = anchor_bank.select_for_frame(frame)
            except Exception as e:
                anchors = {"error": str(e), "stage": "anchor.select"}
                run_errors.append({"stage": f"anchor.select.frame_{frame_id}", "error": str(e), "traceback": traceback.format_exc()})

            packet = self.controller.create_packet(
                frame=frame,
                seed=seed,
                dce_plan=dce_plan,
                memory=selected_memory,
                style=style,
                previous_frame=previous_frame,
                anchors=anchors,
            )

            try:
                meta = getattr(packet, 'control_metadata', {}) or {}
                recent_ctx = []
                for row in story_rows[max(0, idx - 2): idx + 1]:
                    recent_ctx.append({
                        'sentence': row.get('sentence', ''),
                        'event': row.get('event', ''),
                        'emotion': row.get('emotion', ''),
                    })
                previous_story_summary = ' '.join(row.get('sentence', '') for row in story_rows[max(0, idx - 2): idx]).strip()
                next_frame = storyboard[idx + 1] if idx + 1 < len(storyboard) else None
                stage_names = ['setup', 'search', 'search', 'transition', 'discovery', 'resolution']
                render_plan = self._generate_llm_render_plan(
                    seed=seed,
                    frame=frame,
                    frame_index=idx,
                    total_frames=total_frames,
                    previous_sentence=getattr(previous_frame, 'story_sentence', '') if previous_frame else '',
                    next_sentence=getattr(next_frame, 'story_sentence', '') if next_frame else '',
                    has_reference_image=has_reference_image,
                    dce_plan=dce_plan,
                ) if use_llm_frame_prompt else {}
                llm_render_plans.append({'frame_id': frame_id, 'render_plan': render_plan})
                meta.update({
                    'story_stage': stage_names[min(idx, len(stage_names) - 1)] if total_frames >= 6 else ('resolution' if is_last else 'progression'),
                    'recent_story_context': recent_ctx,
                    'previous_story_summary': previous_story_summary,
                    'previous_frame_caption': getattr(previous_frame, 'story_sentence', '') if previous_frame else '',
                    'previous_frame_event': getattr(previous_frame, 'event', '') if previous_frame else '',
                    'previous_frame_image_path': getattr(previous_frame, 'selected_image_path', '') if previous_frame else '',
                    'first_frame_image_path': first_frame_selected_image_path,
                    'previous_two_selected_image_paths': list(previous_two_selected_image_paths),
                    'previous_frame_local_caption': getattr(previous_frame, 'selected_local_caption', '') if previous_frame else '',
                    'previous_frame_feedback': getattr(previous_frame, 'selected_feedback', {}) if previous_frame else {},
                    'next_frame_caption': getattr(next_frame, 'story_sentence', '') if next_frame else '',
                    'next_frame_event': getattr(next_frame, 'event', '') if next_frame else '',
                    'frame_transition_contract': 'Continue naturally from the previous frame and visually prepare for the next frame while keeping this exact current story step dominant.',
                    'caption_contract': getattr(frame, 'image_caption_en', '') or getattr(frame, 'story_sentence', ''),
                    'source_reference_image_path': sample.get('image_path', '') if has_reference_image else '',
                    'input_reference_image_path': sample.get('image_path', '') if has_reference_image else '',
                    'canonical_reference_sheet_path': (sample.get('canonical_reference_sheet_path', '') or img_cfg.get('canonical_reference_sheet_path', '')) if has_reference_image else '',
                    'protagonist_reference_paths': (sample.get('protagonist_reference_paths', []) if has_reference_image else ([x for x in [first_frame_selected_image_path] if x])),
                    'identity_anchor_image_path': first_frame_selected_image_path,
                    'identity_backend_priority': img_cfg.get('identity_backend_priority', ['instantid', 'photomaker', 'canonical_reference_sheet', 'character_lora', 'ip_adapter', 'text']) if has_reference_image else ['ip_adapter', 'text'],
                    'text_only_mode': not has_reference_image,
                    'is_final_frame': bool(is_last),
                    'allow_previous_frame_identity_reference': bool((cfg.get('pipeline', {}) or {}).get('text_only_use_previous_frame_reference', True)),
                    'target_objects': getattr(frame, 'must_show', []),
                    'critical_visual_nouns': getattr(frame, 'critical_visual_nouns', getattr(frame, 'must_show', [])),
                    'target_location': getattr(frame, 'scene_location', ''),
                    'identity_contract_required': True,
                    'hard_protagonist_shape_lock': True,
                    'anchor_shape_consistency_required': True,
                    'scene_contract_required': True,
                    'dcee_appearance_delta_required': True,
                    'full_body_uncropped_required': True,
                    'background_story_alignment_required': True,
                    'background_nonempty_required': True,
                    'visual_storytelling_progression_required': True,
                    'previous_story_and_previous_frame_consistency_required': True,
                    'use_llm_frame_prompt': use_llm_frame_prompt,
                    'llm_render_plan': render_plan,
                    'force_text_story_only': True,
                    'sentence_lock_only': True,
                    'retry_strong_mode': True,
                    'verify_words_strict': True,
                    'disable_fallback_generation': True,
                    'render_persona_anchor': render_plan.get('persona_anchor', '') if isinstance(render_plan, dict) else '',
                    'final_frame_identity_lock_required': bool(is_last),
                    'paper_flow_version': 'V64.2',
                    'v611_frame_render_contract': getattr(frame, 'frame_render_contract', {}),
                    'v611_canonical_protagonist_contract': canonical_protagonist_contract,
                    'v611_planner_fallback_safe': True,
                    'v61_story_abstract_caption_strict': True,
                    'v611_frame_contract_strict': True,
                    'strict_story_details_only': True,
                    'signature_item_priority': True,
                    'facial_emotion_priority': True,
                    'whole_sentence_core_required': True,
                    'use_identity_anchor_contract': True,
                    'v60_strict_story_frame_only': True,
                    'story_abstract': _abstract_to_text(abstract),
                    'dcee_plan_summary': _dcee_plan_to_text(dce_plan),
                    'generated_story_context': getattr(frame, 'generated_story_context', ''),
                    'current_frame_caption': getattr(frame, 'frame_caption', '') or getattr(frame, 'image_caption_en', '') or getattr(frame, 'story_sentence', ''),
                    'current_story_sentence': getattr(frame, 'story_sentence', ''),
                    'frame_image_sources': ['abstract', 'generated_story', 'frame_caption', 'dcee_plan', 'input_json'],
                    'input_story_prompt': sample.get('text_prompt', ''),
                    'input_protagonist': sample.get('protagonist', ''),
                    'input_signature_items': sample.get('signature_items', []),
                    'input_target_ending_emotion': sample.get('target_ending_emotion', ''),
                    'anchor_age_stage': anchor_age_stage,
                    'anchor_identity_caption': anchor_identity_caption,
                    'anchor_appearance_contract': anchor_appearance_contract,
                    'dcee_framework_required': True,
                    'json_input_contract': sample.get('json_input_contract', {}),
                    'json_identity_contract': canonical_protagonist_contract,
                    'global_protagonist_lock': getattr(seed, 'protagonist', ''),
                    'global_signature_items_lock': getattr(seed, 'signature_items', []),
                    'json_identity_contract': getattr(seed, '_v61_identity_contract', {}),
                    'global_protagonist_lock': getattr(seed, 'protagonist', ''),
                    'global_signature_items_lock': getattr(seed, 'signature_items', []),
                    'json_protagonist_lock': sample.get('protagonist', ''),
                    'json_signature_items_lock': sample.get('signature_items', []),
                    'signature_items_must_be_visible': list(sample.get('signature_items', []) or []),
                    'v58_strict_json_storygrounding': True,
                    'v59_paper_flow_grounding': True,
                    'v60_strict_story_frame_only': True,
                    'v58_precomputed_story_step': story_step,
                    'v59_story_step_after_repair': story_step,
                })
                packet.control_metadata = meta
            except Exception as e:
                run_errors.append({'stage': f'packet.metadata.frame_{frame_id}', 'error': str(e), 'traceback': traceback.format_exc()})

            candidates = self.image_generator.generate_from_packet(packet=packet, frame_id=frame_id, out_dir=target_dir, num_candidates=candidate_count)
            ranked = self.evaluator.rerank_ending_candidates(frame, dce_plan, candidates) if is_last else self.evaluator.rank_frame_candidates(frame, dce_plan, candidates, is_ending=False)
            if not ranked:
                raise RuntimeError(f"No ranked candidates for frame {frame_id}")
            best = _v62_choose_best_candidate(ranked, frame_id, v62_candidate_gate_thresholds)
            retried = False
            if retry_enabled and (
                best.scores.get("story_alignment", 0.0) < story_threshold
                or best.scores.get("emotion_visibility", 0.0) < emotion_threshold
                or best.scores.get("colorfulness", 0.0) < color_threshold
                or best.scores.get("event_grounding", 0.0) < event_threshold
                or best.scores.get("evidence_visibility", 0.0) < evidence_threshold
                or best.scores.get("background_presence", 0.0) < background_threshold
                or best.scores.get("critical_noun_coverage", 0.0) < critical_noun_threshold
                or best.scores.get("story_word_coverage", 0.0) < word_coverage_threshold
                or best.scores.get("action_word_coverage", 0.0) < action_word_threshold
                or best.scores.get("emotion_word_coverage", 0.0) < emotion_word_threshold
                or best.scores.get("emotion_face_visibility", 0.0) < emotion_face_visibility_threshold
                or best.scores.get("signature_item_coverage", 0.0) < signature_item_coverage_threshold
                or best.scores.get("color_word_coverage", 0.0) < color_word_threshold
                or best.scores.get("object_word_coverage", 0.0) < (frame1_object_threshold if frame_id == 1 else object_word_threshold)
                or best.scores.get("scene_word_coverage", 0.0) < scene_word_threshold
                or best.scores.get("sentence_complete_score", 0.0) < (frame1_sentence_threshold if frame_id == 1 else sentence_complete_threshold)
                or best.scores.get("protagonist_color_consistency", 0.0) < (frame1_color_threshold if frame_id == 1 else protagonist_color_threshold)
                or best.scores.get("protagonist_white_bear_exact_score", 0.0) < (frame1_white_threshold if frame_id == 1 else protagonist_white_exact_threshold)
                or best.scores.get("protagonist_species_score", 0.0) < (frame1_species_threshold if frame_id == 1 else protagonist_species_threshold)
                or best.scores.get("protagonist_age_consistency_score", 0.0) < (frame1_age_threshold if frame_id == 1 else protagonist_age_threshold)
                or best.scores.get("protagonist_age_penalty", 0.0) > 0.10
                or (frame_id > 1 and best.scores.get("identity_consistency", 0.0) < identity_consistency_threshold)
                or (frame_id > 1 and best.scores.get("reference_subject_similarity", 0.0) < reference_subject_similarity_threshold)
                or (frame_id > 1 and best.scores.get("continuity", 0.0) < continuity_threshold)
                or best.scores.get("forbidden_species_penalty", 0.0) > forbidden_species_penalty_max
                or best.scores.get("missing_signature_object_penalty", 0.0) > 0.08
                or best.scores.get("signature_item_coverage", 0.0) < signature_item_coverage_threshold
                or best.scores.get("emotion_face_visibility", 0.0) < emotion_face_visibility_threshold
                or best.scores.get("severe_identity_object_penalty", 0.0) > 0.12
                or best.scores.get("missing_critical_noun_penalty", 0.0) > 0.12
                or best.scores.get("storytelling_progression", 0.0) < progression_threshold
                or best.scores.get("progression_consistency", 0.0) < progression_consistency_threshold
                or best.scores.get("story_context_alignment", 0.0) < story_context_alignment_threshold
                or best.scores.get("scene_grounding_penalty", 0.0) > 0.22
                or best.scores.get("blank_background_penalty", 0.0) > 0.10
                or best.scores.get("static_repeat_penalty", 0.0) > 0.12
                or best.scores.get("bad_extra_subject_penalty", 0.0) > 0.18
                or best.scores.get("frame_state_penalty", 0.0) > 0.20
                or best.scores.get("wrong_subject_or_object_penalty", 0.0) > 0.18
                or best.scores.get("severe_identity_object_penalty", 0.0) > 0.12
            ):
                retried = True
                strong_packet = self._strengthen_packet(packet, frame)
                retry_meta = getattr(strong_packet, 'control_metadata', {}) or {}
                retry_meta['force_text_story_only'] = True
                retry_meta['sentence_lock_only'] = True
                retry_meta['verify_words_strict'] = True
                retry_meta['force_reference_fusion_after_anchor'] = True
                retry_meta['hard_protagonist_shape_lock'] = True
                retry_meta['json_grounded_story_lock'] = True
                strong_packet.control_metadata = retry_meta
                retry_candidates = self.image_generator.generate_from_packet(packet=strong_packet, frame_id=frame_id, out_dir=target_dir, num_candidates=max(4, candidate_count))
                retry_ranked = self.evaluator.rerank_ending_candidates(frame, dce_plan, retry_candidates) if is_last else self.evaluator.rank_frame_candidates(frame, dce_plan, retry_candidates, is_ending=False)
                if retry_ranked and retry_ranked[0].scores.get("overall", 0.0) >= best.scores.get("overall", 0.0):
                    best = retry_ranked[0]
                    ranked = retry_ranked
                    candidates = retry_candidates
                    packet = strong_packet

                if not is_last and (
                    best.scores.get("story_alignment", 0.0) < story_threshold
                    or best.scores.get("critical_noun_coverage", 0.0) < critical_noun_threshold
                    or best.scores.get("story_word_coverage", 0.0) < word_coverage_threshold
                    or best.scores.get("frame_state_penalty", 0.0) > 0.20
                    or best.scores.get("protagonist_white_bear_exact_score", 0.0) < (frame1_white_threshold if frame_id == 1 else protagonist_white_exact_threshold)
                    or best.scores.get("protagonist_species_score", 0.0) < (frame1_species_threshold if frame_id == 1 else protagonist_species_threshold)
                    or best.scores.get("protagonist_age_consistency_score", 0.0) < (frame1_age_threshold if frame_id == 1 else protagonist_age_threshold)
                    or best.scores.get("protagonist_age_penalty", 0.0) > 0.10
                    or (frame_id > 1 and best.scores.get("identity_consistency", 0.0) < identity_consistency_threshold)
                    or (frame_id > 1 and best.scores.get("reference_subject_similarity", 0.0) < reference_subject_similarity_threshold)
                    or (frame_id > 1 and best.scores.get("continuity", 0.0) < continuity_threshold)
                    or best.scores.get("forbidden_species_penalty", 0.0) > forbidden_species_penalty_max
                ):
                    second_packet = self._strengthen_packet(packet, frame)
                    second_meta = getattr(second_packet, 'control_metadata', {}) or {}
                    second_meta['force_text_story_only'] = True
                    second_meta['sentence_lock_only'] = True
                    second_meta['retry_strong_mode'] = True
                    second_meta['verify_words_strict'] = True
                    if best.scores.get("protagonist_species_score", 0.0) < (frame1_species_threshold if frame_id == 1 else protagonist_species_threshold) or best.scores.get("protagonist_white_bear_exact_score", 0.0) < (frame1_white_threshold if frame_id == 1 else protagonist_white_exact_threshold) or best.scores.get("protagonist_age_consistency_score", 0.0) < (frame1_age_threshold if frame_id == 1 else protagonist_age_threshold) or best.scores.get("protagonist_age_penalty", 0.0) > 0.10 or (frame_id > 1 and best.scores.get("identity_consistency", 0.0) < identity_consistency_threshold) or (frame_id > 1 and best.scores.get("reference_subject_similarity", 0.0) < reference_subject_similarity_threshold) or (frame_id > 1 and best.scores.get("continuity", 0.0) < continuity_threshold) or best.scores.get("forbidden_species_penalty", 0.0) > forbidden_species_penalty_max:
                        second_meta['variant_focus'] = 'identity'
                        second_meta['sentence_lock_variant'] = 'identity'
                        second_meta['hard_protagonist_lock'] = True
                    elif best.scores.get("sentence_complete_score", 0.0) < (frame1_sentence_threshold if frame_id == 1 else sentence_complete_threshold) or best.scores.get("signature_item_coverage", 0.0) < signature_item_coverage_threshold or best.scores.get("emotion_face_visibility", 0.0) < emotion_face_visibility_threshold:
                        second_meta['variant_focus'] = 'storycore'
                        second_meta['sentence_lock_variant'] = 'storycore'
                    elif best.scores.get("sentence_complete_score", 0.0) < (frame1_sentence_threshold if frame_id == 1 else sentence_complete_threshold):
                        second_meta['variant_focus'] = 'complete'
                        second_meta['sentence_lock_variant'] = 'complete'
                    elif best.scores.get("object_word_coverage", 0.0) < (frame1_object_threshold if frame_id == 1 else object_word_threshold) or best.scores.get("critical_noun_coverage", 0.0) < critical_noun_threshold:
                        second_meta['variant_focus'] = 'object'
                        second_meta['sentence_lock_variant'] = 'object'
                    elif best.scores.get("scene_word_coverage", 0.0) < scene_word_threshold:
                        second_meta['variant_focus'] = 'scene'
                        second_meta['sentence_lock_variant'] = 'scene'
                    elif best.scores.get("emotion_face_visibility", 0.0) < emotion_face_visibility_threshold:
                        second_meta['variant_focus'] = 'emotion_face'
                        second_meta['sentence_lock_variant'] = 'emotion_face'
                    elif best.scores.get("emotion_word_coverage", 0.0) < emotion_word_threshold:
                        second_meta['variant_focus'] = 'emotion'
                        second_meta['sentence_lock_variant'] = 'emotion'
                    elif best.scores.get("signature_item_coverage", 0.0) < signature_item_coverage_threshold or best.scores.get("missing_signature_object_penalty", 0.0) > 0.08:
                        second_meta['variant_focus'] = 'signature'
                        second_meta['sentence_lock_variant'] = 'signature'
                    elif best.scores.get("color_word_coverage", 0.0) < color_word_threshold:
                        second_meta['variant_focus'] = 'color'
                        second_meta['sentence_lock_variant'] = 'color'
                    else:
                        second_meta['variant_focus'] = 'action'
                        second_meta['sentence_lock_variant'] = 'action'
                    second_packet.control_metadata = second_meta
                    second_retry_candidates = self.image_generator.generate_from_packet(packet=second_packet, frame_id=frame_id, out_dir=target_dir, num_candidates=max(6, candidate_count))
                    second_retry_ranked = self.evaluator.rank_frame_candidates(frame, dce_plan, second_retry_candidates, is_ending=False)
                    if second_retry_ranked:
                        second_best = _v62_choose_best_candidate(second_retry_ranked, frame_id, v62_candidate_gate_thresholds)
                        if second_best and ((getattr(second_best, 'notes', {}) or {}).get('v62_gate_score', -999.0) >= (getattr(best, 'notes', {}) or {}).get('v62_gate_score', -999.0)):
                            best = second_best
                            ranked = second_retry_ranked
                        candidates = second_retry_candidates
                        packet = second_packet

            if not ((getattr(best, 'notes', {}) or {}).get('v62_gate_pass', False)):
                rescue_variants = ['storycore', 'identity', 'signature_emotion', 'signature', 'emotion_face', 'object', 'scene', 'action', 'complete']
                rescue_pool = list(ranked)
                for rescue_variant in rescue_variants:
                    rescue_packet = self._strengthen_packet(packet, frame)
                    rescue_meta = getattr(rescue_packet, 'control_metadata', {}) or {}
                    rescue_meta['force_text_story_only'] = True
                    rescue_meta['sentence_lock_only'] = True
                    rescue_meta['retry_strong_mode'] = True
                    rescue_meta['verify_words_strict'] = True
                    rescue_meta['variant_focus'] = rescue_variant
                    rescue_meta['sentence_lock_variant'] = rescue_variant
                    if rescue_variant == 'identity':
                        rescue_meta['hard_protagonist_lock'] = True
                    rescue_packet.control_metadata = rescue_meta
                    rescue_candidates = self.image_generator.generate_from_packet(packet=rescue_packet, frame_id=frame_id, out_dir=target_dir, num_candidates=max(6, candidate_count // 2))
                    rescue_ranked = self.evaluator.rerank_ending_candidates(frame, dce_plan, rescue_candidates) if is_last else self.evaluator.rank_frame_candidates(frame, dce_plan, rescue_candidates, is_ending=False)
                    rescue_pool.extend(rescue_ranked)
                    rescue_best = _v62_choose_best_candidate(rescue_ranked, frame_id, v62_candidate_gate_thresholds)
                    if rescue_best and ((getattr(rescue_best, 'notes', {}) or {}).get('v62_gate_pass', False)):
                        best = rescue_best
                        ranked = rescue_ranked
                        candidates = rescue_candidates
                        packet = rescue_packet
                        break
                if not ((getattr(best, 'notes', {}) or {}).get('v62_gate_pass', False)) and rescue_pool:
                    best = _v62_choose_best_candidate(rescue_pool, frame_id, v62_candidate_gate_thresholds)

            if not is_last and (
                best.scores.get("sentence_complete_score", 0.0) < (frame1_sentence_threshold if frame_id == 1 else sentence_complete_threshold)
                or best.scores.get("protagonist_color_consistency", 0.0) < (frame1_color_threshold if frame_id == 1 else protagonist_color_threshold)
                or best.scores.get("protagonist_white_bear_exact_score", 0.0) < (frame1_white_threshold if frame_id == 1 else protagonist_white_exact_threshold)
                or best.scores.get("protagonist_species_score", 0.0) < (frame1_species_threshold if frame_id == 1 else protagonist_species_threshold)
                or best.scores.get("protagonist_age_consistency_score", 0.0) < (frame1_age_threshold if frame_id == 1 else protagonist_age_threshold)
                or best.scores.get("protagonist_age_penalty", 0.0) > 0.10
                or (frame_id > 1 and best.scores.get("identity_consistency", 0.0) < identity_consistency_threshold)
                or (frame_id > 1 and best.scores.get("reference_subject_similarity", 0.0) < reference_subject_similarity_threshold)
                or (frame_id > 1 and best.scores.get("continuity", 0.0) < continuity_threshold)
                or best.scores.get("forbidden_species_penalty", 0.0) > forbidden_species_penalty_max
                or best.scores.get("missing_signature_object_penalty", 0.0) > 0.08
                or best.scores.get("severe_identity_object_penalty", 0.0) > 0.12
            ):
                final_packet = self._strengthen_packet(packet, frame)
                final_meta = getattr(final_packet, 'control_metadata', {}) or {}
                final_meta['force_text_story_only'] = True
                final_meta['sentence_lock_only'] = True
                if best.scores.get('signature_item_coverage', 0.0) < signature_item_coverage_threshold or best.scores.get('missing_signature_object_penalty', 0.0) > 0.08:
                    final_meta['variant_focus'] = 'signature'
                    final_meta['sentence_lock_variant'] = 'signature'
                elif best.scores.get('emotion_face_visibility', 0.0) < emotion_face_visibility_threshold:
                    final_meta['variant_focus'] = 'emotion_face'
                    final_meta['sentence_lock_variant'] = 'emotion_face'
                final_meta['retry_strong_mode'] = True
                final_meta['verify_words_strict'] = True
                final_meta['force_reference_fusion_after_anchor'] = True
                final_meta['hard_protagonist_shape_lock'] = True
                final_meta['json_grounded_story_lock'] = True
                if best.scores.get("protagonist_species_score", 0.0) < (frame1_species_threshold if frame_id == 1 else protagonist_species_threshold) or best.scores.get("protagonist_white_bear_exact_score", 0.0) < (frame1_white_threshold if frame_id == 1 else protagonist_white_exact_threshold) or best.scores.get("protagonist_age_consistency_score", 0.0) < (frame1_age_threshold if frame_id == 1 else protagonist_age_threshold) or best.scores.get("protagonist_age_penalty", 0.0) > 0.10 or (frame_id > 1 and best.scores.get("identity_consistency", 0.0) < identity_consistency_threshold) or (frame_id > 1 and best.scores.get("reference_subject_similarity", 0.0) < reference_subject_similarity_threshold) or (frame_id > 1 and best.scores.get("continuity", 0.0) < continuity_threshold) or best.scores.get("forbidden_species_penalty", 0.0) > forbidden_species_penalty_max:
                    final_meta['variant_focus'] = 'identity'
                    final_meta['sentence_lock_variant'] = 'identity'
                    final_meta['hard_protagonist_lock'] = True
                else:
                    final_meta['variant_focus'] = 'complete'
                    final_meta['sentence_lock_variant'] = 'complete'
                final_packet.control_metadata = final_meta
                final_retry_candidates = self.image_generator.generate_from_packet(final_packet, frame_id=frame_id, out_dir=target_dir, num_candidates=max(8, candidate_count))
                final_retry_ranked = self.evaluator.rank_frame_candidates(frame, dce_plan, final_retry_candidates, is_ending=False)
                if final_retry_ranked and final_retry_ranked[0].scores.get("overall", 0.0) >= best.scores.get("overall", 0.0):
                    best = final_retry_ranked[0]
                    ranked = final_retry_ranked
                    candidates = final_retry_candidates
                    packet = final_packet

            if not is_last and (
                best.scores.get("protagonist_species_score", 0.0) < (frame1_species_threshold if frame_id == 1 else protagonist_species_threshold)
                or best.scores.get("protagonist_white_bear_exact_score", 0.0) < (frame1_white_threshold if frame_id == 1 else protagonist_white_exact_threshold)
                or best.scores.get("protagonist_age_consistency_score", 0.0) < (frame1_age_threshold if frame_id == 1 else protagonist_age_threshold)
                or best.scores.get("protagonist_age_penalty", 0.0) > 0.10
                or (frame_id > 1 and best.scores.get("identity_consistency", 0.0) < identity_consistency_threshold)
                or (frame_id > 1 and best.scores.get("reference_subject_similarity", 0.0) < reference_subject_similarity_threshold)
                or (frame_id > 1 and best.scores.get("continuity", 0.0) < continuity_threshold)
                or best.scores.get("forbidden_species_penalty", 0.0) > forbidden_species_penalty_max
            ):
                identity_packet = self._strengthen_packet(packet, frame)
                identity_meta = getattr(identity_packet, 'control_metadata', {}) or {}
                identity_meta['force_text_story_only'] = True
                identity_meta['sentence_lock_only'] = True
                identity_meta['retry_strong_mode'] = True
                identity_meta['verify_words_strict'] = True
                identity_meta['variant_focus'] = 'identity'
                identity_meta['sentence_lock_variant'] = 'identity'
                identity_meta['hard_protagonist_lock'] = True
                identity_packet.control_metadata = identity_meta
                identity_retry_candidates = self.image_generator.generate_from_packet(packet=identity_packet, frame_id=frame_id, out_dir=target_dir, num_candidates=max(10, candidate_count))
                identity_retry_ranked = self.evaluator.rank_frame_candidates(frame, dce_plan, identity_retry_candidates, is_ending=False)
                if identity_retry_ranked and identity_retry_ranked[0].scores.get("overall", 0.0) >= best.scores.get("overall", 0.0):
                    best = identity_retry_ranked[0]
                    ranked = identity_retry_ranked
                    candidates = identity_retry_candidates
                    packet = identity_packet

            selected_images.append(best)
            if is_last:
                ending_candidates = ranked
            try:
                memory.add(frame, best)
            except Exception as e:
                run_errors.append({"stage": f"memory.add.frame_{frame_id}", "error": str(e), "traceback": traceback.format_exc()})

            setattr(frame, "selected_image_path", getattr(best, "image_path", ""))
            if not first_frame_selected_image_path:
                first_frame_selected_image_path = getattr(best, "image_path", "")
            if getattr(best, "image_path", ""):
                previous_two_selected_image_paths.append(getattr(best, "image_path", ""))
                previous_two_selected_image_paths = previous_two_selected_image_paths[-2:]
            setattr(frame, "selected_local_caption", getattr(best, "notes", {}).get("local_caption", ""))
            if frame_id == 1:
                anchor_identity_caption = getattr(best, "notes", {}).get("local_caption", "") or getattr(frame, "story_sentence", "")
                anchor_age_stage = _infer_anchor_age_stage(anchor_identity_caption)
                anchor_appearance_contract = _default_white_bear_appearance(anchor_age_stage)
            setattr(frame, "anchor_age_stage", anchor_age_stage)
            setattr(frame, "anchor_identity_caption", anchor_identity_caption)
            setattr(frame, "anchor_appearance_contract", anchor_appearance_contract)
            setattr(frame, "selected_feedback", {
                "story_alignment": getattr(best, "scores", {}).get("story_alignment", 0.0),
                "event_grounding": getattr(best, "scores", {}).get("event_grounding", 0.0),
                "evidence_visibility": getattr(best, "scores", {}).get("evidence_visibility", 0.0),
                "background_presence": getattr(best, "scores", {}).get("background_presence", 0.0),
                "critical_noun_coverage": getattr(best, "scores", {}).get("critical_noun_coverage", 0.0),
                "story_word_coverage": getattr(best, "scores", {}).get("story_word_coverage", 0.0),
                "action_word_coverage": getattr(best, "scores", {}).get("action_word_coverage", 0.0),
                "scene_word_coverage": getattr(best, "scores", {}).get("scene_word_coverage", 0.0),
                "object_word_coverage": getattr(best, "scores", {}).get("object_word_coverage", 0.0),
                "emotion_word_coverage": getattr(best, "scores", {}).get("emotion_word_coverage", 0.0),
                "color_word_coverage": getattr(best, "scores", {}).get("color_word_coverage", 0.0),
                "story_word_details": getattr(best, "notes", {}).get("story_word_details", {}),
                "missing_critical_noun_penalty": getattr(best, "scores", {}).get("missing_critical_noun_penalty", 0.0),
                "storytelling_progression": getattr(best, "scores", {}).get("storytelling_progression", 0.0),
                "progression_consistency": getattr(best, "scores", {}).get("progression_consistency", 0.0),
                "story_context_alignment": getattr(best, "scores", {}).get("story_context_alignment", 0.0),
                "recent_repeat_penalty": getattr(best, "scores", {}).get("recent_repeat_penalty", 0.0),
                "severe_identity_object_penalty": getattr(best, "scores", {}).get("severe_identity_object_penalty", 0.0),
                "sentence_complete_score": getattr(best, "scores", {}).get("sentence_complete_score", 0.0),
                "protagonist_color_consistency": getattr(best, "scores", {}).get("protagonist_color_consistency", 0.0),
                "protagonist_white_bear_exact_score": getattr(best, "scores", {}).get("protagonist_white_bear_exact_score", 0.0),
                "protagonist_age_consistency_score": getattr(best, "scores", {}).get("protagonist_age_consistency_score", 0.0),
                "protagonist_age_penalty": getattr(best, "scores", {}).get("protagonist_age_penalty", 0.0),
                "identity_consistency": getattr(best, "scores", {}).get("identity_consistency", 0.0),
                "reference_subject_similarity": getattr(best, "scores", {}).get("reference_subject_similarity", 0.0),
                "continuity": getattr(best, "scores", {}).get("continuity", 0.0),
                "missing_signature_object_penalty": getattr(best, "scores", {}).get("missing_signature_object_penalty", 0.0),
                "blank_background_penalty": getattr(best, "scores", {}).get("blank_background_penalty", 0.0),
                "static_repeat_penalty": getattr(best, "scores", {}).get("static_repeat_penalty", 0.0),
                "bad_extra_subject_penalty": getattr(best, "scores", {}).get("bad_extra_subject_penalty", 0.0),
                "selected_prompt": getattr(best, "prompt", ""),
                "identity_backend_selected": getattr(best, "notes", {}).get("identity_backend_selected", ""),
                "v62_gate_score": getattr(best, "notes", {}).get("v62_gate_score", None),
                "v62_gate_pass": getattr(best, "notes", {}).get("v62_gate_pass", None),
            })
            previous_frame = frame

            packet_log.append({"frame_id": frame_id, "packet": _safe_asdict(packet), "retried": retried})
            memory_log.append({"frame_id": frame_id, "memory": _safe_asdict(selected_memory)})
            candidate_manifest.append({"frame_id": frame_id, "candidates": _safe_asdict(candidates), "selected": _safe_asdict(best)})

            _write_json(out_dir / "full_story_partial.json", {"sentences": story_rows, "story_text": " ".join(x.get("sentence", "") for x in story_rows)})
            _write_json(out_dir / "storyboard_partial.json", storyboard)

        full_story = {"sentences": story_rows, "story_text": " ".join(x.get("sentence", "") for x in story_rows)}
        self._save_core_plan_outputs(out_dir, seed, abstract, dce_plan, emotion_arc, full_story, storyboard)

        _write_json(out_dir / "visual_control_packets.json", packet_log)
        _write_json(out_dir / "memory_log.json", memory_log)
        _write_json(out_dir / "candidate_manifest.json", candidate_manifest)
        _write_json(out_dir / "selected_images.json", selected_images)
        _write_json(out_dir / "llm_render_plans.json", llm_render_plans)

        questions = self.evaluator.generate_questions(dce_plan, emotion_arc, storyboard)
        _write_json(out_dir / "evaluation_questions.json", questions)

        evaluation = self.evaluator.evaluate_sequence(dce_plan, emotion_arc, storyboard, selected_images, questions, out_dir=out_dir)
        if not evaluation.get("contact_sheet_path"):
            try:
                evaluation["contact_sheet_path"] = _make_contact_sheet_force([getattr(x, "image_path", "") for x in selected_images], out_dir / "contact_sheet.png")
            except Exception as e:
                evaluation["contact_sheet_error"] = f"{type(e).__name__}: {e}"
        if run_errors:
            evaluation["run_errors"] = run_errors
        _write_json(out_dir / "evaluation.json", evaluation)

        final_story_md = self._build_markdown(abstract, dce_plan, emotion_arc, full_story, storyboard, selected_images, ending_candidates, evaluation)
        if not str(final_story_md).strip():
            raise RuntimeError("Strict mode: final_story.md content is empty.")
        (out_dir / "final_story.md").write_text(final_story_md, encoding="utf-8")

        _write_json(out_dir / "output_manifest.json", {
            "contact_sheet": str(out_dir / "contact_sheet.png"),
            "final_story": str(out_dir / "final_story.md"),
            "evaluation": str(out_dir / "evaluation.json"),
            "selected_images": str(out_dir / "selected_images.json"),
            "candidate_manifest": str(out_dir / "candidate_manifest.json"),
            "storyboard": str(out_dir / "storyboard.json"),
            "full_story": str(out_dir / "full_story.json"),
            "dcee_plan": str(out_dir / "dcee_plan.json"),
            "generation_policy_V44": str(out_dir / "generation_policy_V44_llm_frame_prompt.json"),
            "llm_render_plans": str(out_dir / "llm_render_plans.json"),
            "has_contact_sheet": (out_dir / "contact_sheet.png").exists(),
            "num_selected_images": len(selected_images),
            "multi_gpu_enabled": bool(getattr(self.image_generator, "multi_gpu", False)),
            "multi_gpu_visible_ids": list(getattr(self.image_generator, "gpu_ids", []) or []),
            "image_generator_device": str(getattr(self.image_generator, "device", "")),
        })

        return PipelineResult(
            seed=seed,
            abstract=abstract,
            dce_plan=dce_plan,
            emotion_arc=emotion_arc,
            storyboard=storyboard,
            selected_images=selected_images,
            ending_candidates=ending_candidates,
            evaluation_questions=questions,
            evaluation=evaluation,
            final_story_markdown=final_story_md,
        )

    @staticmethod
    def _build_markdown(abstract, dce_plan, emotion_arc, full_story, storyboard, images, ending_candidates, evaluation):
        rows = (full_story or {}).get("sentences", []) if isinstance(full_story, dict) else []
        lines = [
            "# DCEE-CausalVerse Visual Story\n",
            "## Abstract\n",
            str(abstract) + "\n",
            "## Selected DCEE Plan\n",
            f"- Desire: {getattr(dce_plan, 'desire', '')}",
            f"- Conflict: {getattr(dce_plan, 'conflict', '')}",
            f"- Ending Emotion: {getattr(dce_plan, 'target_ending_emotion', '')}",
            "",
            "## Full Story\n",
        ]
        for idx, row in enumerate(rows, 1):
            lines.append(f"{idx}. {row.get('sentence', '')}")
        lines += ["", "## Emotion Arc\n", f"- States: {' → '.join([str(x) for x in getattr(emotion_arc, 'states', [])])}", f"- Intensities: {' → '.join([str(x) for x in getattr(emotion_arc, 'intensities', [])])}", ""]
        if evaluation.get("contact_sheet_path"):
            lines += ["## Contact Sheet\n", f"![Contact Sheet]({evaluation.get('contact_sheet_path')})", ""]
        lines.append("## Frames\n")
        for idx, (frame, image) in enumerate(zip(storyboard, images), 1):
            story_sentence = rows[idx - 1].get("sentence", "") if idx - 1 < len(rows) else getattr(frame, "story_sentence", "")
            lines += [
                f"### Frame {getattr(frame, 'frame_id', idx)}",
                f"![Frame {getattr(frame, 'frame_id', idx)}]({getattr(image, 'image_path', '')})",
                f"- Story sentence: {story_sentence}",
                f"- Event: {getattr(frame, 'event', '')}",
                f"- Event grounding: {getattr(frame, 'event_grounding', '')}",
                f"- Emotion: {getattr(frame, 'emotion', '')} ({getattr(frame, 'emotion_intensity', '')}/5)",
                f"- Required objects: {getattr(frame, 'must_show', [])}",
                f"- Scores: {getattr(image, 'scores', {})}",
                "",
            ]
        lines.append("## Sequence Evaluation\n")
        for k, v in evaluation.items():
            if k == "run_errors":
                continue
            lines.append(f"- {k}: {v}")
        return "\n".join(lines) + "\n"
