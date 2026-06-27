from __future__ import annotations

from typing import Any, Dict, List
import re

_STOP = {
    'the','a','an','this','that','these','those','his','her','their','its','our','your','my',
    'and','or','but','then','after','before','while','when','where','with','without','into','onto',
    'from','over','under','near','beside','behind','through','across','around','inside','outside',
    'looks','look','looks','felt','feels','feeling','is','are','was','were','be','being','been',
    'walks','walked','runs','ran','holds','held','holding','sees','saw','finds','found','tries','tried',
    'because','there','here','very','more','most','just','still','again','only'
}


def clean_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "").replace("\n", " ")).strip()


def as_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [clean_text(v) for v in x if clean_text(v)]
    if isinstance(x, dict):
        out = []
        for v in x.values():
            out.extend(as_list(v))
        return out
    s = clean_text(x)
    return [s] if s else []


def unique(items: List[str], limit: int | None = None) -> List[str]:
    out = []
    for item in items:
        t = clean_text(item)
        if t and t not in out:
            out.append(t)
    return out if limit is None else out[:limit]


def extract_entity_candidates(text: Any) -> List[str]:
    s = clean_text(text)
    if not s:
        return []
    out: List[str] = []
    # quoted or article-led noun-ish phrases
    for m in re.findall(r'(?:the|a|an|his|her|their|its)\s+([A-Za-z][A-Za-z\-]*(?:\s+[A-Za-z][A-Za-z\-]*){0,2})', s):
        out.append(m)
    # important single tokens
    for tok in re.findall(r"[A-Za-z][A-Za-z\-']+", s):
        low = tok.lower()
        if len(low) < 4 or low in _STOP:
            continue
        out.append(tok)
    return unique(out, 10)


def _frame_row(full_story: Dict[str, Any], idx: int) -> Dict[str, Any]:
    rows = (full_story or {}).get('sentences', []) if isinstance(full_story, dict) else []
    if 0 <= idx < len(rows) and isinstance(rows[idx], dict):
        return rows[idx]
    return {}


def build_story_graph(full_story: Dict[str, Any], storyboard: List[Any], seed: Any = None, dce_plan: Any = None) -> Dict[str, Any]:
    rows = (full_story or {}).get('sentences', []) if isinstance(full_story, dict) else []
    frames: List[Dict[str, Any]] = []
    freq: Dict[str, int] = {}
    global_background = []
    setting = clean_text(getattr(seed, 'setting', '') if seed is not None else '')
    if setting:
        global_background.append(setting)
    wc = getattr(seed, 'world_context', {}) if seed is not None else {}
    if isinstance(wc, dict):
        global_background.extend(as_list(wc.get('setting')) + as_list(wc.get('background')) + as_list(wc.get('environment')))

    for idx, frame in enumerate(storyboard or []):
        row = _frame_row(full_story, idx)
        entities = unique(
            as_list(row.get('subject'))
            + as_list(row.get('object'))
            + as_list(row.get('required_objects'))
            + extract_entity_candidates(row.get('sentence'))
            + extract_entity_candidates(row.get('image_sentence'))
            + as_list(getattr(frame, 'key_objects', []))
            + as_list(getattr(frame, 'evidence_objects', []))
            + as_list(getattr(frame, 'emotion_evidence', [])),
            12,
        )
        for ent in entities:
            freq[ent] = freq.get(ent, 0) + 1
        location = clean_text(row.get('location') or getattr(frame, 'scene_location', '') or setting)
        weather = clean_text(getattr(frame, 'weather', ''))
        atmosphere = clean_text(getattr(frame, 'atmosphere', ''))
        frames.append({
            'frame_id': idx + 1,
            'sentence': clean_text(row.get('sentence') or getattr(frame, 'story_sentence', '')),
            'image_sentence': clean_text(row.get('image_sentence') or row.get('sentence') or getattr(frame, 'story_sentence', '')),
            'event': clean_text(getattr(frame, 'event', '') or row.get('action') or row.get('event')),
            'emotion': clean_text(getattr(frame, 'emotion', '') or row.get('emotion')),
            'entities': entities,
            'location': location,
            'weather': weather,
            'atmosphere': atmosphere,
            'must_show': unique(as_list(getattr(frame, 'must_show', [])) + entities, 14),
        })

    recurring = [k for k, v in freq.items() if v >= 2]
    for fr in frames:
        prev_ents = frames[fr['frame_id'] - 2]['entities'] if fr['frame_id'] > 1 else []
        fr['carry_over_entities'] = [x for x in fr['entities'] if x in prev_ents]
        fr['recurring_entities'] = [x for x in fr['entities'] if x in recurring]
        fr['world_cues'] = unique(as_list(fr.get('location')) + as_list(fr.get('weather')) + as_list(fr.get('atmosphere')) + global_background, 8)

    protagonist = clean_text(getattr(seed, 'protagonist', '') if seed is not None else '') or 'protagonist'
    return {
        'protagonist': protagonist,
        'global_background': unique(global_background, 8),
        'recurring_entities': recurring,
        'frames': frames,
    }


def get_frame_graph_hints(story_graph: Dict[str, Any] | None, frame_index: int, story_sentence: str = '') -> Dict[str, Any]:
    graph = story_graph or {}
    frames = graph.get('frames', []) if isinstance(graph, dict) else []
    if 0 <= frame_index < len(frames):
        fr = frames[frame_index]
        return {
            'current_entities': unique(fr.get('entities', []), 10),
            'carry_over_entities': unique(fr.get('carry_over_entities', []), 6),
            'recurring_entities': unique(fr.get('recurring_entities', []), 6),
            'world_cues': unique(fr.get('world_cues', []), 6),
            'location': clean_text(fr.get('location', '')),
            'weather': clean_text(fr.get('weather', '')),
            'atmosphere': clean_text(fr.get('atmosphere', '')),
        }
    ents = extract_entity_candidates(story_sentence)
    return {
        'current_entities': ents,
        'carry_over_entities': [],
        'recurring_entities': [],
        'world_cues': unique(as_list(graph.get('global_background', [])), 6),
        'location': '',
        'weather': '',
        'atmosphere': '',
    }
