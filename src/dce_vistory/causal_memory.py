from __future__ import annotations
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List


def _txt(x):
    if x is None:
        return ''
    if isinstance(x, (list, tuple)):
        return ' '.join(_txt(v) for v in x)
    if isinstance(x, dict):
        return ' '.join(_txt(v) for v in x.values())
    if is_dataclass(x):
        return _txt(asdict(x))
    return str(x)


def _tokens(x):
    return set(t.lower().strip('.,;:!?()[]{}') for t in _txt(x).split() if len(t) > 2)


def _jaccard(a, b):
    A = _tokens(a)
    B = _tokens(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


class DCEECausalMemoryStore:
    """
    Multi-history causal memory store.

    Improvements over a single-history selector:
    - selects multiple relevant histories at once
    - keeps entity/world continuity summaries
    - exposes reference image paths from salient history to reduce auto-regressive drift
    """

    def __init__(self, max_items: int = 12):
        self.max_items = max_items
        self.items: List[Dict[str, Any]] = []
        self.sink: Dict[str, Any] = {}

    def initialize(self, seed: Any, dce_plan: Any):
        self.sink = {
            'identity': getattr(seed, 'character_profiles', []),
            'protagonist': getattr(seed, 'protagonist', ''),
            'setting': getattr(seed, 'setting', ''),
            'desire': getattr(dce_plan, 'desire', ''),
            'conflict': getattr(dce_plan, 'conflict', ''),
            'target_ending_emotion': getattr(dce_plan, 'target_ending_emotion', ''),
            'event_chain': getattr(dce_plan, 'event_chain', getattr(dce_plan, 'event_spine', [])),
            'world_context': getattr(seed, 'world_context', {}),
            'global_visual_symbols': getattr(seed, 'visual_symbols', {}),
        }

    def _continuity_constraints(self, frame: Any, selected: List[Dict[str, Any]]) -> Dict[str, Any]:
        protagonist = self.sink.get('protagonist', 'protagonist')
        must_keep = []
        for m in selected:
            must_keep.extend(m.get('key_objects', []) or [])
        must_keep.extend(list(getattr(frame, 'key_objects', []) or [])[:3])
        must_keep.extend(list(getattr(frame, 'evidence_objects', []) or [])[:3])
        stable_scene = []
        for m in selected:
            loc = m.get('scene_location', '')
            if loc:
                stable_scene.append(loc)
        return {
            'same_protagonist': protagonist,
            'preserve_age_gender_face_hair_outfit': True,
            'must_keep_objects_if_relevant': list(dict.fromkeys([x for x in must_keep if x]))[:8],
            'target_emotion': getattr(frame, 'emotion', ''),
            'scene_location': getattr(frame, 'scene_location', ''),
            'weather': getattr(frame, 'weather', ''),
            'stable_scene_memory': list(dict.fromkeys(stable_scene))[:3],
        }

    def _summarize_selected(self, selected: List[Dict[str, Any]]) -> str:
        if not selected:
            return 'No prior visual history. Introduce the first clear scene of the story.'
        parts = []
        for m in selected[:4]:
            parts.append(
                f"frame {m.get('frame_id')}: story={m.get('story_sentence','')}; event={m.get('event','')}; emotion={m.get('emotion','')}; key_objects={m.get('key_objects',[])}"
            )
        return ' | '.join(parts)

    def select(self, frame: Any, dce_plan: Any, emotion_arc: Any, strategy: str = 'adaptive_causal', top_k: int = 4) -> Dict[str, Any]:
        query = {
            'event': getattr(frame, 'event', ''),
            'story_sentence': getattr(frame, 'story_sentence', ''),
            'emotion': getattr(frame, 'emotion', ''),
            'event_grounding': getattr(frame, 'event_grounding', ''),
            'evidence': getattr(frame, 'emotion_evidence', []),
            'key_objects': getattr(frame, 'key_objects', []),
            'must_show': getattr(frame, 'must_show', []),
            'scene_location': getattr(frame, 'scene_location', ''),
            'weather': getattr(frame, 'weather', ''),
            'conflict': getattr(dce_plan, 'conflict', ''),
        }
        scored = []
        total = max(1, len(self.items))
        for idx, item in enumerate(self.items):
            recency = (idx + 1) / total
            score = (
                0.22 * _jaccard(query.get('story_sentence'), item.get('story_sentence'))
                + 0.16 * _jaccard(query.get('event'), item.get('event'))
                + 0.12 * _jaccard(query.get('event_grounding'), item.get('event_grounding'))
                + 0.12 * _jaccard(query.get('evidence'), item.get('evidence'))
                + 0.14 * _jaccard(query.get('key_objects'), item.get('key_objects'))
                + 0.10 * _jaccard(query.get('must_show'), item.get('key_objects'))
                + 0.08 * _jaccard(query.get('scene_location'), item.get('scene_location'))
                + 0.02 * _jaccard(query.get('weather'), item.get('weather'))
                + 0.04 * _jaccard(query.get('emotion'), item.get('emotion'))
                + 0.10 * recency
            )
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [dict(score=round(s, 4), **m) for s, m in scored[:top_k]]
        ref_images = [m.get('image_path', '') for m in selected if m.get('image_path')]
        entity_memory = []
        world_memory = []
        for m in selected:
            entity_memory.extend(m.get('key_objects', []) or [])
            if m.get('scene_location'):
                world_memory.append(m.get('scene_location'))
            if m.get('weather'):
                world_memory.append(m.get('weather'))
        return {
            'causal_sink': self.sink,
            'selected_memories': selected,
            'salient_history': self._summarize_selected(selected),
            'multi_history_summary': self._summarize_selected(selected),
            'continuity_constraints': self._continuity_constraints(frame, selected),
            'entity_memory': list(dict.fromkeys([x for x in entity_memory if x]))[:12],
            'world_memory': list(dict.fromkeys([x for x in world_memory if x]))[:8],
            'reference_memory_images': ref_images[:3],
            'query': query,
        }

    def add(self, frame: Any, image: Any):
        item = {
            'frame_id': getattr(frame, 'frame_id', None),
            'story_sentence': getattr(frame, 'story_sentence', ''),
            'event': getattr(frame, 'event', ''),
            'event_grounding': getattr(frame, 'event_grounding', ''),
            'evidence': getattr(frame, 'emotion_evidence', []),
            'emotion': getattr(frame, 'emotion', ''),
            'key_objects': list(getattr(frame, 'key_objects', []) or []) + list(getattr(frame, 'evidence_objects', []) or []),
            'scene_location': getattr(frame, 'scene_location', ''),
            'weather': getattr(frame, 'weather', ''),
            'image_path': getattr(image, 'image_path', ''),
            'scores': getattr(image, 'scores', {}),
        }
        self.items.append(item)
        if len(self.items) > self.max_items:
            self.items = self.items[-self.max_items:]


NarrativeMemoryStore = DCEECausalMemoryStore
