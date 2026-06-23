
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
    Story-faithful causal memory store.

    Inspired by:
    - ViSTA: salient history selection
    - StoryGen / Make-A-Story: auto-regressive story memory
    - DCEE: causal event/evidence memory specialized for narrative control
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

    def _continuity_constraints(self, frame: Any) -> Dict[str, Any]:
        protagonist = self.sink.get('protagonist', 'protagonist')
        must_keep = list(getattr(frame, 'key_objects', []) or [])[:2] + list(getattr(frame, 'evidence_objects', []) or [])[:2]
        return {
            'same_protagonist': protagonist,
            'preserve_age_gender_face_hair_outfit': True,
            'must_keep_objects_if_relevant': must_keep,
            'target_emotion': getattr(frame, 'emotion', ''),
            'scene_location': getattr(frame, 'scene_location', ''),
            'weather': getattr(frame, 'weather', ''),
        }

    def _summarize_selected(self, selected: List[Dict[str, Any]]) -> str:
        if not selected:
            return 'No prior visual history. Introduce the first clear scene of the story.'
        parts = []
        for m in selected[:3]:
            parts.append(
                f"frame {m.get('frame_id')}: event={m.get('event','')}; story={m.get('story_sentence','')}; emotion={m.get('emotion','')}; key_objects={m.get('key_objects',[])}"
            )
        return ' | '.join(parts)

    def select(self, frame: Any, dce_plan: Any, emotion_arc: Any, strategy: str = 'adaptive_causal', top_k: int = 3) -> Dict[str, Any]:
        query = {
            'event': getattr(frame, 'event', ''),
            'story_sentence': getattr(frame, 'story_sentence', ''),
            'emotion': getattr(frame, 'emotion', ''),
            'event_grounding': getattr(frame, 'event_grounding', ''),
            'evidence': getattr(frame, 'emotion_evidence', []),
            'key_objects': getattr(frame, 'key_objects', []),
            'scene_location': getattr(frame, 'scene_location', ''),
            'weather': getattr(frame, 'weather', ''),
            'conflict': getattr(dce_plan, 'conflict', ''),
        }
        scored = []
        total = max(1, len(self.items))
        for idx, item in enumerate(self.items):
            recency = (idx + 1) / total
            score = (
                0.24 * _jaccard(query.get('story_sentence'), item.get('story_sentence'))
                + 0.18 * _jaccard(query.get('event'), item.get('event'))
                + 0.14 * _jaccard(query.get('event_grounding'), item.get('event_grounding'))
                + 0.12 * _jaccard(query.get('evidence'), item.get('evidence'))
                + 0.10 * _jaccard(query.get('key_objects'), item.get('key_objects'))
                + 0.08 * _jaccard(query.get('scene_location'), item.get('scene_location'))
                + 0.06 * _jaccard(query.get('weather'), item.get('weather'))
                + 0.04 * _jaccard(query.get('emotion'), item.get('emotion'))
                + 0.04 * recency
            )
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [dict(score=round(s, 4), **m) for s, m in scored[:top_k]]
        return {
            'causal_sink': self.sink,
            'selected_memories': selected,
            'salient_history': self._summarize_selected(selected),
            'continuity_constraints': self._continuity_constraints(frame),
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
            'key_objects': getattr(frame, 'key_objects', []),
            'scene_location': getattr(frame, 'scene_location', ''),
            'weather': getattr(frame, 'weather', ''),
            'image_path': getattr(image, 'image_path', ''),
            'scores': getattr(image, 'scores', {}),
        }
        self.items.append(item)
        if len(self.items) > self.max_items:
            self.items = self.items[-self.max_items:]

# Backward compatible alias.
NarrativeMemoryStore = DCEECausalMemoryStore
