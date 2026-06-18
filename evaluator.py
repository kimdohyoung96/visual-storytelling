from __future__ import annotations
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List


def _txt(x):
    if x is None: return ''
    if isinstance(x, (list, tuple)): return ' '.join(_txt(v) for v in x)
    if isinstance(x, dict): return ' '.join(_txt(v) for v in x.values())
    if is_dataclass(x): return _txt(asdict(x))
    return str(x)

def _tokens(x):
    return set(t.lower().strip('.,;:!?()[]{}') for t in _txt(x).split() if len(t)>2)

def _jaccard(a,b):
    A=_tokens(a); B=_tokens(b)
    if not A or not B: return 0.0
    return len(A & B)/len(A | B)

class DCEECausalMemoryStore:
    """Causal sink memory inspired by salient history selection and causal attention, but specialized for DCEE."""
    def __init__(self, max_items:int=12):
        self.max_items=max_items
        self.items: List[Dict[str, Any]]=[]
        self.sink: Dict[str, Any]={}

    def initialize(self, seed: Any, dce_plan: Any):
        self.sink={
            'identity': getattr(seed,'character_profiles', []),
            'desire': getattr(dce_plan,'desire',''),
            'conflict': getattr(dce_plan,'conflict',''),
            'target_ending_emotion': getattr(dce_plan,'target_ending_emotion',''),
            'event_chain': getattr(dce_plan,'event_chain',getattr(dce_plan,'event_spine',[])),
        }

    def select(self, frame: Any, dce_plan: Any, emotion_arc: Any, strategy: str='adaptive_causal', top_k:int=4) -> Dict[str, Any]:
        query={
            'event':getattr(frame,'event',''), 'emotion':getattr(frame,'emotion',''), 'event_grounding':getattr(frame,'event_grounding',''),
            'evidence':getattr(frame,'emotion_evidence',[]), 'key_objects':getattr(frame,'key_objects',[]), 'conflict':getattr(dce_plan,'conflict','')
        }
        scored=[]
        for item in self.items:
            s=0.24*_jaccard(query.get('event'), item.get('event')) + 0.20*_jaccard(query.get('evidence'), item.get('evidence')) + 0.18*_jaccard(query.get('conflict'), item.get('conflict')) + 0.15*_jaccard(query.get('emotion'), item.get('emotion')) + 0.13*_jaccard(query.get('key_objects'), item.get('key_objects')) + 0.10*_jaccard(query.get('event_grounding'), item.get('event_grounding'))
            scored.append((s,item))
        scored.sort(key=lambda x:x[0], reverse=True)
        return {'causal_sink':self.sink, 'selected_memories':[dict(score=round(s,4), **m) for s,m in scored[:top_k]], 'query':query}

    def add(self, frame: Any, image: Any):
        item={'frame_id':getattr(frame,'frame_id',None), 'event':getattr(frame,'event',''), 'event_grounding':getattr(frame,'event_grounding',''), 'evidence':getattr(frame,'emotion_evidence',[]), 'emotion':getattr(frame,'emotion',''), 'key_objects':getattr(frame,'key_objects',[]), 'conflict':getattr(frame,'desire_link',''), 'image_path':getattr(image,'image_path',''), 'scores':getattr(image,'scores',{})}
        self.items.append(item)
        if len(self.items)>self.max_items: self.items=self.items[-self.max_items:]

# Backward compatible alias.
NarrativeMemoryStore = DCEECausalMemoryStore
