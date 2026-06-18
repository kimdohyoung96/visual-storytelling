from __future__ import annotations
from typing import Any, Dict, List
from dataclasses import asdict, is_dataclass

class DCEEAnchorBank:
    """Lightweight text anchor bank for characters, evidence objects, and world anchors."""
    def __init__(self):
        self.character_anchors: Dict[str, Dict[str, Any]]={}
        self.object_anchors: Dict[str, Dict[str, Any]]={}
        self.world_anchors: Dict[str, Dict[str, Any]]={}

    @staticmethod
    def _profile_text(p):
        if hasattr(p,'to_prompt'):
            try: return p.to_prompt()
            except Exception: pass
        if is_dataclass(p): d=asdict(p)
        else: d=getattr(p,'__dict__',{}) if not isinstance(p,dict) else p
        return '; '.join(str(v) for v in d.values() if v)

    def build_from_seed_and_plan(self, seed: Any, dce_plan: Any):
        for p in getattr(seed,'character_profiles',[]) or []:
            name=str(getattr(p,'name','') or getattr(p,'role','character'))
            self.character_anchors[name]={'name':name,'text':self._profile_text(p),'role':getattr(p,'role','')}
        for obj in getattr(seed,'objects',[]) or []:
            name=str(obj if not isinstance(obj,dict) else obj.get('name') or obj.get('object') or obj)
            self.object_anchors[name]={'name':name,'text':name,'source':'seed_object'}
        for ev in getattr(dce_plan,'event_chain',getattr(dce_plan,'event_spine',[])) or []:
            if not isinstance(ev,dict): continue
            for obj in (ev.get('evidence_objects') or ev.get('key_objects') or []):
                name=str(obj if not isinstance(obj,dict) else obj.get('name') or obj.get('object') or obj)
                self.object_anchors[name]={'name':name,'text':name,'source':'event_evidence','causal_role':ev.get('causal_role','')}
        wc=getattr(seed,'world_context',{}) or {}
        if wc:
            self.world_anchors['global_world']={'name':'global_world','text':str(wc)}
        return self

    def select_for_frame(self, frame: Any) -> Dict[str, Any]:
        keys=[str(x) for x in (getattr(frame,'key_objects',[]) or [])+(getattr(frame,'evidence_objects',[]) or [])+(getattr(frame,'emotion_evidence',[]) or [])]
        selected_objects={k:v for k,v in self.object_anchors.items() if any(k.lower() in q.lower() or q.lower() in k.lower() for q in keys)}
        return {'characters':self.character_anchors, 'objects':selected_objects, 'world':self.world_anchors}
