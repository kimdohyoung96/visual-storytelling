from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from typing import Any, Dict, List
import json

from .llm import BaseLLM
from .schema import StorySeed, DCEPlan, EmotionArc, StoryboardFrame, ImageUnderstanding, CharacterProfile
from .prompts import (
    SYSTEM_NARRATIVE, QUALITY_SUFFIX, NEGATIVE_PROMPT,
    story_seed_prompt, story_abstract_prompt,
    dcee_branch_plan_prompt, dcee_candidate_selection_prompt,
    emotion_arc_prompt, storyboard_prompt, canonicalize_storyboard_prompt,
    get_emotion_rule, emotion_rule_text, emotion_delta_text, choose_shot_type, choose_camera_distance,
)
from .utils import extract_json


def _field_names(cls) -> set[str]:
    return {f.name for f in fields(cls)} if is_dataclass(cls) else set()


def _safe_make(cls, kwargs: Dict[str, Any]):
    names = _field_names(cls)
    if names:
        init_kwargs = {k: v for k, v in kwargs.items() if k in names}
        obj = cls(**init_kwargs)
        for k, v in kwargs.items():
            if k not in names:
                try: setattr(obj, k, v)
                except Exception: pass
        return obj
    return cls(**kwargs)


def _to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None: return {}
    if is_dataclass(obj):
        d = asdict(obj)
        d.update({k:v for k,v in getattr(obj,'__dict__',{}).items() if k not in d})
        return d
    if isinstance(obj, dict): return obj
    return getattr(obj, '__dict__', {}) or {}


def _string_list(value: Any) -> List[str]:
    if value is None: return []
    if isinstance(value, str): value=[value]
    elif isinstance(value, dict): value=[value]
    elif not isinstance(value, list): value=[value]
    out=[]
    for item in value:
        if item is None: continue
        if isinstance(item, str): text=item
        elif isinstance(item, dict): text=item.get('name') or item.get('object') or item.get('item') or item.get('description') or item.get('title') or item.get('event') or item.get('visual_grounding') or str(item)
        else: text=str(item)
        text=str(text).strip()
        if text: out.append(text)
    return list(dict.fromkeys(out))


def _ending_synonym(target: str) -> str:
    t=(target or '').lower().strip()
    return {'happy':'joy','happiness':'joy','sad':'sadness','sad ending':'sadness','angry':'anger','fearful':'fear','relieved':'relief'}.get(t, t or 'resolution')


class DCEPlanner:
    """DCEE-CausalVerse planner. Keeps the old class name for compatibility."""
    def __init__(self, llm: BaseLLM, temperature: float = 0.4, max_tokens: int = 1800):
        self.llm = llm
        self.temperature = temperature
        self.max_tokens = min(int(max_tokens), 1800)

    def _llm_json(self, prompt: str, max_tokens: int | None = None, temperature: float | None = None):
        txt = self.llm.generate(SYSTEM_NARRATIVE, prompt, temperature=self.temperature if temperature is None else temperature, max_tokens=min(max_tokens or self.max_tokens, self.max_tokens))
        return extract_json(txt)

    def build_seed(self, sample: Dict[str, Any], image_summary: ImageUnderstanding | None) -> StorySeed:
        data = self._llm_json(story_seed_prompt(sample, _to_dict(image_summary) if image_summary else None), max_tokens=1000)
        data['objects'] = _string_list(data.get('objects', []))
        data['characters'] = _string_list(data.get('characters', []))
        profiles = self._build_character_profiles(data, sample, image_summary)
        world_context = data.get('world_context', {}) if isinstance(data.get('world_context', {}), dict) else {'raw_world_context': str(data.get('world_context'))}
        if image_summary:
            world_context.setdefault('time_of_day', getattr(image_summary, 'time_of_day', ''))
            world_context.setdefault('weather_prior', getattr(image_summary, 'weather', ''))
            world_context.setdefault('environment_prior', getattr(image_summary, 'environment_details', []))
        seed = _safe_make(StorySeed, {
            'image_summary': image_summary,
            'text_prompt': sample.get('text_prompt',''), 'protagonist': sample.get('protagonist',''),
            'target_ending_emotion': sample.get('target_ending_emotion',''), 'genre': sample.get('genre',''), 'style': sample.get('style',''),
            'setting': data.get('setting', getattr(image_summary,'setting','') if image_summary else ''),
            'objects': data.get('objects', []), 'characters': data.get('characters', []), 'mood': data.get('mood',''),
            'visual_symbols': data.get('visual_symbols', {}), 'world_context': world_context, 'character_profiles': profiles, 'raw_input': sample,
        })
        seed.world_context = world_context; seed.character_profiles = profiles; seed.raw_input = sample
        return seed

    def _build_character_profiles(self, data, sample, image_summary):
        profiles=[]
        raw=data.get('character_profiles', [])
        if isinstance(raw, dict): raw=[raw]
        if not isinstance(raw, list): raw=[]
        for row in raw:
            if not isinstance(row, dict): continue
            profiles.append(_safe_make(CharacterProfile, {
                'name': row.get('name',''), 'role': row.get('role',''), 'age_group': row.get('age_group','adult'), 'gender': row.get('gender','unspecified'),
                'face': row.get('face',''), 'hair': row.get('hair',''), 'body': row.get('body',''), 'outfit': row.get('outfit',''),
                'signature_items': _string_list(row.get('signature_items', [])), 'color_palette': row.get('color_palette',''), 'identity_anchor_prompt': row.get('identity_anchor_prompt',''),
            }))
        protagonist=sample.get('protagonist','protagonist')
        if not any(getattr(p,'role','')=='protagonist' or getattr(p,'name','').lower()==str(protagonist).lower() for p in profiles):
            hint=getattr(image_summary,'caption','') if image_summary else sample.get('text_prompt','')
            signature_items=_string_list(_string_list(sample.get('signature_items', [])) + _string_list(data.get('objects', []))[:3])
            profiles.insert(0, _safe_make(CharacterProfile, {
                'name': protagonist, 'role':'protagonist', 'age_group': sample.get('age_group','adult'), 'gender': sample.get('gender','unspecified'),
                'face': f'recognizable consistent face or character features based on: {hint}', 'hair':'same hairstyle or head shape in every frame',
                'body':'same body shape and proportions in every frame', 'outfit': sample.get('outfit','same main outfit, same colors, same accessories in every frame'),
                'signature_items': signature_items, 'color_palette':'stable protagonist color palette across all frames',
                'identity_anchor_prompt': f'{protagonist} must look like the same character in every frame; same face, outfit, body shape, and signature items.'
            }))
        return profiles

    def generate_abstract(self, seed: StorySeed) -> str:
        return self.llm.generate(SYSTEM_NARRATIVE, story_abstract_prompt(_to_dict(seed)), temperature=self.temperature, max_tokens=800).strip()

    def _fallback_plan(self, seed, abstract):
        target=getattr(seed,'target_ending_emotion','sadness') or 'sadness'
        protagonist=getattr(seed,'protagonist','protagonist') or 'protagonist'
        return {
            'candidate_id':'fallback-1', 'desire': f'{protagonist} wants to resolve the central problem.',
            'conflict':'The desire is blocked by an external obstacle and an internal moral/emotional pressure.',
            'conflict_escalation':'The obstacle forces a visible choice that changes the protagonist emotion.',
            'event_chain':[
                {'event_id':'e1','event':'The protagonist discovers the problem','causal_role':'introduces desire','visual_grounding':'the protagonist faces the object or place that starts the story','emotion_effect':'hope or concern','key_objects':[], 'evidence_objects':[]},
                {'event_id':'e2','event':'The conflict becomes visible','causal_role':'escalates conflict','visual_grounding':'an obstacle, rival, loss, or failed attempt is visible','emotion_effect':'anxiety or doubt','key_objects':[], 'evidence_objects':[]},
                {'event_id':'e3','event':'A decisive event changes the outcome','causal_role':'turning point','visual_grounding':'the protagonist reacts to a concrete event with visible evidence','emotion_effect':target,'key_objects':[], 'evidence_objects':[]},
            ],
            'turning_point':'A decisive event makes the ending emotion inevitable.', 'ending_emotion':target, 'target_ending_emotion':target, 'ending_state':f'The protagonist ends in {target}.', 'rationale':'fallback plan'
        }

    def generate_dce_plan(self, seed: StorySeed, abstract: str) -> DCEPlan:
        # New idea: sample several Desire->Conflict routes before committing to events.
        n = int(getattr(seed, 'raw_input', {}).get('num_dcee_candidates', 4) if isinstance(getattr(seed, 'raw_input', {}), dict) else 4)
        n = max(1, min(6, n))
        try:
            data = self._llm_json(dcee_branch_plan_prompt(_to_dict(seed), abstract, num_candidates=n), max_tokens=1600, temperature=max(0.55, self.temperature))
            candidates = data.get('candidates', data if isinstance(data, list) else [])
            if isinstance(candidates, dict): candidates=[candidates]
            if not candidates: candidates=[self._fallback_plan(seed, abstract)]
        except Exception:
            candidates=[self._fallback_plan(seed, abstract)]
        candidates = [self._normalize_candidate(c, i, seed) for i,c in enumerate(candidates)]
        selected = self._select_best_candidate(seed, abstract, candidates)
        event_chain = selected.get('event_chain', selected.get('event_spine', []))
        dce_plan = _safe_make(DCEPlan, {
            'protagonist': selected.get('protagonist', getattr(seed,'protagonist','')), 'desire': selected.get('desire',''), 'fear': selected.get('fear',''),
            'misbelief': selected.get('misbelief',''), 'obstacle': selected.get('obstacle',''), 'conflict': selected.get('conflict',''),
            'event_spine': event_chain, 'turning_point': selected.get('turning_point',''),
            'target_ending_emotion': selected.get('target_ending_emotion', selected.get('ending_emotion', getattr(seed,'target_ending_emotion',''))),
            'ending_state': selected.get('ending_state',''), 'moral_or_theme': selected.get('moral_or_theme',''),
            'event_chain': event_chain, 'dcee_candidates': candidates, 'selected_candidate': selected,
            'planning_structure':'DCEE-Tree: multi-sampled Desire-Conflict routes -> selected Event Chain -> Ending Emotion'
        })
        dce_plan.event_chain = event_chain; dce_plan.dcee_candidates = candidates; dce_plan.selected_candidate=selected
        dce_plan.planning_structure='DCEE-Tree: multi-sampled Desire-Conflict routes -> selected Event Chain -> Ending Emotion'
        return dce_plan

    def _normalize_candidate(self, c, idx, seed):
        if not isinstance(c, dict): c={'candidate_id':f'c{idx+1}', 'event_chain':[str(c)]}
        c.setdefault('candidate_id', f'c{idx+1}')
        c.setdefault('target_ending_emotion', c.get('ending_emotion', getattr(seed,'target_ending_emotion','')))
        chain=c.get('event_chain', c.get('event_spine', []))
        if isinstance(chain, dict): chain=chain.get('events',[chain])
        if not isinstance(chain, list): chain=[str(chain)]
        norm=[]
        for j,e in enumerate(chain):
            if not isinstance(e, dict): e={'event': str(e)}
            e.setdefault('event_id', f'e{j+1}'); e.setdefault('causal_role', e.get('role','causes or intensifies emotion'))
            e.setdefault('visual_grounding', e.get('visual_evidence', e.get('description', e.get('event',''))))
            e['key_objects']=_string_list(e.get('key_objects', [])); e['evidence_objects']=_string_list(e.get('evidence_objects', e.get('visual_evidence_objects', [])))
            norm.append(e)
        c['event_chain']=norm; c['event_spine']=norm
        return c

    def _select_best_candidate(self, seed, abstract, candidates):
        # First try LLM judge; fallback to heuristic.
        try:
            judge=self._llm_json(dcee_candidate_selection_prompt(_to_dict(seed), abstract, candidates), max_tokens=900, temperature=0.0)
            sid=str(judge.get('selected_candidate_id',''))
            for c in candidates:
                if str(c.get('candidate_id')) == sid:
                    c['selection_scores']=judge.get('scores', [])
                    c['selection_reason']=judge.get('reason','')
                    return c
        except Exception:
            pass
        def score(c):
            chain=c.get('event_chain', [])
            visual=sum(1 for e in chain if e.get('visual_grounding'))
            evidence=sum(len(_string_list(e.get('evidence_objects',[]))) for e in chain)
            return len(chain)*2 + visual + evidence + (2 if c.get('turning_point') else 0) + (2 if c.get('conflict') else 0)
        best=max(candidates, key=score)
        best['selection_reason']='heuristic selected for visual event coverage and evidence availability'
        return best

    def generate_emotion_arc(self, seed: StorySeed, abstract: str, dce_plan: DCEPlan, num_frames: int) -> EmotionArc:
        try:
            data=self._llm_json(emotion_arc_prompt(_to_dict(seed), abstract, _to_dict(dce_plan), num_frames), max_tokens=1000)
        except Exception: data={}
        states=data.get('states', []); intensities=data.get('intensities', [])
        if len(states)!=num_frames: states=(states+[_ending_synonym(getattr(dce_plan,'target_ending_emotion',''))]*num_frames)[:num_frames]
        if len(intensities)!=num_frames: intensities=(intensities+[3]*num_frames)[:num_frames]
        if states: states[-1]=_ending_synonym(getattr(dce_plan,'target_ending_emotion',''))
        return _safe_make(EmotionArc, {'states':states, 'intensities':[max(1,min(5,int(x))) for x in intensities], 'rationale':data.get('rationale',''), 'valence_curve':data.get('valence_curve',[]), 'arousal_curve':data.get('arousal_curve',[]), 'suspense_curve':data.get('suspense_curve',[])})

    def generate_storyboard(self, seed: StorySeed, abstract: str, dce_plan: DCEPlan, emotion_arc: EmotionArc) -> List[StoryboardFrame]:
        states=getattr(emotion_arc,'states',[]); intensities=getattr(emotion_arc,'intensities',[]); num_frames=len(states)
        try:
            rows=self._llm_json(storyboard_prompt(_to_dict(seed), abstract, _to_dict(dce_plan), _to_dict(emotion_arc), num_frames), max_tokens=1600)
            if isinstance(rows, dict): rows=rows.get('storyboard', rows.get('frames', rows))
            if not isinstance(rows, list): raise ValueError('storyboard not list')
        except Exception:
            rows=[]
        if not rows:
            chain=getattr(dce_plan,'event_chain',getattr(dce_plan,'event_spine',[])) or []
            rows=[{'frame_id':i+1,'event':(chain[min(i,len(chain)-1)].get('event','event') if chain else 'event'), 'narrative_function':'event progression'} for i in range(num_frames)]
        try:
            crows=self._llm_json(canonicalize_storyboard_prompt(_to_dict(seed), _to_dict(dce_plan), rows), max_tokens=1400, temperature=0.0)
            if isinstance(crows, dict): crows=crows.get('storyboard', crows.get('frames', crows))
            if isinstance(crows, list): rows=crows
        except Exception: pass
        return self._postprocess_storyboard(rows, seed, dce_plan, emotion_arc)

    def _postprocess_storyboard(self, rows, seed, dce_plan, emotion_arc):
        protagonist_profile=self._get_protagonist_profile(seed)
        protagonist_identity=self._profile_to_prompt(protagonist_profile) if protagonist_profile else getattr(seed,'protagonist','protagonist')
        character_reference_prompt=f'Use the same protagonist identity in every frame: {protagonist_identity}. Keep identity stable while allowing emotion-specific expressions and poses.'
        world_context=getattr(seed,'world_context',{}) or {}; img=getattr(seed,'image_summary',None)
        base_time=world_context.get('time_of_day', getattr(img,'time_of_day','') if img else '')
        base_weather=world_context.get('weather_prior', getattr(img,'weather','') if img else '')
        base_env=world_context.get('environment_prior', getattr(img,'environment_details',[]) if img else [])
        chain=getattr(dce_plan,'event_chain',getattr(dce_plan,'event_spine',[])) or []
        frames=[]; prev_emotion=None; prev_world=None
        states=getattr(emotion_arc,'states',[]); intensities=getattr(emotion_arc,'intensities',[])
        for idx,row in enumerate(rows):
            if not isinstance(row, dict): row={'event':str(row)}
            emotion=row.get('emotion', states[min(idx,len(states)-1)] if states else 'neutral')
            intensity=max(1,min(5,int(row.get('emotion_intensity', intensities[min(idx,len(intensities)-1)] if intensities else 3))))
            rule=get_emotion_rule(emotion)
            linked=chain[min(idx,len(chain)-1)] if chain else {}
            if not isinstance(linked, dict): linked={'event':str(linked),'visual_grounding':str(linked),'causal_role':'causes emotion'}
            ev=row.get('event') or linked.get('event') or linked.get('description','')
            evrole=row.get('event_causal_role') or linked.get('causal_role','causes or intensifies the frame emotion')
            evground=row.get('event_grounding') or linked.get('visual_grounding', ev)
            evidence_objects=_string_list(row.get('evidence_objects', linked.get('evidence_objects', [])))
            key_objects=_string_list(row.get('key_objects', linked.get('key_objects', [])))
            env=_string_list(row.get('environment_details', base_env)) + [f"lighting style: {rule['lighting']}", f"composition supports {emotion}: {rule['composition']}", 'full-color environment, not grayscale']
            loc=row.get('scene_location', getattr(seed,'setting',''))
            weather=row.get('weather', base_weather) or rule['weather']
            transition=row.get('scene_transition', 'Establish the initial world state.' if not prev_world else f"The scene evolves from {prev_world['scene_location']} in {prev_world['weather']} weather to {loc} in {weather} weather.")
            nf=row.get('narrative_function','event progression')
            shot=choose_shot_type(idx, len(rows), nf); cam=choose_camera_distance(shot)
            evidence=_string_list([ev, evground]+key_objects[:3]+evidence_objects[:3]+([f"evidence of desire: {row.get('desire_link')}"] if row.get('desire_link') else []))
            must_show=_string_list(key_objects[:3]+evidence_objects[:3]+[ev, evground, f'facial evidence of {emotion}', f'body evidence of {emotion}', 'the current DCEE event', 'the visual cause of the emotion', 'full-color emotional lighting'])
            neg=NEGATIVE_PROMPT + ('; '+getattr(protagonist_profile,'negative_identity_prompt','') if protagonist_profile else '')
            frame=_safe_make(StoryboardFrame, {
                'frame_id':int(row.get('frame_id',idx+1)), 'caption':row.get('caption',''), 'narrative_function':nf, 'event':ev,
                'protagonist_state':row.get('protagonist_state',''), 'desire_link':row.get('desire_link',''), 'conflict_level':int(row.get('conflict_level',1)),
                'emotion':emotion, 'emotion_intensity':intensity, 'visual_focus':row.get('visual_focus',''), 'key_objects':key_objects, 'evidence_objects':evidence_objects,
                'facial_cue':row.get('facial_cue') or rule['face'], 'body_cue':row.get('body_cue') or rule['body'], 'event_cue':row.get('event_cue',ev),
                'scene_cue':row.get('scene_cue','The background and environment must visually support the DCEE event and emotion.'), 'cinematic_cue':row.get('cinematic_cue', f'{shot}, cinematic storytelling composition'),
                'scene_location':loc, 'time_of_day':row.get('time_of_day',base_time), 'weather':weather, 'atmosphere':row.get('atmosphere','') or f'{emotion} atmosphere', 'environment_details':env,
                'supporting_cast':_string_list(row.get('supporting_cast',[])), 'scene_transition':transition, 'character_identity':protagonist_identity, 'character_reference_prompt':character_reference_prompt,
                'emotion_delta':emotion_delta_text(prev_emotion,emotion,intensity), 'emotion_visual_rule':emotion_rule_text(emotion),
                'composition_rule':f"{shot}, {cam} distance, {rule['composition']}. Show the DCEE event and visual evidence.", 'quality_rule':QUALITY_SUFFIX, 'negative_prompt':neg,
                'dcee_stage':'Event','event_causal_role':evrole,'event_grounding':evground,'event_emotion_causal_consistency':f"The event '{ev}' should naturally explain or intensify '{emotion}'.",
                'shot_type':shot,'camera_distance':cam,'color_palette':rule['palette'],'lighting_style':rule['lighting'],'must_show':must_show,'emotion_evidence':evidence,
            })
            frames.append(frame); prev_emotion=emotion; prev_world={'scene_location':loc,'weather':weather}
        return frames

    @staticmethod
    def _profile_to_prompt(profile: CharacterProfile) -> str:
        if hasattr(profile, 'to_prompt'):
            try: return profile.to_prompt()
            except Exception: pass
        return '; '.join(str(getattr(profile,k,'')) for k in ['name','role','face','hair','body','outfit','signature_items','color_palette','identity_anchor_prompt'] if getattr(profile,k,''))

    @staticmethod
    def _get_protagonist_profile(seed: StorySeed):
        for p in getattr(seed,'character_profiles',[]) or []:
            if getattr(p,'role','')=='protagonist' or getattr(p,'name','').lower()==getattr(seed,'protagonist','').lower(): return p
        cps=getattr(seed,'character_profiles',[]) or []
        return cps[0] if cps else None
