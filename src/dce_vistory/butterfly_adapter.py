from __future__ import annotations
from dataclasses import asdict
from typing import Any, Dict
from .emotion_world_rules import get_world_rule
from .latent_schema import CharacterLatent, WorldLatent, EmotionLatent, VisualControlPacket


def _normalize_symbolic_objects(value):
    """
    Normalize seed.visual_symbols / symbolic_objects for WorldLatent.
    WorldLatent.to_prompt() expects a dict and calls .items().
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items() if str(k).strip()}
    if isinstance(value, str):
        text = value.strip()
        return {text: "visual symbol"} if text else {}
    if isinstance(value, (list, tuple)):
        out = {}
        for item in value:
            if item is None:
                continue
            if isinstance(item, dict):
                key = item.get("object") or item.get("name") or item.get("symbol") or item.get("item") or item.get("key")
                val = item.get("meaning") or item.get("symbolic_meaning") or item.get("description") or item.get("value") or "visual symbol"
                if key:
                    out[str(key)] = str(val)
            else:
                text = str(item).strip()
                if text:
                    out[text] = "visual symbol"
        return out
    return {str(value): "visual symbol"}


class ButterflyController:
    """DCEE-CausalVerse visual controller with Character/World/Emotion/Event/Evidence branches."""
    def __init__(self, quality_suffix: str, negative_prompt: str, num_hypotheses: int=3):
        self.quality_suffix=quality_suffix; self.negative_prompt=negative_prompt; self.num_hypotheses=num_hypotheses

    def build_character_latent(self, seed: Any) -> CharacterLatent:
        profile=None
        for p in getattr(seed,'character_profiles',[]) or []:
            if getattr(p,'role','')=='protagonist' or getattr(p,'name','').lower()==getattr(seed,'protagonist','').lower(): profile=p; break
        if profile is None and getattr(seed,'character_profiles',None): profile=seed.character_profiles[0]
        if profile is None:
            return CharacterLatent(name=getattr(seed,'protagonist','protagonist'), role='protagonist', identity_prompt=f"same protagonist identity: {getattr(seed,'protagonist','protagonist')}")
        if hasattr(profile,'to_prompt'):
            try: ident=profile.to_prompt()
            except Exception: ident=str(getattr(profile,'identity_anchor_prompt',getattr(profile,'name','protagonist')))
        else:
            ident='; '.join(str(getattr(profile,k,'')) for k in ['name','role','face','hair','body','outfit','identity_anchor_prompt'] if getattr(profile,k,''))
        return CharacterLatent(name=getattr(profile,'name',getattr(seed,'protagonist','protagonist')), role=getattr(profile,'role','protagonist'), identity_prompt=ident, outfit_prompt=getattr(profile,'outfit',''), signature_items=getattr(profile,'signature_items',[]) or [], reference_image=None, negative_prompt=getattr(profile,'negative_identity_prompt','different identity, changed face, changed outfit, inconsistent character, child version, older version, gender changed'))

    def create_packet(self, frame: Any, seed: Any, dce_plan: Any, memory: Dict[str,Any], style: str, previous_frame: Any=None, anchors: Dict[str,Any] | None=None) -> VisualControlPacket:
        character=self.build_character_latent(seed); world_rule=get_world_rule(getattr(frame,'emotion',''))
        world=WorldLatent(scene_location=getattr(frame,'scene_location','') or getattr(seed,'setting',''), time_of_day=getattr(frame,'time_of_day','') or 'cinematic story time', weather=getattr(frame,'weather','') or world_rule.get('weather','cinematic weather'), atmosphere=getattr(frame,'atmosphere','') or world_rule.get('environment','emotionally meaningful atmosphere'), environment_details=list(getattr(frame,'environment_details',[]) or [])+[world_rule.get('environment','story-relevant environment'), f"lighting: {getattr(frame,'lighting_style','') or world_rule.get('lighting','')}", f"color palette: {getattr(frame,'color_palette','') or world_rule.get('color','')}", f"composition: {getattr(frame,'composition_rule','') or world_rule.get('composition','')}"], scene_transition=getattr(frame,'scene_transition',''), symbolic_objects=_normalize_symbolic_objects(getattr(seed,'visual_symbols',{}) if hasattr(seed,'visual_symbols') else {}))
        emotion=EmotionLatent(emotion=getattr(frame,'emotion',''), intensity=int(getattr(frame,'emotion_intensity',3)), delta_from_previous=getattr(frame,'emotion_delta',''), facial_rule=getattr(frame,'facial_cue',''), body_rule=getattr(frame,'body_cue',''), lighting_rule=getattr(frame,'lighting_style','') or world_rule.get('lighting',''), color_rule=getattr(frame,'color_palette','') or world_rule.get('color',''), composition_rule=getattr(frame,'composition_rule','') or world_rule.get('composition',''))
        intensity=max(1,min(5,emotion.intensity)); emotion_w=0.18+0.07*intensity
        adapter_weights={'character_adapter':0.26,'world_adapter':0.18,'emotion_adapter':emotion_w,'event_adapter':0.18,'evidence_adapter':0.20}
        identity_lock = getattr(frame, 'identity_lock', {}) or {}
        identity_lock_text = '; '.join(f"{k}: {v}" for k, v in identity_lock.items() if v)
        char_text=(character.identity_prompt + '; STRICT IDENTITY LOCK: same age, gender, face shape, hairstyle, body proportions, outfit; only expression, pose, event, emotion, and background may change. ' + identity_lock_text + '; ') + (f"; outfit: {character.outfit_prompt}" if character.outfit_prompt else '') + (('; signature items: '+', '.join(character.signature_items)) if character.signature_items else '')
        story_text=f"story sentence: {getattr(frame,'story_sentence','')}; alignment reason: {getattr(frame,'story_alignment_reason','')}"
        event_text=f"DCEE visible event: {getattr(frame,'event','')}; causal role: {getattr(frame,'event_causal_role','')}; event grounding: {getattr(frame,'event_grounding','')}; narrative function: {getattr(frame,'narrative_function','')}"
        evidence_text=f"visual evidence objects: {getattr(frame,'evidence_objects',[])}; emotion evidence: {getattr(frame,'emotion_evidence',[])}; must show: {getattr(frame,'must_show',[])}; visual cause of emotion must be visible"
        world_text=world.to_prompt(); emotion_text=emotion.to_prompt(); anchor_text=str(anchors or {})
        positive=f"""
[STYLE] {style}
[CHARACTER] {char_text}. Preserve identity but allow frame-specific facial expression and pose.
[STORY SENTENCE] {story_text}
[DCEE EVENT] {event_text}
[EVIDENCE] {evidence_text}
[WORLD] {world_text}
[EMOTION CAUSED BY EVENT] {emotion_text}
[CAUSAL MEMORY] {memory}
[ANCHORS] {anchor_text}
[QUALITY] {self.quality_suffix}
Create a coherent full-color cinematic storybook illustration. The event, evidence, and emotion cause must be visible.
""".strip()
        return VisualControlPacket(frame_id=int(getattr(frame,'frame_id',0)), positive_prompt=positive, negative_prompt=self.negative_prompt+'; '+character.negative_prompt+'; missing event, missing evidence, weak emotion, portrait only', adapter_weights=adapter_weights, control_metadata={'character_text':char_text,'world_text':world_text,'emotion_text':emotion_text,'story_text':story_text,'event_text':event_text,'evidence_text':evidence_text,'anchor_text':anchor_text,'dcee_event_text':event_text,'event_causal_role':getattr(frame,'event_causal_role',''),'event_grounding':getattr(frame,'event_grounding',''),'evidence_objects':getattr(frame,'evidence_objects',[]),'emotion_evidence':getattr(frame,'emotion_evidence',[]),'must_show':getattr(frame,'must_show',[]),'world':asdict(world),'emotion':asdict(emotion),'character':asdict(character)}, reference_images={})
