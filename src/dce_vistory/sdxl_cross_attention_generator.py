
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Dict, List
import torch
from .adapters_pytorch import ButterflyAdapterStack
from .latent_schema import VisualControlPacket
from .schema import CandidateImage
from .story_visual_alignment import extract_visual_contract, build_prompt_variant, build_negative_variant, local_story_keywords, limit_words


def _clean(x):
    return re.sub(r'\s+',' ',str(x or '').replace('\n',' ')).strip()


def _compact_prompt(packet: VisualControlPacket, mode: str = 'event_first'):
    contract = extract_visual_contract(packet)
    return build_prompt_variant(contract, mode)


def _compact_negative(packet):
    return build_negative_variant(packet)


class SDXLButterflyCrossAttentionGenerator:
    def __init__(self, model_id='stabilityai/stable-diffusion-xl-base-1.0', device='cuda', width=1024, height=1024, num_inference_steps=40, guidance_scale=8.0, seed=42, adapter_ckpt=None, enable_cpu_offload=False, character_tokens=8, world_tokens=8, emotion_tokens=8, event_tokens=8, evidence_tokens=8):
        from diffusers import StableDiffusionXLPipeline
        self.device=device; self.width=int(width); self.height=int(height); self.num_inference_steps=int(num_inference_steps); self.guidance_scale=float(guidance_scale); self.seed=int(seed); self.dtype=torch.float16 if device.startswith('cuda') else torch.float32
        self.pipe=StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=self.dtype, use_safetensors=True)
        if enable_cpu_offload and device.startswith('cuda'): self.pipe.enable_model_cpu_offload()
        else: self.pipe.to(device)
        try: self.pipe.vae.enable_slicing()
        except Exception: pass
        self.adapter_stack=ButterflyAdapterStack(character_tokens, world_tokens, emotion_tokens, event_tokens, evidence_tokens).to(device)
        if self.dtype==torch.float16: self.adapter_stack=self.adapter_stack.half()
        if adapter_ckpt: self.adapter_stack.load_state_dict(torch.load(adapter_ckpt,map_location=device), strict=False)
        self.adapter_stack.eval()

    def _encode_prompt(self, prompt, negative_prompt=None, do_cfg=True):
        out=self.pipe.encode_prompt(prompt=prompt, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=do_cfg, negative_prompt=negative_prompt or '')
        if len(out)!=4: raise RuntimeError(f'Unexpected encode_prompt output length: {len(out)}')
        return out

    def _encode_control(self, text):
        pe,_,_,_=self._encode_prompt(limit_words(text, 70), negative_prompt='', do_cfg=False)
        return pe

    def _control_texts_from_packet(self, packet):
        m=packet.control_metadata or {}
        contract = extract_visual_contract(packet)
        char = contract['identity_prompt'] + '; ' + contract['outfit_prompt'] + '; ' + ', '.join(contract['signature_items'])
        world = ', '.join([contract['scene_location'], contract['weather'], contract['atmosphere'], *contract['environment_details']])
        emotion = f"emotion {contract['emotion']} intensity {contract['emotion_intensity']}; facial {contract['facial_rule']}; body {contract['body_rule']}; lighting {contract['lighting_rule']}; color {contract['color_rule']}"
        event = f"story sentence {contract['story_sentence']}; event {contract['event_short']}; grounding {contract['event_grounding']}"
        evidence = 'must show ' + ', '.join(contract['must_show'] or contract['evidence_objects'] or contract['emotion_evidence'])
        return {'character_adapter': limit_words(char, 35), 'world_adapter': limit_words(world, 35), 'emotion_adapter': limit_words(emotion, 35), 'event_adapter': limit_words(event, 35), 'evidence_adapter': limit_words(evidence, 35)}

    @torch.no_grad()
    def generate_from_packet(self, packet: VisualControlPacket, frame_id:int, out_dir:Path, num_candidates:int=1) -> List[CandidateImage]:
        out_dir.mkdir(parents=True, exist_ok=True)
        modes = ['event_first', 'evidence_first', 'emotion_first', 'continuity_first']
        neg = _compact_negative(packet)
        control_texts=self._control_texts_from_packet(packet)
        control_embeds={n:self._encode_control(t) for n,t in control_texts.items()}
        res=[]
        contract = extract_visual_contract(packet)
        kw = local_story_keywords(contract)
        for cid in range(num_candidates):
            mode = modes[cid % len(modes)]
            pos = _compact_prompt(packet, mode)
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds=self._encode_prompt(pos, neg, True)
            prompt_embeds=prompt_embeds.to(self.device,self.dtype); negative_prompt_embeds=negative_prompt_embeds.to(self.device,self.dtype); pooled_prompt_embeds=pooled_prompt_embeds.to(self.device,self.dtype); negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(self.device,self.dtype)
            aug, tokens=self.adapter_stack(prompt_embeds, control_embeds, packet.adapter_weights or {})
            if tokens.shape[1]>0: negative_prompt_embeds=torch.cat([negative_prompt_embeds, torch.zeros_like(tokens)], dim=1)
            sd=self.seed+int(frame_id)*1000+cid
            gen=torch.Generator(device=self.device).manual_seed(sd) if self.device.startswith('cuda') else torch.Generator().manual_seed(sd)
            img=self.pipe(prompt_embeds=aug, negative_prompt_embeds=negative_prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds, width=self.width, height=self.height, num_inference_steps=self.num_inference_steps, guidance_scale=self.guidance_scale, generator=gen).images[0]
            path=out_dir/f'frame_{int(frame_id):03d}_cand_{cid:02d}.png'; img.save(path)
            res.append(CandidateImage(frame_id=int(frame_id), candidate_id=cid, image_path=str(path), prompt=pos, scores={'image_quality':0.0,'identity_consistency':0.0,'emotion_visibility':0.0,'emotion_cause_visibility':0.0,'event_grounding':0.0,'evidence_visibility':0.0,'event_emotion_causal_consistency':0.0,'scene_alignment':0.0,'event_alignment':0.0,'story_alignment':0.0,'colorfulness':0.0,'continuity':0.0,'overall':0.0}, notes={'seed':sd,'prompt_variant_mode':mode,'adapter_weights':packet.adapter_weights,'cross_attention_adapter_tokens':int(tokens.shape[1]),'negative_prompt':neg,'prompt_compacted_for_clip_77_tokens':True,'control_texts':control_texts,'story_keywords':kw,'contract':contract}))
        return res
