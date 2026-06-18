from __future__ import annotations
import json, re
from pathlib import Path
from typing import Dict, List
import torch
from .adapters_pytorch import ButterflyAdapterStack
from .latent_schema import VisualControlPacket
from .schema import CandidateImage

def _clean(x): return re.sub(r'\s+',' ',str(x or '').replace('\n',' ')).strip()
def _limit(x, n=60):
    words=_clean(x).split(); return ' '.join(words[:n])

def _compact_prompt(packet: VisualControlPacket):
    m=packet.control_metadata or {}
    return _limit('full-color cinematic storybook illustration, rich natural colors, expressive face and body, clear DCEE event, visible evidence, visual cause of emotion, detailed world, '+', '.join(_limit(m.get(k,''),18) for k in ['emotion_text','event_text','evidence_text','world_text','character_text']), 70)

def _compact_negative(packet):
    return _limit('monochrome, grayscale, black and white, pencil sketch, line art only, colorless, missing evidence, missing event, emotionless face, weak expression, stiff pose, empty background, low quality, blurry, bad anatomy, text, watermark, '+_clean(getattr(packet,'negative_prompt','')), 70)

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
        pe,_,_,_=self._encode_prompt(_limit(text,70), negative_prompt='', do_cfg=False); return pe
    def _control_texts_from_packet(self, packet):
        m=packet.control_metadata or {}
        return {name:_limit(m.get(key, json.dumps(m.get(key.replace('_text',''),{}), ensure_ascii=False)),70) for name,key in [('character_adapter','character_text'),('world_adapter','world_text'),('emotion_adapter','emotion_text'),('event_adapter','event_text'),('evidence_adapter','evidence_text')]}
    @torch.no_grad()
    def generate_from_packet(self, packet: VisualControlPacket, frame_id:int, out_dir:Path, num_candidates:int=1) -> List[CandidateImage]:
        out_dir.mkdir(parents=True, exist_ok=True); pos=_compact_prompt(packet); neg=_compact_negative(packet)
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds=self._encode_prompt(pos, neg, True)
        control_texts=self._control_texts_from_packet(packet); control_embeds={n:self._encode_control(t) for n,t in control_texts.items()}
        prompt_embeds=prompt_embeds.to(self.device,self.dtype); negative_prompt_embeds=negative_prompt_embeds.to(self.device,self.dtype); pooled_prompt_embeds=pooled_prompt_embeds.to(self.device,self.dtype); negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(self.device,self.dtype)
        aug, tokens=self.adapter_stack(prompt_embeds, control_embeds, packet.adapter_weights or {})
        if tokens.shape[1]>0: negative_prompt_embeds=torch.cat([negative_prompt_embeds, torch.zeros_like(tokens)], dim=1)
        res=[]
        for cid in range(num_candidates):
            sd=self.seed+int(frame_id)*1000+cid; gen=torch.Generator(device=self.device).manual_seed(sd) if self.device.startswith('cuda') else torch.Generator().manual_seed(sd)
            img=self.pipe(prompt_embeds=aug, negative_prompt_embeds=negative_prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds, width=self.width, height=self.height, num_inference_steps=self.num_inference_steps, guidance_scale=self.guidance_scale, generator=gen).images[0]
            path=out_dir/f'frame_{int(frame_id):03d}_cand_{cid:02d}.png'; img.save(path)
            res.append(CandidateImage(frame_id=int(frame_id), candidate_id=cid, image_path=str(path), prompt=pos, scores={'image_quality':0.0,'identity_consistency':0.0,'emotion_visibility':0.0,'emotion_cause_visibility':0.0,'event_grounding':0.0,'evidence_visibility':0.0,'event_emotion_causal_consistency':0.0,'scene_alignment':0.0,'event_alignment':0.0,'colorfulness':0.0,'overall':0.0}, notes={'seed':sd,'adapter_weights':packet.adapter_weights,'cross_attention_adapter_tokens':int(tokens.shape[1]),'negative_prompt':neg,'prompt_compacted_for_clip_77_tokens':True,'control_texts':control_texts}))
        return res
