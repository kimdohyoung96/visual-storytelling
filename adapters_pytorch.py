from __future__ import annotations
from typing import Dict, Tuple
import torch
from torch import nn

class CrossAttentionTokenAdapter(nn.Module):
    def __init__(self, name: str, num_tokens: int, init_logit_gate: float = 2.0):
        super().__init__(); self.name=name; self.num_tokens=int(num_tokens); self.logit_gate=nn.Parameter(torch.tensor(float(init_logit_gate)))
    def forward(self, condition_embeds: torch.Tensor, adapter_weight: float=1.0) -> torch.Tensor:
        if condition_embeds.ndim != 3: raise ValueError(f'{self.name} expected [B,L,D], got {tuple(condition_embeds.shape)}')
        _, L, _ = condition_embeds.shape
        if L >= self.num_tokens: tokens=condition_embeds[:, :self.num_tokens, :]
        else:
            rep=(self.num_tokens+L-1)//max(L,1); tokens=condition_embeds.repeat(1,rep,1)[:, :self.num_tokens, :]
        return tokens * torch.sigmoid(self.logit_gate).to(tokens.dtype) * float(adapter_weight)

class ButterflyAdapterStack(nn.Module):
    """Five factorized adapters: character, world, emotion, event, evidence."""
    def __init__(self, character_tokens:int=8, world_tokens:int=8, emotion_tokens:int=8, event_tokens:int=8, evidence_tokens:int=8):
        super().__init__()
        self.character_adapter=CrossAttentionTokenAdapter('character_adapter', character_tokens, 2.0)
        self.world_adapter=CrossAttentionTokenAdapter('world_adapter', world_tokens, 1.8)
        self.emotion_adapter=CrossAttentionTokenAdapter('emotion_adapter', emotion_tokens, 2.2)
        self.event_adapter=CrossAttentionTokenAdapter('event_adapter', event_tokens, 1.8)
        self.evidence_adapter=CrossAttentionTokenAdapter('evidence_adapter', evidence_tokens, 2.3)
    def forward(self, prompt_embeds: torch.Tensor, control_embeds: Dict[str,torch.Tensor], adapter_weights: Dict[str,float]) -> Tuple[torch.Tensor, torch.Tensor]:
        extra=[]
        adapters={'character_adapter':self.character_adapter,'world_adapter':self.world_adapter,'emotion_adapter':self.emotion_adapter,'event_adapter':self.event_adapter,'evidence_adapter':self.evidence_adapter}
        for name, adapter in adapters.items():
            cond=control_embeds.get(name)
            if cond is None: continue
            cond=cond.to(prompt_embeds.device, prompt_embeds.dtype)
            extra.append(adapter(cond, adapter_weight=adapter_weights.get(name,1.0)))
        if not extra:
            empty=prompt_embeds.new_zeros((prompt_embeds.shape[0],0,prompt_embeds.shape[-1])); return prompt_embeds, empty
        adapter_tokens=torch.cat(extra, dim=1)
        return torch.cat([prompt_embeds, adapter_tokens], dim=1), adapter_tokens
