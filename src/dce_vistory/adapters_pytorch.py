from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn


class CrossAttentionTokenAdapter(nn.Module):
    """
    Lightweight PyTorch adapter that converts control text embeddings into extra SDXL cross-attention tokens.

    The adapter can run without training because it uses SDXL text-encoder embeddings directly.
    The learnable gate and token budget can be trained later.
    """

    def __init__(self, name: str, num_tokens: int, init_logit_gate: float = 2.0):
        super().__init__()
        self.name = name
        self.num_tokens = int(num_tokens)
        self.logit_gate = nn.Parameter(torch.tensor(float(init_logit_gate)))

    def forward(self, condition_embeds: torch.Tensor, adapter_weight: float = 1.0) -> torch.Tensor:
        if condition_embeds.ndim != 3:
            raise ValueError(f"{self.name} expected [B,L,D], got {tuple(condition_embeds.shape)}")

        _, seq_len, _ = condition_embeds.shape

        if seq_len >= self.num_tokens:
            tokens = condition_embeds[:, : self.num_tokens, :]
        else:
            repeat_factor = (self.num_tokens + seq_len - 1) // max(seq_len, 1)
            tokens = condition_embeds.repeat(1, repeat_factor, 1)[:, : self.num_tokens, :]

        gate = torch.sigmoid(self.logit_gate).to(tokens.dtype)
        return tokens * gate * float(adapter_weight)


class ButterflyAdapterStack(nn.Module):
    """
    Four SDXL cross-attention adapters:
    - character adapter
    - world adapter
    - emotion adapter
    - event adapter
    """

    def __init__(
        self,
        character_tokens: int = 8,
        world_tokens: int = 8,
        emotion_tokens: int = 6,
        event_tokens: int = 6,
    ):
        super().__init__()
        self.character_adapter = CrossAttentionTokenAdapter("character_adapter", character_tokens, init_logit_gate=2.0)
        self.world_adapter = CrossAttentionTokenAdapter("world_adapter", world_tokens, init_logit_gate=1.8)
        self.emotion_adapter = CrossAttentionTokenAdapter("emotion_adapter", emotion_tokens, init_logit_gate=2.2)
        self.event_adapter = CrossAttentionTokenAdapter("event_adapter", event_tokens, init_logit_gate=1.6)

    def forward(
        self,
        prompt_embeds: torch.Tensor,
        control_embeds: Dict[str, torch.Tensor],
        adapter_weights: Dict[str, float],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        extra_tokens = []

        adapters = {
            "character_adapter": self.character_adapter,
            "world_adapter": self.world_adapter,
            "emotion_adapter": self.emotion_adapter,
            "event_adapter": self.event_adapter,
        }

        for name, adapter in adapters.items():
            cond = control_embeds.get(name)
            if cond is None:
                continue
            cond = cond.to(prompt_embeds.device, prompt_embeds.dtype)
            weight = adapter_weights.get(name, 1.0)
            extra_tokens.append(adapter(cond, adapter_weight=weight))

        if not extra_tokens:
            empty = prompt_embeds.new_zeros((prompt_embeds.shape[0], 0, prompt_embeds.shape[-1]))
            return prompt_embeds, empty

        adapter_tokens = torch.cat(extra_tokens, dim=1)
        augmented_prompt_embeds = torch.cat([prompt_embeds, adapter_tokens], dim=1)
        return augmented_prompt_embeds, adapter_tokens
