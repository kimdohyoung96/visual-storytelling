# DCE-ViStory Butterfly Adapter Design

This design upgrades DCE-ViStory from prompt-only visual storytelling into a ViSTA-inspired,
adapter-ready architecture.

Pipeline:

Input image/text/protagonist/ending emotion
→ DCE story planner
→ emotion/world storyboard
→ Butterfly Decoder-Encoder-Decoder controller
→ character/world/emotion adapter packets
→ multi-candidate image generation
→ consistency-aware reranking

Butterfly structure:

1. Decoder-1: Narrative Expansion Decoder
   - expands one frame into several visual hypotheses.
   - each hypothesis has event, shot, emotion, weather, time, atmosphere, and background details.

2. Encoder: Consistency Fusion Encoder
   - compresses character, world, emotion, and previous-frame memory into a compact latent.
   - prevents prompt explosion.
   - preserves identity/world continuity.

3. Decoder-2: Visual Control Decoder
   - converts the compact latent into diffusion-ready prompts/control packets.
   - can later be connected to IP-Adapter, ControlNet, LoRA, learned cross-attention adapters, or a ViSTA-style multimodal adapter.

Why not only encoder-decoder?

A simple encoder-decoder is good for direct translation, but visual storytelling is not direct translation.
It needs divergent scene imagination first, then consistency fusion, then image control.
So the Decoder-Encoder-Decoder design is better for this project.
