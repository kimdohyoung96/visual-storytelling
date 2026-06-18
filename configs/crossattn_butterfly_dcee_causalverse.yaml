# DCEE-CausalVerse STRICT FINAL config: no DummyLLM, no fallback LLM
# DCEE-CausalVerse stable v3 config
llm:
  provider: openrouter
  model: openai/gpt-4o-mini
  temperature: 0.35
  max_tokens: 1600

vlm:
  provider: openrouter
  model: openai/gpt-4o-mini
  temperature: 0.0
  max_tokens: 800

image_understanding:
  provider: llm_caption
  caption_model: Salesforce/blip-image-captioning-base

image_generator:
  provider: sdxl_crossattn
  model_id: stabilityai/stable-diffusion-xl-base-1.0
  device: cuda
  width: 1024
  height: 1024
  num_inference_steps: 40
  guidance_scale: 8.0
  num_candidates_per_frame: 2
  num_ending_candidates: 5
  seed: 42
  enable_cpu_offload: false

cross_attention_adapters:
  character_tokens: 8
  world_tokens: 8
  emotion_tokens: 8
  event_tokens: 8
  evidence_tokens: 8

butterfly:
  num_hypotheses: 3
  quality_suffix: "full-color cinematic storybook illustration, event-grounded, evidence-visible, emotionally expressive, rich natural colors, detailed world, sharp focus"
  negative_prompt: "monochrome, grayscale, black and white, missing event, missing evidence, weak emotion, low quality, blurry"

pipeline:
  memory_strategy: adaptive_causal
  emotion_retry: true
  emotion_visibility_threshold: 0.76
  colorfulness_threshold: 0.35
  event_grounding_threshold: 0.70
  evidence_visibility_threshold: 0.68

evaluation:
  use_vlm: false
  save_contact_sheet: true
