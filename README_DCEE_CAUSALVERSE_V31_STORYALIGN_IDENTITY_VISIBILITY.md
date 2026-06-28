# DCEE-CausalVerse V31: Story-Alignment / Identity-Consistency / Visibility-Rescue Patch

## Why V31
V31 directly addresses the three failure modes observed in generated contact sheets:
1. images not following the generated story/caption,
2. protagonist identity drifting across frames,
3. protagonist visibility collapsing in dark emotional scenes.

## Main upgrades
### 1) Stronger story-to-image alignment
- caption-locked rendering remains the core contract
- `prompt_2` now repeats the exact frame caption, current action, visible cause/evidence, location, weather, atmosphere, and required objects
- retry logic is stricter for story alignment, event grounding, and evidence visibility

### 2) Stronger protagonist consistency
- subject reference image remains active
- later frames can now use **previous-frame img2img continuity generation**
- candidate set mixes:
  - text2img candidates for event freshness
  - previous-frame img2img candidates for identity continuity
- selection favors identity consistency while still prioritizing story alignment

### 3) Better readability in dark scenes
- prompt includes explicit visibility rules for sad / empty / dark moods
- protagonist must remain readable with rim light, silhouette separation, and visible face/body
- automatic post-generation visibility rescue lightly lifts underexposed images
- evaluation adds `subject_visibility` and `crop_penalty`

## Modified files
- `src/dce_vistory/frame_director.py`
- `src/dce_vistory/sdxl_cross_attention_generator.py`
- `src/dce_vistory/evaluator.py`
- `src/dce_vistory/pipeline_crossattn_butterfly.py`

## New / important config fields
```yaml
image_generator:
  model_id: stabilityai/stable-diffusion-xl-base-1.0
  quality_model_preset: sdxl_base   # or juggernaut_xl / realvis_xl
  num_inference_steps: 44
  guidance_scale: 9.0
  use_refiner: true
  refiner_model_id: stabilityai/stable-diffusion-xl-refiner-1.0
  refiner_strength: 0.80
  use_previous_frame_img2img: true
  previous_frame_strength: 0.40

pipeline:
  story_alignment_threshold: 0.82
  subject_visibility_threshold: 0.58
  crop_penalty_threshold: 0.14
```

## Expected effect
- closer frame-to-caption matching,
- stronger protagonist continuity across frames,
- fewer unreadable dark frames,
- fewer generic sitting/standing frames,
- fewer extra subjects.
