# DCEE-CausalVerse V30: English Caption-Locked Story-Faithful SDXL/Refiner Patch

## Goal
This patch upgrades the V29 protagonist-only pipeline so that story-to-image generation is more faithful to the DCEE storyboard.

## Core changes
1. **English-only text generation**
   - story sentences, captions, and image contracts are generated in English
   - if a story step is returned in Korean, it is translated into English before image generation

2. **Stronger protagonist-only enforcement**
   - story steps reject extra humans, animals, helpers, or duplicate protagonists
   - visual inventory removes agent-like terms and keeps only grounded props/background elements

3. **Stronger caption-locked prompt construction**
   - each frame is treated as an exact image contract
   - prompts emphasize one protagonist, one action, one scene, visible evidence, uncropped composition, and no unrelated objects

4. **Improved SDXL generation**
   - optional SDXL refiner stage added for higher-quality final rendering
   - stronger prompt / prompt_2 pairing
   - DPM++-style scheduler with Karras sigmas
   - optional backbone replacement remains possible through `model_id`

5. **Better story-faithful reranking**
   - extra-subject penalty increased
   - selected notes now include `v30_selection_reason`

## Modified files
- `src/dce_vistory/prompts.py`
- `src/dce_vistory/planner.py`
- `src/dce_vistory/frame_director.py`
- `src/dce_vistory/sdxl_cross_attention_generator.py`
- `src/dce_vistory/pipeline_crossattn_butterfly.py`
- `src/dce_vistory/evaluator.py`

## Key config knobs
Inside your image generator config, V30 now supports:
- `model_id`
- `use_refiner`
- `refiner_model_id`
- `refiner_strength`
- `aesthetic_score`
- `negative_aesthetic_score`

Recommended defaults:
```yaml
image_generator:
  model_id: stabilityai/stable-diffusion-xl-base-1.0
  use_refiner: true
  refiner_model_id: stabilityai/stable-diffusion-xl-refiner-1.0
  refiner_strength: 0.80
  num_inference_steps: 40
  guidance_scale: 8.0
```

## Run
```powershell
$env:PYTHONPATH="src"
python run_crossattn_butterfly_pipeline.py --config configs/crossattn_butterfly_dcee_causalverse.yaml --input examples/sad_1_b3.json --out outputs/DCEE_v30_example
```

## Expected outputs
- `generation_policy_V30.json`
- `full_story.json` (English story)
- `storyboard.json` (English frame contracts)
- `candidate_manifest.json`
- `selected_images.json`
- `contact_sheet.png`
- `final_story.md`
