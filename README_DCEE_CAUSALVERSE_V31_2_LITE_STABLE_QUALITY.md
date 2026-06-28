# DCEE-CausalVerse V31.2-Lite Stable Quality Patch

## Why this patch
The previous V31 line added too many experimental components:
- previous-frame img2img continuity
- multi-reference image composition
- default SDXL refiner loading
- long free-form prompts
- many prompt modes and extra metadata

Those components can reduce image quality, distort the protagonist, and make the generated frame less faithful to the current story caption.

V31.2-Lite removes those unstable parts and keeps only the parts that directly help the current DCEE pipeline.

## Main direction
- Keep the V30 English caption-grounded story pipeline.
- Generate each image from the current frame caption only.
- Keep the prompt short and caption-first to avoid SDXL/CLIP prompt truncation.
- Use only the input/source protagonist image as the identity reference.
- Do not use previous-frame img2img by default.
- Do not use refiner by default.
- Keep multiple candidates and select the best via evaluator.

## Key changes

### 1. SDXL wrapper simplified
Modified:
- `src/dce_vistory/sdxl_cross_attention_generator.py`

Removed from V31:
- previous-frame img2img
- multi-reference collage
- default refiner
- long free-form frame_director prompt as actual SDXL prompt

Kept:
- English caption contract
- subject/source reference only
- compact caption-first prompt
- token audit log
- multiple candidate generation

### 2. Prompt length guard
Each prompt is built from short ordered segments:
1. exact frame caption
2. protagonist
3. identity
4. action
5. evidence
6. required objects
7. location/emotion
8. no extra subjects

The generator greedily fits these segments into the SDXL tokenizer window.

### 3. Short negative prompt
Negative prompt is shortened to critical failure terms only:
- human/person/child
- extra animal
- duplicate protagonist
- wrong event
- missing action/evidence
- cropped body
- underexposed protagonist
- generic portrait

### 4. Safer defaults
Recommended defaults:
```yaml
image_generator:
  num_inference_steps: 36
  guidance_scale: 7.5
  use_refiner: false
  use_ip_adapter: true
  ip_adapter_scale: 0.26
  quality_model_preset: sdxl_base
```

### 5. Output logs
Each candidate contains:
```json
"generator_version": "V31.2-Lite",
"token_report": {...},
"removed_v31_components": [
  "previous_frame_img2img",
  "multi_reference_collage",
  "default_refiner",
  "long_freeform_prompt"
]
```

## Modified files
- `src/dce_vistory/prompts.py`
- `src/dce_vistory/planner.py`
- `src/dce_vistory/frame_director.py`
- `src/dce_vistory/sdxl_cross_attention_generator.py`
- `src/dce_vistory/pipeline_crossattn_butterfly.py`
- `src/dce_vistory/evaluator.py`

## Apply
```powershell
cd C:\Users\nkm11\visual-storytelling
Expand-Archive -Path .\DCEE_CausalVerse_V31_2_lite_stable_quality_patch.zip -DestinationPath . -Force
```

## Run
```powershell
$env:PYTHONPATH="src"
python run_crossattn_butterfly_pipeline.py --config configs/crossattn_butterfly_dcee_causalverse.yaml --input examples/sad_1_b3.json --out outputs/DCEE_v31_2_lite_sad_1
```

## Check
- `generation_policy_V31_2_lite.json`
- `candidate_manifest.json`
- `selected_images.json`
- candidate notes: `token_report`
