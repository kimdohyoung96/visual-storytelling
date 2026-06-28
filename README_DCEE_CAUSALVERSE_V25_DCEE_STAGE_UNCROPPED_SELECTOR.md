# DCEE-CausalVerse V25 DCEE Stage + Uncropped Selector Patch

## Why V25

V24.1 only fixed the validator. It did not necessarily change image generation enough, and with the same SDXL seed old-looking images can repeat if the prompt and seed path are effectively unchanged.

V25 fixes three concrete problems:

1. **V23/V24-like identical images**
   - Adds prompt-hash seed policy:
     `seed = base_seed + 250000 + frame_id*1000 + candidate_id + prompt_hash`
   - Cleans old frame images/contact sheet before each run in the output folder.
   - This prevents stale outputs and makes changed prompts actually produce changed candidates.

2. **Images do not follow DCEE story flow**
   - Adds explicit DCEE stage scaffolding for each frame:
     - Frame 1: Desire setup
     - Frame 2: Conflict trigger
     - Frame 3: Escalating event/action
     - Frame 4: Consequence event/evidence
     - Frame 5: Turning point with visible evidence
     - Frame 6: Ending emotion with visible consequence
   - Every frame must satisfy:
     `protagonist + action + cause/evidence + required objects + background + emotion`.

3. **Bad crops / cut-off face or feet**
   - Adds centered full-body / mostly full-body composition constraints.
   - Adds negative prompts:
     `cropped head`, `cropped face`, `cropped feet`, `cut off body`, `partial body`, `out of frame`.
   - Evaluator penalizes cropped candidates through `crop_penalty` and `full_body_visibility`.

## Modified files

- `src/dce_vistory/prompts.py`
- `src/dce_vistory/planner.py`
- `src/dce_vistory/pipeline_crossattn_butterfly.py`
- `src/dce_vistory/frame_director.py`
- `src/dce_vistory/sdxl_cross_attention_generator.py`
- `src/dce_vistory/evaluator.py`

## Important

Candidate generation is still preserved. V25 does **not** force one image per frame. It still generates candidates and selects the best one, but now:

- stale images are deleted before a run
- prompt changes produce different seeds
- candidate selection penalizes duplicate/cropped/wrong-event images
- DCEE event flow is explicit

## Apply

```powershell
cd C:\Users\nkm11\visual-storytelling
Expand-Archive -Path .\DCEE_CausalVerse_V25_dcee_stage_uncropped_selector_patch.zip -DestinationPath . -Force
```

## Run with a fresh output folder

```powershell
$env:PYTHONPATH="src"
python run_crossattn_butterfly_pipeline.py --config configs/crossattn_butterfly_dcee_causalverse.yaml --input examples/sad_1_b3.json --out outputs/DCEE_v25_W_sad_1
```

## Check files

- `generation_policy_V25.json`
- `candidate_manifest.json`
- `selected_images.json`
- `storyboard.json`
- `contact_sheet.png`

In `candidate_manifest.json`, candidates should include:

- `v25_selection_reason`
- `v25_vlm_judgment` if VLM is active
- `v25_pairwise_selection` if pairwise VLM selection is active
- `prompt_hash_seed_offset`
- `v25_seed_policy`

## Expected improvement

V25 should reduce:
- old repeated V23/V24 outputs
- duplicate white bear / small second bear
- cropped body / cropped face / cropped feet
- generic portrait frames
- frames that do not show the story event
