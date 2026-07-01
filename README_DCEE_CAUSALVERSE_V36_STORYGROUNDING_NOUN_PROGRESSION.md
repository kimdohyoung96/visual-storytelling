# DCEE-CausalVerse V36 Patch

## What V36 changes
1. **Story-grounded visual storytelling strengthened**
   - Better prompt contracts for text + image + protagonist + situation inputs.
   - Each frame is pushed to behave like the exact current story step, not a generic character portrait.

2. **Critical noun coverage added**
   - Added detection / scoring / penalty for core nouns and scene props.
   - Examples: `jar`, `roots`, `slope`, `lake`, `forest`, and similar current-step objects.
   - Missing critical nouns now lowers ranking and can trigger retry.

3. **Progression enforcement strengthened**
   - Added previous/next-step aware prompt language.
   - Added progression consistency score and stronger repeated-scene penalty.
   - Retry now triggers when the selected frame is too static or weakly grounded.

4. **Background + scene grounding kept strong**
   - Non-empty, story-matching background is still required.
   - Candidate selection still penalizes blank / gray / generic backgrounds.

## Main files changed
- `src/dce_vistory/frame_director.py`
- `src/dce_vistory/sdxl_cross_attention_generator.py`
- `src/dce_vistory/evaluator.py`
- `src/dce_vistory/pipeline_crossattn_butterfly.py`
- `config_v36.yaml`

## Recommended optional assets
For even stronger protagonist consistency, optionally prepare one or more of:
- `canonical_reference_sheet_path`
- `character_lora_path`
- `instantid_adapter_path` + `instantid_controlnet_path`
- `photomaker_adapter_path`

Without those, V36 still runs in IP-Adapter + text identity-lock mode.
