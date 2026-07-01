# DCEE-CausalVerse V37 Patch

## What V37 changes
1. **Two-pass pipeline**
   - First pass generates the full story sequence and frame specs.
   - Second pass generates images with access to previous story summary, previous frame result, and next-frame hint.

2. **Identity backends included in the design**
   - Added routing and config hooks for:
     - `InstantID`
     - `PhotoMaker`
     - `character LoRA`
     - `canonical reference sheet`
   - V37 builds a **composite protagonist reference bank** from canonical sheet, uploaded protagonist image(s), and previous selected frame.

3. **Previous story + previous frame continuity strengthened**
   - Prompts now include:
     - previous story summary
     - previous frame caption
     - previous frame visual caption
     - next frame caption hint
   - Added `story_context_alignment` scoring.

4. **Visual storytelling grounding kept strong**
   - Critical noun coverage, progression consistency, and non-empty background checks remain active.

## Main files changed
- `src/dce_vistory/pipeline_crossattn_butterfly.py`
- `src/dce_vistory/sdxl_cross_attention_generator.py`
- `src/dce_vistory/evaluator.py`
- `config_v37.yaml`
- `run_v37.py`

## Optional assets for best V37 performance
Prepare one or more of the following if available:
- `canonical_reference_sheet_path`
- `protagonist_reference_paths` (multiple protagonist images)
- `character_lora_path`
- `instantid_adapter_path` + `instantid_controlnet_path`
- `photomaker_adapter_path`

If those are not provided, V37 still runs with IP-Adapter + text identity lock + previous-frame continuity.
