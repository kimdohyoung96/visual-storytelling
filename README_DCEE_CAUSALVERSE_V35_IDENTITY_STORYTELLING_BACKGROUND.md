# DCEE-CausalVerse V35 Patch

## Main changes
1. **Identity backend hooks**
   - Added optional config/code hooks for canonical reference sheet, character LoRA, InstantID, and PhotoMaker.
   - The patch still runs even if these are disabled.

2. **Stronger visual storytelling**
   - Added `storytelling_contract` so each frame behaves like one progressing story scene.
   - Added progression-aware scoring and static-repeat penalty.

3. **Background generation control**
   - Added `background_contract` and background-presence scoring.
   - Added blank-background penalty and retry triggers.

4. **V35 retry logic**
   - Retries now trigger not only on weak story/event/evidence/emotion, but also on weak background presence, weak storytelling progression, blank background, and static repeated scene.

## Files to replace
- `src/dce_vistory/frame_director.py`
- `src/dce_vistory/sdxl_cross_attention_generator.py`
- `src/dce_vistory/evaluator.py`
- `src/dce_vistory/pipeline_crossattn_butterfly.py`

## Optional assets needed to fully activate advanced identity backends
If you want the full V35 identity backend path beyond basic IP-Adapter, please additionally prepare at least one of the following:
- `canonical_reference_sheet_path`: a clean multi-view or character-sheet image of the protagonist
- `character_lora_path`: LoRA weights trained for the protagonist
- `instantid_adapter_path` and `instantid_controlnet_path`
- `photomaker_adapter_path`

Without those assets, V35 still works in IP-Adapter + text identity-lock mode.
