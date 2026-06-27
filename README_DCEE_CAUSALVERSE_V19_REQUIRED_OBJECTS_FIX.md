# DCEE-CausalVerse V19 Required Objects Fix Patch

## Error fixed

The V19 run failed at:

```text
RuntimeError: Strict LLM JSON generation failed at stage=generate_story_step_1.
attempt 1: ValueError: required_objects empty
attempt 2: ValueError: required_objects empty
```

## Cause

The V19 planner correctly forced protagonist-only generation, but the validator was too strict.
If the LLM returned a valid protagonist-centered sentence with `required_objects: []`, the run stopped before the code could repair it.

## Fix

This patch modifies only:

```text
src/dce_vistory/planner.py
```

The fix adds:

```python
_derive_required_objects(...)
_derive_background_elements(...)
```

Now required objects are reconstructed from grounded fields:

- protagonist
- object
- location
- background_elements
- visible_cause
- seed.objects
- seed.setting
- seed.world_context
- raw_input objects / signature_items

The run no longer fails only because `required_objects` is empty.

## Apply

Unzip at the project root and overwrite.

```powershell
cd C:\Users\nkm11\visual-storytelling
Expand-Archive -Path .\DCEE_CausalVerse_V19_required_objects_fix_patch.zip -DestinationPath . -Force
```

## Run

```powershell
$env:PYTHONPATH="src"
python run_crossattn_butterfly_pipeline.py --config configs/crossattn_butterfly_dcee_causalverse.yaml --input examples/sad_1_b1.json --out outputs/DCEE_v19_sad_2
```
