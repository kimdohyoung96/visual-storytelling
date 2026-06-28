# DCEE-CausalVerse V24.1 Forbidden Validator Fix

## Fixed error

```text
RuntimeError: Strict LLM JSON generation failed at stage=generate_story_step_1.
attempt 1: ValueError: story step contains forbidden ungrounded agents
attempt 2: ValueError: story step contains forbidden ungrounded agents
```

## Cause

V24 prompt asks the LLM to output negative constraint fields:

- `absent_objects`
- `forbidden_visuals`

Those fields are supposed to contain words like:

- duplicate protagonist
- second bear
- extra character
- human
- split panel

But the validator was checking the whole JSON object for forbidden words.
So even correct negative constraints caused the run to fail.

## Fix

`planner.py` now separates:

1. **positive story fields** that will actually be rendered
2. **negative constraint fields** that describe what must not appear

The forbidden-agent validator now checks only positive story fields, not `absent_objects` or `forbidden_visuals`.

## Modified file

```text
src/dce_vistory/planner.py
```

## Apply

```powershell
cd C:\Users\nkm11\visual-storytelling
Expand-Archive -Path .\DCEE_CausalVerse_V24_1_forbidden_validator_fix_patch.zip -DestinationPath . -Force
```

## Run

```powershell
$env:PYTHONPATH="src"
python run_crossattn_butterfly_pipeline.py --config configs/crossattn_butterfly_dcee_causalverse.yaml --input examples/sad_1_b3.json --out outputs/DCEE_v24_1_W_sad_1
```
