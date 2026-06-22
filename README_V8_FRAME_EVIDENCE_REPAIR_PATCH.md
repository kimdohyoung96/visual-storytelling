# DCEE-CausalVerse v8 frame evidence repair patch

## Problem

The pipeline reached postprocess_storyboard, but frame 6 was missing `key_objects` or `evidence_objects`:

```text
RuntimeError: Storyboard frame 6 missing key_objects/evidence_objects.
```

This means the generated/canonicalized storyboard frame existed, but its evidence fields were omitted.

## Fix

This patch updates only:

- `src/dce_vistory/planner.py`

It repairs missing frame-level objects using existing context:

- current storyboard row
- aligned story sentence
- linked DCEE event
- `must_show`
- seed objects

This is not DummyLLM and not static fallback. It preserves the generated story/event semantics and only normalizes missing evidence fields from already-generated context.

## Apply

Unzip at project root and overwrite:

```text
src\dce_vistory\planner.py
```

## Run

```powershell
$env:PYTHONPATH="src"
python run_crossattn_butterfly_pipeline.py --config configs/crossattn_butterfly_dcee_causalverse.yaml --input examples/sad_1.json --out outputs/DCEE_v8_3_sad_1
```
