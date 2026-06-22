# DCEE-CausalVerse v7 symbolic_objects type-fix patch

## Problem

The pipeline reached visual control packet creation, but `seed.visual_symbols` was a list.
`WorldLatent.to_prompt()` expects `symbolic_objects` to be a dictionary and calls `.items()`.

Error:

```text
AttributeError: 'list' object has no attribute 'items'
```

## Fix

This patch normalizes `visual_symbols` / `symbolic_objects` before passing them to `WorldLatent`.

Accepted input shapes:
- dict
- list[str]
- list[dict]
- string

This is not DummyLLM and not static fallback. It is a type normalization fix for API JSON variance.

## Changed file

- `src/dce_vistory/butterfly_adapter.py`

## Apply

Unzip at your project root and overwrite:

```text
src\dce_vistory\butterfly_adapter.py
```

## Run

```powershell
$env:PYTHONPATH="src"
python run_crossattn_butterfly_pipeline.py --config configs/crossattn_butterfly_dcee_causalverse.yaml --input examples/sad_1.json --out outputs/DCEE_v7_sad_2
```
