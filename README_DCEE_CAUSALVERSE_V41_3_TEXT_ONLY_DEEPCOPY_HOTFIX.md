# DCEE-CausalVerse V41.3 text-only deepcopy hotfix

## Fixed error
The previous text-only summary hotfix could still crash during `dataclasses.asdict(seed)` with:

```text
TypeError: 'str' object is not callable
```

## Cause
`TextOnlyImageSummary.__getattr__()` returned an empty string for every missing attribute. Python `copy.deepcopy()` probes special methods such as `__deepcopy__`. Because `__getattr__('__deepcopy__')` returned `''`, copy.py tried to call a string.

## Main file changed
- `src/dce_vistory/pipeline_crossattn_butterfly.py`

## Patch
- dunder attributes now raise `AttributeError`
- explicit `__deepcopy__()` added

## Run
```bash
python run_v41_3.py   --config configs/config_v41_3_text_only_deepcopy_hotfix.yaml   --input examples/white_bear_text_only_v41.json   --out outputs/DCEE_v41_3_white_bear_text_only
```
