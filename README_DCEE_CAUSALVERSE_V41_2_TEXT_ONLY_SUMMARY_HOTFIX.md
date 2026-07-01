# DCEE-CausalVerse V41.2 text-only summary hotfix

## Fixed error
V41.1 fixed `image_summary.setting`, but the planner fallback also accessed `image_summary.mood`.
This caused:

```text
AttributeError: 'types.SimpleNamespace' object has no attribute 'mood'
```

## What changed
`src/dce_vistory/pipeline_crossattn_butterfly.py` now uses `TextOnlyImageSummary`, an attribute-compatible object that contains:

- `caption`
- `summary`
- `description`
- `objects`
- `object_candidates`
- `key_objects`
- `visible_objects`
- `scene`
- `setting`
- `mood`
- `style`
- `protagonist`
- `genre`
- `signature_items`

It also has `__getattr__` defaults so text-only runs do not crash if planner fallback accesses another image-summary field.

## Recommended run

```bash
python run_v41_2.py \
  --config configs/config_v41_2_text_only_summary_hotfix.yaml \
  --input examples/white_bear_text_only_v41.json \
  --out outputs/DCEE_v41_2_white_bear_text_only
```
