# DCEE-CausalVerse V41.1 Text-only Seed Hotfix

## Fixed error
V41 crashed in text-only mode because `pipeline_crossattn_butterfly.py` passed `image_summary` as a plain dict when `image_path` was missing. If the planner's strict JSON seed generation failed, its fallback accessed `image_summary.setting`, which caused:

```text
AttributeError: 'dict' object has no attribute 'setting'
```

## Fix
V41.1 changes the text-only `image_summary` to an attribute-compatible `SimpleNamespace` with:

- `caption`
- `objects`
- `scene`
- `setting`
- `notes`

This keeps the no-image-path text-only behavior while making planner fallback safe.

## Run
```bash
python run_v41_1.py \
  --config configs/config_v41_1_text_only_seed_hotfix.yaml \
  --input examples/white_bear_text_only_v41.json \
  --out outputs/DCEE_v41_1_white_bear_text_only
```

You can also use the existing `run_v41.py`; the key fix is in `src/dce_vistory/pipeline_crossattn_butterfly.py`.
