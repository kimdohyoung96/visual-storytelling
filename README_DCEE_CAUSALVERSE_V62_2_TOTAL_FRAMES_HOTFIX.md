# DCEE-CausalVerse V62.2 — total_frames Hotfix

## Fixed error

V62.1 failed with:

```text
UnboundLocalError: cannot access local variable 'total_frames' where it is not associated with a value
```

Cause:
`precomputed_story_steps = _build_precomputed_story_steps(sample, total_frames)` was executed before `total_frames` was initialized.

Fix:
`total_frames = int(sample.get("num_frames", 6))` is now initialized before strict JSON story-flow preprocessing.

## Run

```bash
python run_v62_2.py \
  --config configs/config_v62_2_story_exact_selection_hotfix.yaml \
  --input examples/white_bear_text_only_v62_2.json \
  --out outputs/DCEE_v62_2_white_bear_text_only
```
