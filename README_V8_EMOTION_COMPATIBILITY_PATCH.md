# DCEE-CausalVerse v8 emotion compatibility patch

## Problem

The API generated an emotion arc ending with `regret` while the input target was `sad`.

```text
RuntimeError: Final emotion state does not match target. final=regret, target=sadness
```

This is too strict. In visual storytelling, `regret` is a valid fine-grained subtype of a sad ending.

## Fix

This patch updates `src/dce_vistory/planner.py`.

It adds semantic emotion-family compatibility:

- sad / sadness accepts: regret, grief, sorrow, melancholy, despair, disappointment, remorse, guilt
- happy / joy accepts: relief, gratitude, hope, delight, contentment, satisfaction, triumph
- fear accepts: anxiety, dread, terror, worry, panic
- anger accepts: frustration, resentment, rage

If the final emotion is compatible, the code:
1. preserves the fine-grained LLM label as `final_emotion_subtype`
2. normalizes the final `states[-1]` to the requested target label for downstream checks

Example:
```json
{
  "target_ending_emotion": "sadness",
  "final_emotion_subtype": "regret",
  "states": ["concern", "...", "sadness"]
}
```

If the final emotion is truly incompatible, the code asks the API to repair the emotion arc once.

This is not DummyLLM and not static fallback. It is semantic validation for target emotion families.

## Apply

Unzip at the project root and overwrite:

```text
src\dce_vistory\planner.py
```

## Run

```powershell
$env:PYTHONPATH="src"
python run_crossattn_butterfly_pipeline.py --config configs/crossattn_butterfly_dcee_causalverse.yaml --input examples/sad_1.json --out outputs/DCEE_v8_sad_2
```
