# DCEE-CausalVerse V21.1 Import Fix Patch

## Fixed error

```text
ImportError: cannot import name 'eval_questions_prompt' from 'dce_vistory.prompts'
```

## Cause

`evaluator.py` imports:

```python
from .prompts import SYSTEM_NARRATIVE, SYSTEM_VLM, eval_questions_prompt
```

but the previous V21 `prompts.py` did not include `eval_questions_prompt`.

## Fix

This patch adds backward-compatible prompt functions to:

```text
src/dce_vistory/prompts.py
```

including:

- `eval_questions_prompt`
- `frame_prompt`
- `dcee_branch_plan_prompt`
- `dcee_candidate_selection_prompt`
- `dce_plan_prompt`
- `emotion_delta_text`
- `storyboard_prompt`
- `canonicalize_storyboard_prompt`

## Apply

```powershell
cd C:\Users\nkm11\visual-storytelling
Expand-Archive -Path .\DCEE_CausalVerse_V21_1_import_fix_patch.zip -DestinationPath . -Force
```

## Run

```powershell
$env:PYTHONPATH="src"
python run_crossattn_butterfly_pipeline.py --config configs/crossattn_butterfly_dcee_causalverse.yaml --input examples/sad_1_b3.json --out outputs/DCEE_v21W_sad_2
```
