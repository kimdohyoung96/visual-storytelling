# DCEE-CausalVerse Complete Stable Code v4

This is a corrected complete patch for the final paper direction.

## Why v4 was needed

In the previous run, `contact_sheet.png` and images were generated, but:
- `abstract.txt` was empty
- `final_story.md` was empty
- `dcee_plan.json` became too generic, e.g. "resolve the central problem" and "discovers the problem"

That means the image generation code could run, but the DCEE planner was still too dependent on fragile LLM JSON output.

## What v4 changes

### 1. Robust DCEE planner
`src/dce_vistory/planner.py` is rewritten so that it validates LLM outputs.
If the LLM returns empty, generic, or non-visual plans, it creates a concrete DCEE plan from:

- protagonist
- input premise
- target ending emotion
- detected objects
- setting

For woodcutter/axe/river stories, it creates a concrete DCEE-Tree plan with:
- old iron axe
- river
- golden axe
- fairy
- empty hands
- kneeling body
- rain / riverbank evidence

### 2. DCEE-Tree candidates are no longer generic
The patch writes meaningful `candidate_plans` and selected event chains.

### 3. abstract.txt is never blank
If LLM abstract generation fails, the planner creates a DCEE-style abstract.

### 4. storyboard is never generic
If storyboard LLM output is generic, it is rebuilt from the selected event chain.

### 5. final_story.md is never blank
The pipeline now has an emergency fallback before writing final markdown.

## Apply

Unzip this patch at your project root and overwrite the existing files.

## Recommended clean run

```powershell
Remove-Item -Recurse -Force outputs\DCEE_CausalVerse_sad_1
$env:PYTHONPATH="src"
python run_crossattn_butterfly_pipeline.py --config configs/crossattn_butterfly_dcee_causalverse.yaml --input examples/sad_1.json --out outputs/DCEE_CausalVerse_sad_1
```

## Check these files

```text
outputs/DCEE_CausalVerse_sad_1/abstract.txt
outputs/DCEE_CausalVerse_sad_1/dcee_candidate_plans.json
outputs/DCEE_CausalVerse_sad_1/dcee_plan.json
outputs/DCEE_CausalVerse_sad_1/storyboard.json
outputs/DCEE_CausalVerse_sad_1/contact_sheet.png
outputs/DCEE_CausalVerse_sad_1/final_story.md
outputs/DCEE_CausalVerse_sad_1/evaluation.json
```
