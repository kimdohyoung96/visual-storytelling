# DCEE-CausalVerse V22 Story-Locked Candidate Selector Patch

Base patch: `DCEE_CausalVerse_V21_1_import_fix_patch.zip`

## What V22 fixes

1. **Duplicate protagonist / two bears problem**
   - Adds strict single-protagonist prompt constraints.
   - Adds duplicate-protagonist negative prompts.
   - Removes prior generated frame image collages from IP-Adapter reference by default because they can amplify previous visual errors.

2. **Wrong candidate selection problem**
   - Reduces image-quality dominance in candidate ranking.
   - Adds story-first ranking: story alignment, action, evidence, emotion, identity, and single-scene score dominate.
   - Penalizes background-only prompt variants when they ignore action.
   - Adds variant priority so `continuity_locked` / `emotion_locked` can beat a prettier but story-wrong `background_locked` image.

3. **Bad required_objects extraction**
   - Stops using noisy phrase extraction from the story sentence as required objects.
   - Filters abstract clauses such as `the loss of the jar of honey`, `heavy heart`, `quietly on the`, etc.
   - Keeps only concrete visual inventory: protagonist, jar, river, riverbank, bush, rocks, trees, etc.

4. **Panda-specific negative prompt removed**
   - Replaces `human face replacing panda` style fixed negatives with generic subject-aware negatives.

## Modified files

- `src/dce_vistory/prompts.py`
- `src/dce_vistory/planner.py`
- `src/dce_vistory/pipeline_crossattn_butterfly.py`
- `src/dce_vistory/frame_director.py`
- `src/dce_vistory/evaluator.py`
- `src/dce_vistory/sdxl_cross_attention_generator.py`

## Run

```powershell
$env:PYTHONPATH="src"
python run_crossattn_butterfly_pipeline.py --config configs/crossattn_butterfly_dcee_causalverse.yaml --input examples/sad_1_b3.json --out outputs/DCEE_v22W_sad_1
```
