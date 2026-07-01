# DCEE-CausalVerse V39 Patch

## Main idea
When end-to-end story-scene prompting keeps failing, V39 adds a fallback generation strategy:
1. generate the **protagonist only** according to the current DCEE state,
2. generate the **background/story scene plate** according to the current story sentence,
3. **fuse** the subject layer and the story background into one final frame.

## What changed
- Added optional `subject_scene_fusion` path in `sdxl_cross_attention_generator.py`
- Uses a **single stable subject reference** for protagonist identity
- Background plate prompt explicitly requests:
  - current story location
  - current story props / nouns
  - no protagonist in the background plate
  - empty space for subject placement
- Foreground prompt explicitly requests:
  - exactly one protagonist
  - full body
  - DCEE state / pose / expression
  - simple clean background for extraction
- Added simple edge-based foreground extraction and composition
- Keeps the normal scene-first path as a fallback candidate mode

## Files changed
- `src/dce_vistory/sdxl_cross_attention_generator.py`
- `src/dce_vistory/pipeline_crossattn_butterfly.py`
- `config_v39.yaml`
- `run_v39.py`

## Suggested run command
```bash
python run_v39.py --config configs/config_v39.yaml --input examples/happy_1_b3_short.json --out outputs/DCEE_v39_happy_1
```
