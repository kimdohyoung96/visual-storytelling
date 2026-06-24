# DCEE-CausalVerse v11 Story-Exact Alignment Patch

This patch is for the case where story generation works well, but the generated images do **not** match the generated story.

## Key idea
The old code still behaved too much like generic T2I generation with weak story grounding.
This patch makes image generation depend much more directly on:
1. the exact story sentence,
2. the visible DCEE event,
3. evidence objects / must-show objects,
4. the visible emotional cause,
5. continuity with previous frames.

## What changed

### 1) New helper: `story_visual_alignment.py`
Builds a compact **visual contract** from each storyboard frame and creates prompt variants:
- `event_first`
- `evidence_first`
- `emotion_first`
- `continuity_first`

### 2) `sdxl_cross_attention_generator.py`
Instead of one generic compact prompt, each candidate uses a different story-focused prompt variant.
This makes at least some candidates emphasize:
- action/event,
- visible evidence,
- emotional cause,
- continuity.

### 3) `butterfly_adapter.py`
Adds richer structured metadata into `control_metadata`:
- `frame`
- `visual_focus`
- `camera_shot`
- `key_objects`
- `story_sentence`
- `must_show`
- world / emotion / character blocks

### 4) `evaluator.py`
Adds stronger story-faithfulness reranking.
If `use_vlm` is available, the VLM judges story/event/evidence alignment.
If not, a local BLIP caption scorer is used to compare image captions with the story sentence.

### 5) `pipeline_crossattn_butterfly.py`
- more candidates per frame
- stronger thresholds for retry
- retry explicitly asks that action-defining hands / props / objects stay visible

## Files included
- `src/dce_vistory/story_visual_alignment.py`
- `src/dce_vistory/butterfly_adapter.py`
- `src/dce_vistory/sdxl_cross_attention_generator.py`
- `src/dce_vistory/evaluator.py`
- `src/dce_vistory/pipeline_crossattn_butterfly.py`
- `RECOMMENDED_CONFIG_APPEND.yaml`

## Apply
Unzip at the project root and overwrite existing files.

## Run
```powershell
$env:PYTHONPATH="src"
python run_crossattn_butterfly_pipeline.py --config configs/crossattn_butterfly_dcee_causalverse.yaml --input examples/sad_1.json --out outputs/DCEE_v11_sad_1
```

## Important
If your VLM endpoint works, set:
- `evaluation.use_vlm: true`

If VLM is unavailable, this patch still helps because it also adds a local BLIP caption-based scoring route.
