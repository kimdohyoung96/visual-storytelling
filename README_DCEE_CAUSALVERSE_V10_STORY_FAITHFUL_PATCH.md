# DCEE-CausalVerse v10 Story-Faithful Patch

This patch improves visual-storytelling faithfulness when generated images do not follow the story content.

## What was wrong in the old scaffold?

The previous code could generate visually decent images, but some frames behaved like generic portraits or repeated scenes because:
1. history was too weak,
2. the frame prompt did not prioritize the story sentence strongly enough,
3. candidate reranking did not sufficiently reward story-event alignment,
4. retry generation mostly strengthened emotion/evidence, but not story-beat matching.

## Main improvements in this patch

### 1) ViSTA-inspired salient history selection
- `causal_memory.py`
- selects salient previous story-image memories and returns:
  - `salient_history`
  - `continuity_constraints`
- used in the current frame prompt

### 2) StoryGen / Make-A-Story inspired auto-regressive continuity
- `butterfly_adapter.py`
- current frame prompt explicitly includes:
  - story sentence
  - DCEE event
  - evidence objects
  - previous-frame progression constraint
  - same-protagonist identity lock
- prompt now tells the model to advance the story, not repeat a static pose

### 3) TIFA-style story-faithfulness reranking
- `evaluator.py`
- adds frame-level QA-style questions such as:
  - does the image match the story sentence?
  - is the event visible?
  - are evidence objects visible?
  - is the emotional cause visible?
  - does the protagonist remain the same person?
- reranking weights now emphasize:
  - `story_alignment`
  - `event_alignment`
  - `event_grounding`
  - `evidence_visibility`
  - `emotion_visibility`
  - `continuity`

### 4) stronger retry packet
- `pipeline_crossattn_butterfly.py`
- low-scoring frames are regenerated using a stronger packet that explicitly repairs:
  - story sentence mismatch
  - generic portrait outputs
  - repeated composition
  - weak event/evidence visibility

## Files changed
- `src/dce_vistory/causal_memory.py`
- `src/dce_vistory/butterfly_adapter.py`
- `src/dce_vistory/evaluator.py`
- `src/dce_vistory/pipeline_crossattn_butterfly.py`

## How to apply
Unzip at the project root and overwrite the existing files.

## Run
```powershell
$env:PYTHONPATH="src"
python run_crossattn_butterfly_pipeline.py --config configs/crossattn_butterfly_dcee_causalverse.yaml --input examples/sad_1.json --out outputs/DCEE_v10_sad_1
```

## Recommended settings
- `evaluation.use_vlm: true` if your VLM endpoint works
- `image_generator.num_candidates_per_frame: 3`
- `image_generator.num_ending_candidates: 5`
- keep full-color style prompts

## Research positioning
This patch moves the code closer to the paper direction:
- DCEE causal planning
- ViSTA-like salient history
- StoryGen-like continuity
- TIFA-like faithfulness evaluation
without abandoning your DCEE-specific novelty.
