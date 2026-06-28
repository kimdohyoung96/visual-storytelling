# DCEE-CausalVerse V29: Protagonist-Only Story-Faithful Patch

This patch updates the V28 caption-grounded pipeline to fix the main failure mode observed in recent runs:
(1) the story introduces or allows extra agents, and
(2) the image generator drifts away from the current caption.

## Key changes

### 1) Protagonist-only story generation is enforced more strictly
- Expanded forbidden-agent vocabulary in `planner.py`.
- Added secondary-agent validation for both Korean and English frame captions.
- Every generated step must include a concrete location.
- Every frame step is hard-checked again after protagonist-only sanitization.

### 2) Caption is treated as the image contract
- `prompts.py` now gives a stronger frame-step instruction:
  - exactly one protagonist,
  - one single visible action,
  - one scene,
  - environment/object/internal-emotion conflict only,
  - no new agents.

### 3) Safer frame prompt construction
- `frame_director.py` now emphasizes:
  - single scene,
  - one visible protagonist only,
  - full-body / medium-wide readability,
  - uncropped face/feet/paws,
  - no extra humans/animals,
  - no multi-panel or poster-like portrait.

### 4) Safer SDXL reference policy
- `sdxl_cross_attention_generator.py` now prefers only the protagonist/source reference image.
- It no longer combines multiple memory images as the main identity reference by default.
- Candidate prompt modes are reduced to safer story-faithful modes.

### 5) Stricter candidate reranking
- `evaluator.py` increases the penalty for extra humans/animals and multi-subject captions.
- Story/event/evidence alignment now has more influence on final selection.

### 6) Previous selected image feedback is reused
- `pipeline_crossattn_butterfly.py` stores the selected candidate caption/score feedback into `previous_frame`.
- The next story step can therefore continue the story while correcting visual drift.

## Modified files
- `src/dce_vistory/prompts.py`
- `src/dce_vistory/planner.py`
- `src/dce_vistory/frame_director.py`
- `src/dce_vistory/sdxl_cross_attention_generator.py`
- `src/dce_vistory/evaluator.py`
- `src/dce_vistory/pipeline_crossattn_butterfly.py`

## Expected effect
- Stories stay protagonist-only.
- Captions are easier to render as exact frames.
- The selected image should better match the current caption.
- Extra humans/animals should be strongly discouraged both during generation and during candidate selection.
