# DCEE-CausalVerse V23 Consistory-Lite Event-Grounded Patch

## Why V23

V22 improved identity prompts, but the generated frames still had two major failures:

1. Some frames produced duplicate protagonists or extra subjects.
2. Candidate selection still sometimes chose a more visually pleasing image over the image that best matched the story event.

V23 fixes the pipeline around three principles:

- **ConsiStory-lite subject anchoring**: preserve only the recurring protagonist identity across frames; do not copy prior generated backgrounds or mistakes.
- **Event-grounded frame prompts**: every candidate prompt prioritizes exact event/action/evidence over background beauty.
- **VLM story-first candidate selection**: selection heavily penalizes duplicate protagonist, missing action, missing evidence, split-panel layout, and extra characters.

## Paper connection

This patch is inspired by "Training-Free Consistent Text-to-Image Generation" / ConsiStory:

- subject-specific consistency across a set of prompts
- avoiding full background sharing
- anchor-based subject reuse
- balancing subject consistency with prompt alignment

This implementation does **not** implement full SDSA / DIFT feature injection because that requires U-Net attention and feature hooks. Instead, it applies the useful parts at the system level: subject-only anchoring, event-locked prompts, and strict candidate reranking.

## Modified files

- `src/dce_vistory/prompts.py`
- `src/dce_vistory/planner.py`
- `src/dce_vistory/pipeline_crossattn_butterfly.py`
- `src/dce_vistory/frame_director.py`
- `src/dce_vistory/sdxl_cross_attention_generator.py`
- `src/dce_vistory/evaluator.py`

## Key changes

### 1. Frame prompt is now event-first

Prompt variants are now:

- `event_locked`
- `evidence_locked`
- `emotion_causal_locked`
- `continuity_locked`
- `composition_locked`

The removed weak variants include `background_locked`, which often selected a nice background but missed the event.

### 2. Required objects are cleaned

The object list no longer keeps phrase fragments such as:

- "quietly on the"
- "water with a"
- "staring"
- "heavy"

Only concrete visual elements are kept.

### 3. VLM candidate selector is enabled by default

`pipeline_crossattn_butterfly.py` now defaults to:

```python
use_vlm=bool(ev_cfg.get("use_vlm", True))
```

The evaluator asks whether:

- the image has exactly one protagonist
- the planned action is visible
- evidence objects are visible
- the image is one single coherent scene
- the emotion and emotional cause are visible

If the image has a duplicate protagonist, it receives a strong penalty.

### 4. Candidate selection scores are replaced by VLM values

V22 used optimistic defaults and took `max(default, vlm_score)`.
V23 uses VLM scores directly when available, so a visually good but semantically wrong image loses.

## Run

```powershell
cd C:\Users\nkm11\visual-storytelling
Expand-Archive -Path .\DCEE_CausalVerse_V23_consistory_event_grounded_patch.zip -DestinationPath . -Force

$env:PYTHONPATH="src"
python run_crossattn_butterfly_pipeline.py --config configs/crossattn_butterfly_dcee_causalverse.yaml --input examples/sad_1_b3.json --out outputs/DCEE_v23W_sad_1
```

## Check

After running, inspect:

- `generation_policy_V23.json`
- `candidate_manifest.json`
- `selected_images.json`
- `storyboard.json`
- `contact_sheet.png`

In `candidate_manifest.json`, each candidate should include:

- `v23_selection_reason`
- `v23_vlm_judgment` when VLM is available

If VLM is disabled or unavailable, selection still uses event-first variant priority and local caption fallback, but image-level semantic selection will be weaker.
