# DCEE-CausalVerse v14 Image-Friendly Story Patch

## Problem

The story was sentence-locked, but each generated sentence was still too literary/complex for SDXL.
A sentence such as "the panda realizes the weight of loss in his heart" is meaningful for humans,
but hard to draw faithfully.

## Core idea

Before image generation, rewrite the full story into **image-friendly visual sentences**.

The paper pipeline becomes:

Input image/text
→ DCEE story planning
→ literary full story
→ image-friendly sentence rewrite
→ sentence 1 -> frame 1
→ sentence 2 -> frame 2
→ ...
→ sentence 6 -> ending image

## New rules for the six story sentences

Each sentence must:
- be 8 to 16 words,
- contain exactly one visible action,
- contain one subject,
- contain one main object,
- contain one visible background/location,
- contain one visible emotion cue,
- avoid abstract/internal words such as honesty, integrity, realizes, understands, importance, moral, choice, fate, heart,
- convert abstract meaning into visible objects/actions.

Example:

Bad:
`The panda realizes honesty matters as his heart fills with sorrow.`

Good:
`The panda kneels beside broken bamboo stumps in the rain.`

## Files changed

- `src/dce_vistory/visual_sentence_planner.py`
- `src/dce_vistory/prompts.py`
- `src/dce_vistory/planner.py`
- `src/dce_vistory/frame_director.py`
- `src/dce_vistory/pipeline_crossattn_butterfly.py`
- `RECOMMENDED_CONFIG_IMAGE_FRIENDLY_STORY.yaml`

## Apply

Unzip at project root and overwrite files.

## Run

```powershell
$env:PYTHONPATH="src"
python run_crossattn_butterfly_pipeline.py --config configs/crossattn_butterfly_dcee_causalverse.yaml --input examples/sad_1_p2.json --out outputs/DCEE_v14_image_friendly_sad_1
```

## Check

Open `full_story.json`. It should contain:

```json
"story_mode": "image_friendly_sentence_locked",
"original_sentences": [...],
"sentences": [
  {
    "sentence": "simple drawable sentence",
    "subject": "...",
    "action": "...",
    "object": "...",
    "location": "...",
    "emotion": "...",
    "required_objects": [...]
  }
]
```
