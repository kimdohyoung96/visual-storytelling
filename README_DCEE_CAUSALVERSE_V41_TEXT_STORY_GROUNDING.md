# DCEE-CausalVerse V41 text-story-grounding patch

## Why V41 was created
V40 proved that removing `image_path` can prevent reference-image interference, but the resulting images were still weakly grounded to the story. The model could drift to random animals or unrelated objects. V41 strengthens the **story sentence -> image** path.

## Main changes
### 1. Stronger frame-level noun grounding
`frame_director.py`
- Adds `_strong_story_required_objects(...)`
- Injects story nouns such as:
  - `honey jar`
  - `bushes or underbrush`
  - `tangled roots`
  - `small hill or slope`
  - `forest path`
  - `serene lake`
  - `deep forest trees`
- These are added into `required_objects` so the image prompt always receives them.

### 2. Dedicated text-only prompt template
`sdxl_cross_attention_generator.py`
- Adds a **V41 text-only story grounding** prompt path.
- Explicitly says:
  - exactly one **white bear** protagonist
  - the protagonist must clearly be a **white bear**
  - show the **current action**
  - show the **current frame nouns**
  - do **not** invent unrelated animals or objects
- Adds strong negatives against:
  - fox / squirrel / deer / rabbit / wolf
  - brown bear / grizzly / panda
  - book / reading / truck / car / vehicle
  - city / room / classroom / library

### 3. Text-only candidate mode
`sdxl_cross_attention_generator.py`
- In text-only mode, generation uses dedicated `text_story` candidate modes.
- Subject-scene fusion is disabled in text-only mode.

### 4. Stronger candidate selection
`evaluator.py`
- Adds `protagonist_alignment` reward
- Adds `wrong_subject_or_object_penalty`
- Penalizes captions implying:
  - fox / wrong animal
  - brown bear / panda
  - book / reading / truck / car when the story is about a honey jar search

### 5. New config and run script
- `config_v41_text_story_grounding.yaml`
- `run_v41.py`
- `examples/white_bear_text_only_v41.json`

## Recommended run command
```bash
python run_v41.py   --config configs/config_v41_text_story_grounding.yaml   --input examples/white_bear_text_only_v41.json   --out outputs/DCEE_v41_white_bear_text_only
```
