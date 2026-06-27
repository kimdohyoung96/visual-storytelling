# DCEE-CausalVerse v15 Story Bible Lock Patch

## Problem

The v14 result improved the river/background somewhat, but three failures remained:

1. World/background consistency was weak.
   - forest, riverbank, bamboo grove, river water were not locked globally.
2. Character identity was weak.
   - the panda changed between adult/cub/juvenile forms.
3. Emotion expression was weak.
   - generated frames often showed a panda but not the required facial/body emotion.

## Core fix

v15 adds a global **Story Bible** and injects it into every frame prompt.

The Story Bible stores:

- fixed protagonist identity
- adult/age lock
- negative identity constraints
- fixed world/background anchors
- emotion-to-face/body rendering rules
- role constraints

## New file

- `src/dce_vistory/story_bible.py`

## Modified files

- `src/dce_vistory/frame_director.py`
- `src/dce_vistory/pipeline_crossattn_butterfly.py`
- `src/dce_vistory/visual_sentence_planner.py`
- `src/dce_vistory/prompts.py`
- `src/dce_vistory/sdxl_cross_attention_generator.py`

## What changes in generation

Each frame prompt now includes:

```text
STRICT CHARACTER LOCK:
same adult giant panda protagonist in every frame; large adult body proportions;
black-and-white fur; round expressive face; wearing yellow work shirt...

STRICT WORLD LOCK:
deep green bamboo forest beside a visible riverbank; flowing river water;
bamboo grove; river rocks; green leaves...

STRICT EMOTION LOCK:
face must show tearful eyes/downturned mouth/etc.; body must show slumped shoulders/empty paws/etc.
```

For panda stories, v15 also adds strong negative prompts:

```text
baby panda, panda cub, small juvenile panda, childlike panda,
different panda, multiple pandas, human panda hybrid, missing yellow shirt,
wrong background, missing river, missing bamboo forest, emotionless face
```

## Output check

The pipeline writes:

- `story_bible.json`
- `full_story.json`
- `storyboard.json`
- `visual_control_packets.json`
- `selected_images.json`

Open `story_bible.json` and confirm:

```json
{
  "age_lock": "adult giant panda, not a cub, not a baby, not a juvenile",
  "world": {
    "fixed_setting": "deep green bamboo forest beside a visible riverbank"
  }
}
```

Open `selected_images.json` and confirm each candidate note has `frame_visual_spec` with locked identity/world/emotion prompts.

## Run

```powershell
$env:PYTHONPATH="src"
python run_crossattn_butterfly_pipeline.py --config configs/crossattn_butterfly_dcee_causalverse.yaml --input examples/sad_1_p2.json --out outputs/DCEE_v15_story_bible_sad_1
```

## Recommended config

Copy values from `RECOMMENDED_CONFIG_STORY_BIBLE_LOCK.yaml` into your main config.
