# DCEE-CausalVerse v15.1 Story Bible NameError Fix

## Problem

The v15 run failed at:

```text
NameError: name 'story_bible' is not defined
```

inside:

```text
src/dce_vistory/frame_director.py
```

## Cause

`frame_director.py` used `story_bible`, `bible_world`, and `bible_emotion`, but the local
variables were not initialized inside `build_frame_visual_spec`.

## Fix

This patch modifies only:

```text
src/dce_vistory/frame_director.py
```

It safely loads the Story Bible from:

```python
story_bible = getattr(seed, "_story_bible", {}) or {}
```

and initializes:

```python
bible_world = world_for_frame(story_bible, frame)
bible_emotion = emotion_cue_from_bible(story_bible, emotion)
```

## Apply

Unzip at project root and overwrite:

```text
src\dce_vistory\frame_director.py
```

## Run

```powershell
$env:PYTHONPATH="src"
python run_crossattn_butterfly_pipeline.py --config configs/crossattn_butterfly_dcee_causalverse.yaml --input examples/sad_1_p2.json --out outputs/DCEE_v15_1_sad_1
```
