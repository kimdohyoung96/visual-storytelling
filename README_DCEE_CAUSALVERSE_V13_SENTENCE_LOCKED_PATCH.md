# DCEE-CausalVerse v13 Sentence-Locked Visual Storytelling Patch

## Goal

The paper requirement is strict:

story sentence 1 -> frame 1 image  
story sentence 2 -> frame 2 image  
story sentence 3 -> frame 3 image  
story sentence 4 -> frame 4 image  
story sentence 5 -> frame 5 image  
story sentence 6 -> frame 6 ending image

This patch changes image generation so the generated full story is the source of truth.

## Why previous versions failed

Previous versions mixed too many signals:
- DCEE event
- emotion arc
- evidence
- character
- world
- reference image
- untrained ButterflyAdapter tokens

The main technical issue is that `ButterflyAdapterStack` is not trained. Injecting untrained adapter tokens into SDXL prompt embeddings can weaken the actual sentence-to-image mapping. Therefore v13 disables it by default unless you provide a trained checkpoint.

## Main changes

### 1. `src/dce_vistory/frame_director.py`

Builds a `FrameVisualSpec` for each frame from:
- exact story sentence
- frame id / total frames
- protagonist identity
- primary visible action
- visible event
- visible cause of emotion
- required objects
- location / weather / atmosphere
- facial expression / body pose
- negative constraints

### 2. Sentence lock in `pipeline_crossattn_butterfly.py`

After full story and storyboard are generated:

```python
frame_i.story_sentence = full_story["sentences"][i]["sentence"]
frame_i.caption = full_story["sentences"][i]["sentence"]
frame_i.frame_id = i + 1
frame_i.sentence_locked = True
```

### 3. Direct SDXL prompt generation

`sdxl_cross_attention_generator.py` sends a direct prompt string to SDXL:

```text
FRAME 3 OF 6.
Create one full-color cinematic storybook illustration that visualizes exactly this sentence:
"..."
Primary visible action: ...
Must show these objects: ...
Location/background: ...
Emotion: ...
```

### 4. Candidate variants

For each frame:
- sentence_locked
- action
- objects
- background
- emotion

### 5. Input image reference

IP-Adapter is kept at a lower scale (`0.38`) so the input image preserves subject identity without forcing all frames to copy the same pose/background.

## Files changed

- `src/dce_vistory/frame_director.py`
- `src/dce_vistory/butterfly_adapter.py`
- `src/dce_vistory/sdxl_cross_attention_generator.py`
- `src/dce_vistory/pipeline_crossattn_butterfly.py`
- `RECOMMENDED_CONFIG_SENTENCE_LOCKED.yaml`

## Apply

Unzip at project root and overwrite existing files.

## Run

```powershell
$env:PYTHONPATH="src"
python run_crossattn_butterfly_pipeline.py --config configs/crossattn_butterfly_dcee_causalverse.yaml --input examples/sad_1_p2.json --out outputs/DCEE_v13_sentence_locked_sad_1
```

## Check

Open `selected_images.json` and confirm:

```json
"sentence_locked_generation": true,
"untrained_butterfly_adapter_disabled": true,
"frame_visual_spec": {
  "story_sentence": "..."
}
```
