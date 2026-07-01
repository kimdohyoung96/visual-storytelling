# DCEE-CausalVerse V40 text-only story patch

## Goal
Run the pipeline **without `image_path`** so story generation and image generation are grounded only on:
- `text_prompt`
- `protagonist`
- `target_ending_emotion`
- `genre`
- `style`
- `num_frames`
- `language`
- `signature_items`

## What changed
1. `pipeline_crossattn_butterfly.py`
   - normalizes inputs when `image_path` is missing
   - detects whether any real reference image exists
   - when no reference exists, switches to **text-only story generation mode**
   - skips dependency on input image analysis
   - writes `generation_policy_V40_text_only_story.json`

2. `sdxl_cross_attention_generator.py`
   - if `text_only_mode=True`, it ignores reference-image paths entirely
   - forces identity backend to `text`
   - prompt wording changes from “use the reference image ...” to “generate from text only ...”

3. `config_v40_text_only.yaml`
   - disables `use_ip_adapter`
   - disables subject-scene fusion by default
   - sets identity backend priority to `text` only

4. `run_v40.py`
   - accepts JSONs that omit `image_path`
   - fills missing image-related keys with empty defaults

5. Example input JSON included
   - `examples/white_bear_text_only.json`

## Recommended run command
```bash
python run_v40.py   --config configs/config_v40_text_only.yaml   --input examples/white_bear_text_only.json   --out outputs/DCEE_v40_white_bear_text_only
```
