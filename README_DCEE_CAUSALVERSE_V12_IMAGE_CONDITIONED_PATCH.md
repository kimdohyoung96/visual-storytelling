# DCEE-CausalVerse v12 Image-Conditioned Story-Exact Patch

## Why this patch is necessary

The previous pipeline used `image_path` mostly for image understanding / captioning.
So the uploaded image could influence the text plan, but it did not directly condition SDXL image generation.

This patch makes the input image a visual subject reference during frame generation.

## Main changes

1. `pipeline_crossattn_butterfly.py`
   - saves `sample["image_path"]` into `seed.source_image_path`
   - passes IP-Adapter settings into the SDXL generator

2. `butterfly_adapter.py`
   - writes the input image path into `VisualControlPacket.reference_images`
   - keeps it in `control_metadata["source_reference_image_path"]`

3. `sdxl_cross_attention_generator.py`
   - optionally loads SDXL IP-Adapter
   - passes the input image as `ip_adapter_image`
   - still generates each frame with story sentence / event / evidence / emotion prompts

4. `story_visual_alignment.py`
   - builds compact frame-level visual contracts
   - creates prompt variants: event-first, evidence-first, emotion-first, continuity-first

## Important

If IP-Adapter cannot load in your environment, the run does not crash. It falls back to text-only generation and logs:
- `ip_adapter_loaded: false`
- `ip_adapter_error: ...`

## Apply

Unzip at your project root and overwrite existing files.

## Run

```powershell
$env:PYTHONPATH="src"
python run_crossattn_butterfly_pipeline.py --config configs/crossattn_butterfly_dcee_causalverse.yaml --input examples/sad_1_p2.json --out outputs/DCEE_v12_panda_sad_1
```
