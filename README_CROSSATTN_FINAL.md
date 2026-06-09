# DCE-ViStory Final Cross-Attention Adapter Version

This package adds the final research-code layer for Character / World / Emotion / Event adapters.

Apply this ZIP on top of your current `dce_vistory_v2` project after the world-aware/butterfly patch.

Run:

```powershell
$env:PYTHONPATH="src"
python run_crossattn_butterfly_pipeline.py --config configs/crossattn_butterfly.yaml --input examples/crossattn_woodcutter.json --out outputs/crossattn_woodcutter_run
```

Core implementation:
- `src/dce_vistory/adapters_pytorch.py`
- `src/dce_vistory/sdxl_cross_attention_generator.py`
- `src/dce_vistory/pipeline_crossattn_butterfly.py`

It connects to SDXL cross-attention by appending adapter tokens to `prompt_embeds`.
