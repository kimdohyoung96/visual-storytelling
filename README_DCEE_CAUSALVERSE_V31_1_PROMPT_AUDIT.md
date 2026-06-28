# DCEE-CausalVerse V31.1 Prompt-Audited Story-Aligned Generation Patch

## Purpose

This patch answers the prompt-truncation concern before running V31.

Stable Diffusion XL uses short CLIP text encoder context windows. If a prompt is too long,
important story/action/evidence details can be truncated before image generation. V31.1
therefore changes the SDXL generator to:

1. use a compact front-loaded prompt,
2. keep caption/action/evidence/objects at the beginning of the prompt,
3. compact the negative prompt with extra-subject constraints first,
4. log token counts and truncation flags for every candidate.

## Check after running

Open:

- `candidate_manifest.json`
- `selected_images.json`

Each candidate now contains:

```json
"token_report": {
  "prompt_tokenizer_1": {"tokens": ..., "max_length": ..., "will_truncate": ...},
  "prompt_tokenizer_2": {"tokens": ..., "max_length": ..., "will_truncate": ...}
}
```

If `will_truncate` is `true`, the prompt is still too long and should be shortened further.

## Modified files

- `src/dce_vistory/sdxl_cross_attention_generator.py`
- `src/dce_vistory/pipeline_crossattn_butterfly.py`

## Apply

```powershell
cd C:\Users\nkm11\visual-storytelling
Expand-Archive -Path .\DCEE_CausalVerse_V31_1_prompt_audit_patch.zip -DestinationPath . -Force
```

## Run

```powershell
$env:PYTHONPATH="src"
python run_crossattn_butterfly_pipeline.py --config configs/crossattn_butterfly_dcee_causalverse.yaml --input examples/sad_1_b3.json --out outputs/DCEE_v31_1_W_sad_1
```
