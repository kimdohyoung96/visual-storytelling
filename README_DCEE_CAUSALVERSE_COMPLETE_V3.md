# DCEE-CausalVerse Complete Stable Code v3

This is the cleaned complete code patch aligned with the final paper direction.

## Paper direction implemented

- DCEE: Desire -> Conflict -> Event Chain -> Ending Emotion
- DCEE-Tree planning output support via `dcee_candidate_plans.json`
- Causal Sink Memory
- DCEE Anchor Bank
- Butterfly Visual Controller with Character / World / Emotion / Event / Evidence branches
- Factorized cross-attention controls for SDXL
- Event/evidence/emotion-aware reranking and retry
- Robust outputs for paper inspection

## Important fix

`contact_sheet.png` is now generated directly inside `pipeline_crossattn_butterfly.py`
before evaluation. It no longer depends on `utils.make_contact_sheet`.
The pipeline also always writes:

- `selected_images.json`
- `candidate_manifest.json`
- `eval_questions.json`
- `evaluation.json`
- `final_story.md`
- `output_manifest.json`

If contact sheet generation fails, the reason is recorded in `evaluation.json`.

## Files included

- `src/dce_vistory/prompts.py`
- `src/dce_vistory/planner.py`
- `src/dce_vistory/causal_memory.py`
- `src/dce_vistory/anchor_bank.py`
- `src/dce_vistory/adapters_pytorch.py`
- `src/dce_vistory/butterfly_adapter.py`
- `src/dce_vistory/sdxl_cross_attention_generator.py`
- `src/dce_vistory/evaluator.py`
- `src/dce_vistory/pipeline_crossattn_butterfly.py`
- `configs/crossattn_butterfly_dcee_causalverse.yaml`
- `scripts/make_contact_sheet_from_output.py`

## Apply

Unzip this at your project root and overwrite existing files.

Project root example:

```text
C:\Users\kdhms\Desktop\DET\dce_vistory_v2
```

## Run

```powershell
Remove-Item -Recurse -Force outputs\DCEE_CausalVerse_sad_1
$env:PYTHONPATH="src"
python run_crossattn_butterfly_pipeline.py --config configs/crossattn_butterfly_dcee_causalverse.yaml --input examples/sad_1.json --out outputs/DCEE_CausalVerse_sad_1
```

## If you already generated images but contact_sheet.png is missing

```powershell
python scripts\make_contact_sheet_from_output.py --out outputs\DCEE_CausalVerse_sad_1
```

## Expected output files

```text
outputs/DCEE_CausalVerse_sad_1/
  dcee_candidate_plans.json
  dcee_plan.json
  storyboard.json
  selected_images.json
  candidate_manifest.json
  visual_control_packets.json
  memory_log.json
  ending_candidates.json
  eval_questions.json
  evaluation.json
  contact_sheet.png
  final_story.md
  output_manifest.json
```
