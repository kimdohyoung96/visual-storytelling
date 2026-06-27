# DCEE-CausalVerse V20.1 Single-Image Cleanup Story-Locked Patch

## Fix
- Forces `num_candidates_per_frame = 1`
- Forces `num_ending_candidates = 1`
- Disables retry generation
- Cleans old `frame_*_cand_*.png` files from the output folder at run start
- Narrows visual inventory to reduce objects/people not present in the story

## Run
```powershell
$env:PYTHONPATH="src"
python run_crossattn_butterfly_pipeline.py --config configs/crossattn_butterfly_dcee_causalverse.yaml --input examples/sad_1_b1.json --out outputs/DCEE_v20_1_sad_1
```

Use a new output folder, or rely on V20.1 cleanup if reusing a folder.
