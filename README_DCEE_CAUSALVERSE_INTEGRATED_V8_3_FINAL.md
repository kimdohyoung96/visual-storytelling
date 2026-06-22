# DCEE-CausalVerse Integrated v8.3 Final

This is the cumulative integrated package. It includes all recent patches:

1. v6 final submission fixes
   - conflict level vs emotion arc separation
   - protagonist identity lock
   - target-emotion-aware planning
   - event/evidence-oriented image selection

2. v7 story alignment
   - full_story generation
   - story sentence to frame alignment
   - `story_sentence`, `story_alignment_reason`, `dcee_stage`
   - `full_story.json` output
   - story sentence included in Butterfly visual prompt

3. v7 symbolic object type normalization
   - list/string/dict visual_symbols converted to dict for WorldLatent

4. v8 emotion compatibility
   - `regret` accepted as sad-ending subtype
   - compatible emotion families normalized for downstream checks

5. v8.3 frame evidence repair
   - missing key_objects/evidence_objects repaired from aligned story sentence,
     linked DCEE event, must_show, and seed objects

## Important

If you previously applied patch files one-by-one, it is easy to miss one overwrite.
This integrated package avoids that problem.

## Run

```powershell
$env:PYTHONPATH="src"
python run_crossattn_butterfly_pipeline.py --config configs/crossattn_butterfly_dcee_causalverse.yaml --input examples/sad_1.json --out outputs/DCEE_v8_3_sad_1
```

For happy:

```powershell
$env:PYTHONPATH="src"
python run_crossattn_butterfly_pipeline.py --config configs/crossattn_butterfly_dcee_causalverse.yaml --input examples/happy_1.json --out outputs/DCEE_v8_3_happy_1
```
