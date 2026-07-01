# DCEE-CausalVerse V38 Patch

## Why V38 was created
V37 became too aggressive and unstable:
- the protagonist reference bank could behave like a mini collage and encourage duplicate characters,
- action verbs such as `enters` / `searches` leaked into the critical noun list,
- the prompt became too contract-heavy and did not consistently preserve a strong scene/background,
- ranking penalties were too weak for duplicate characters and plain-background failures.

## What V38 changes
1. **Single-reference identity lock**
   - uses one stable protagonist reference instead of a composite reference collage.
   - previous frame remains a **text continuity hint**, not an identity image collage.

2. **Scene-first prompting**
   - prompt prioritizes: exact story sentence -> main action -> key props -> scene/background.
   - strong instructions against sticker / icon / plain-background outputs.

3. **Cleaner critical visual nouns**
   - filters out action verbs from the visual noun set.
   - keeps object/location nouns such as `lost honey jar`, `dense forest`, `bush`, `tangled roots`, `steep slope`, `serene lake`.

4. **Stronger evaluator penalties**
   - harsher penalties for duplicate protagonists, extra subjects, sticker-like outputs, and missing scene grounding.

5. **Safer execution defaults**
   - uses 896x896 and fewer candidates to reduce OOM risk while preserving visual quality.
   - `run_v38.py` accepts both `--output` and `--out`.

## Main files changed
- `src/dce_vistory/frame_director.py`
- `src/dce_vistory/sdxl_cross_attention_generator.py`
- `src/dce_vistory/evaluator.py`
- `src/dce_vistory/pipeline_crossattn_butterfly.py`
- `config_v38.yaml`
- `run_v38.py`
