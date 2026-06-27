# DCEE-CausalVerse V21 Patch

Patch target: `DCEE_CausalVerse_V19_required_objects_fix_patch.zip`

## Main changes in V21
1. **Input image is a hard identity anchor**
   - Story seed and frame prompts now explicitly preserve protagonist species, fur/color, body shape, age impression, and distinctive appearance.
   - This directly addresses cases like **white bear drifting into brown bear**.

2. **Story-to-image grounding is stronger**
   - Each frame uses both the original story sentence and a simplified `image_sentence` that is easier for the image generator to render.
   - Frame prompts explicitly lock: event, visible cause, required objects, background, weather, and emotion.

3. **Previous selected frame continuity is injected**
   - The current frame prompt now explicitly references the previous selected frame for identity and scene continuity.

4. **Single-scene enforcement per frame image**
   - Candidate generation is still allowed (so you can choose the best candidate), but every candidate is instructed to be **one single coherent scene**, not split panels or multiple moments.

## Modified files
- `src/dce_vistory/prompts.py`
- `src/dce_vistory/planner.py`
- `src/dce_vistory/pipeline_crossattn_butterfly.py`
