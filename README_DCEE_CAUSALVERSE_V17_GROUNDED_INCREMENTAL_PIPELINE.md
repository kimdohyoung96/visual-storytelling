# DCEE-CausalVerse v17 Grounded Incremental Pipeline Patch

This patch fixes two major issues:

1. **Woodcutter leakage / template-story contamination**
   - Earlier code could leak prior example-story entities such as *woodcutter*, *axe*, or *fairy* into a panda story.
   - This patch explicitly computes **forbidden ungrounded entities** from the current input and blocks them in planner generation.
   - If the input is only a panda + text, the story is forced to stay grounded to that panda story.

2. **Whole-story-first generation mismatch**
   - Earlier code generated the whole story first and generated frames afterward.
   - This patch changes the pipeline to an **incremental interleaved mode**:
     - generate story sentence 1 -> generate frame 1
     - generate story sentence 2 using prior story and frame summary -> generate frame 2
     - generate story sentence 3 -> generate frame 3
     - ...
   - This is closer to the paper goal of sentence-level visual storytelling alignment.

## Main modified files

- `src/dce_vistory/prompts.py`
- `src/dce_vistory/planner.py`
- `src/dce_vistory/pipeline_crossattn_butterfly.py`

## Key design changes

### A. Grounded-only planner
`planner.py` now:
- extracts grounded entities from the input text/image summary
- builds a forbidden list for ungrounded template-story terms
- rejects / repairs outputs containing ungrounded terms
- sanitizes remaining leakage if needed

### B. Image-friendly sentence generation
Each story sentence is generated with strict constraints:
- one clear visible action
- one concrete place
- visible required objects
- readable emotion
- visible cause of the emotion
- continuity with the previous frame

### C. Interleaved pipeline
`pipeline_crossattn_butterfly.py` now uses:
- `generate_story_step(...)`
- `story_step_to_frame(...)`
inside the frame loop.

That means the pipeline now behaves like:
1. understand input image/text
2. make grounded seed and DCEE plan
3. for each frame index:
   - make the next story sentence
   - convert that sentence into a storyboard frame spec
   - generate the image immediately
   - store it in memory for the next step

## Expected benefit
Compared with the previous whole-story-first approach, this patch should improve:
- sentence-to-image alignment
- protagonist continuity
- background continuity
- grounding to the input panda story
- prevention of woodcutter-story contamination

## Apply
Unzip and overwrite these files in your project root.
