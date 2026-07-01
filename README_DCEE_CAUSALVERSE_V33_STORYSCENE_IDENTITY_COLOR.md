# DCEE-CausalVerse V33 Story-Scene Identity Color Patch

This patch updates the V32 storyfaith visual bridge to improve three recurring failures observed in the generated contact sheets:

1. **Caption-to-image mismatch**: generation now prioritizes the exact frame caption, location, required objects, and visible action before longer identity text.
2. **Weak protagonist continuity**: the protagonist identity anchor is compressed into a short, stable species/color descriptor (e.g., `one adult white bear, white fur`) to reduce prompt truncation and keep the same subject across frames.
3. **Gray / empty backgrounds**: prompt construction and candidate ranking now explicitly reward readable, full-color environments and penalize gray or empty scenes.

## Main code changes
- `frame_director.py`
  - Compact protagonist identity anchor.
  - Stronger continuity rule requiring a colorful environment that matches the caption.
  - Stronger negative prompt terms for gray / empty backgrounds.
- `sdxl_cross_attention_generator.py`
  - New V33 caption-priority prompt compiler.
  - Prompt order changed to: caption -> protagonist -> action -> place -> required objects -> emotion -> continuity.
  - Background-preserving lighting hints for sad/dark scenes.
  - Candidate variants remain story-faithful while emphasizing background, action, or emotion readability.
  - Retry mode now actually affects generation through control metadata.
- `evaluator.py`
  - Added `gray_background_penalty` and `generic_caption_penalty`.
  - Increased selection weight on story alignment, event grounding, evidence visibility, and scene alignment.
  - Penalizes washed-out / grayscale scenes and generic captions such as `a black and white drawing of a bear`.
- `pipeline_crossattn_butterfly.py`
  - Retry mode now passes strong generation control into `control_metadata` so the generator can react to low-alignment outputs.

## Expected effect
Compared with V32, V33 is designed to generate each frame as:
- **one single coherent scene**,
- **one consistent protagonist**,
- **a readable and colorful background**, and
- **a closer visual realization of the current story caption**.
