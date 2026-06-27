# DCEE-CausalVerse v16 Multi-History Story-Faithful Patch

This patch upgrades the image-generation side of the pipeline so that the generated story is treated as the **source prompt contract** for visual storytelling.

## Main goals

1. **Render all objects mentioned in the generated story sentence.**
2. **Improve story-to-image faithfulness at each frame.**
3. **Improve frame-to-frame continuity** for protagonist, world, and recurring objects.
4. **Address ViSTA limitations 1 and 2**:
   - multi-history selection instead of over-relying on a single salient frame
   - reduce auto-regressive drift by using memory-image references plus global story constraints

## Modified / added files

- `src/dce_vistory/story_graph.py` **(new)**
- `src/dce_vistory/frame_director.py`
- `src/dce_vistory/causal_memory.py`
- `src/dce_vistory/butterfly_adapter.py`
- `src/dce_vistory/sdxl_cross_attention_generator.py`
- `src/dce_vistory/pipeline_crossattn_butterfly.py`
- `src/dce_vistory/evaluator.py`

## What changed

### 1) Story Graph (new)
`story_graph.py` builds a lightweight object/entity graph from the generated full story and storyboard.

For each frame it tracks:
- current entities / objects
- carry-over entities from the previous frame
- recurring entities appearing across multiple frames
- world cues (location / weather / atmosphere)

This lets each frame prompt explicitly include not just the protagonist but also the **other objects mentioned in the story sentence**.

### 2) Frame Visual Spec upgraded
`frame_director.py` now merges:
- story sentence objects
- storyboard `key_objects`
- storyboard `evidence_objects`
- `emotion_evidence`
- story graph entities
- recurring / carry-over entities
- story-bible stable background cues

This directly fixes the issue where **secondary objects in the story were ignored**.

### 3) Multi-history memory selection
`causal_memory.py` no longer behaves like a near-single-history selector.
It now returns:
- `selected_memories`
- `multi_history_summary`
- `entity_memory`
- `world_memory`
- `reference_memory_images`

This is the direct improvement over ViSTA limitation #1.

### 4) Anti-drift image conditioning
`sdxl_cross_attention_generator.py` now builds a **reference collage** from:
- source input image
- up to 2 salient memory-frame images

and feeds that into IP-Adapter.

This reduces the chance that later frames drift into:
- wrong protagonist age / gender / appearance
- wrong world background
- wrong object set

This is the direct improvement over ViSTA limitation #2.

### 5) Stronger sentence-locked prompting
Prompts are now short but strict, and each candidate uses one of:
- `sentence_locked`
- `object_locked`
- `continuity_locked`
- `background_locked`
- `emotion_locked`

This prevents the generator from collapsing into a generic portrait.

## Apply
Unzip this patch at the project root and overwrite the files.

## Recommended run
```powershell
$env:PYTHONPATH="src"
python run_crossattn_butterfly_pipeline.py --config configs/crossattn_butterfly_dcee_causalverse.yaml --input examples/sad_1_p2.json --out outputs/DCEE_v16_sad_1
```

## New outputs
This patch additionally writes:
- `story_graph.json`

## Expected improvement
Compared with previous versions, this patch should improve:
- object coverage from story sentence to image
- continuity of protagonist identity
- continuity of background / setting
- continuity of recurring props and evidence objects
- overall story-faithful visual storytelling quality
