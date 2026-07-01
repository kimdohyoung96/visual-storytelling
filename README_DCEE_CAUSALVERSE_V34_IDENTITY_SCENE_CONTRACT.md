# DCEE-CausalVerse V34 Identity-Lock + Scene-Contract Patch

This patch updates the V33 story-scene identity/color patch. V33 improved the overall visual quality compared with V32, but it still had two recurring failures:

1. **Protagonist identity drift**: the protagonist stayed roughly the same species/color, but face, hands/paws, feet, body proportions, and other details changed between frames.
2. **Weak story-background grounding**: backgrounds often looked cropped, plain, monotonous, or unrelated to the current story sentence.

V34 fixes these by separating the protagonist into two parts:

- **Fixed Identity Lock**: species, age, body size, silhouette, face shape, fur/color pattern, hands/paws, feet, ears, and signature items must remain stable across every frame.
- **DCEE-Controlled Appearance Delta**: only expression, gaze, pose, lighting, emotional tension, and small story-caused surface changes such as dirt/wetness/scratches may change according to Desire, Conflict, Event, and Ending stages.

It also adds a frame-level **Scene Contract**:

- foreground: protagonist action,
- midground: visible evidence/object interaction,
- background: story-specific location, weather, atmosphere, and environmental objects.

## Main code changes

### `frame_director.py`
- Adds `identity_lock`, `dcee_appearance_delta`, `scene_contract`, `scene_elements`, and `foreground_elements` to `FrameVisualSpec`.
- Stops compressing identity down to only generic species/color. The compact anchor remains, but a stricter identity lock is now passed to the generator.
- Infers story-specific background/location from the current caption when frame/story metadata is weak.
- Gives story-row/frame location priority over the global fixed setting, reducing repeated generic backgrounds.
- Separates stable world background from required objects so static background terms do not dominate every frame.
- Strengthens negatives against changed face/body/fur pattern, malformed/missing hands/paws/feet, and cropped protagonist.

### `sdxl_cross_attention_generator.py`
- Replaces V33 caption-priority prompt strategy with `V34_identity_lock_scene_contract_dcee_delta`.
- Adds identity-lock, wide-scene, action, background, emotion, and caption candidate modes.
- Uses stronger IP-Adapter scaling by default and adjusts it by candidate mode: slightly higher for identity candidates, slightly lower for background/wide-scene candidates.
- Prompt order is now: caption → protagonist anchor → identity lock → DCEE appearance delta → action → scene contract → evidence/objects/location/weather/emotion.
- Adds explicit full-body / head-to-toe composition requirements to reduce cropped faces, paws, hands, and feet.

### `evaluator.py`
- Adds lightweight `reference_subject_similarity` using center-crop RGB histogram similarity against the input reference image.
- Adds `crop_penalty_proxy` to penalize likely cropped/portrait-like outputs.
- Increases ranking weight for identity consistency and scene alignment while still keeping story/event/evidence alignment high.
- Penalizes gray backgrounds, generic captions, duplicate protagonists, extra subjects, and cropped outputs more strongly.

### `pipeline_crossattn_butterfly.py`
- Updates generation policy to V34.
- Adds metadata flags requiring identity lock, scene contract, DCEE appearance delta, uncropped full body, and background-story alignment.
- Retry mode now explicitly says identity must remain unchanged while DCEE may only change expression, pose, dirt/wetness, lighting, and emotional tension.

## Expected effect vs V33

V34 should better generate:

- one protagonist with the same face/body/paws/feet/fur pattern across frames,
- DCEE-driven visual changes without redesigning the protagonist,
- less cropped protagonist composition,
- more story-specific backgrounds,
- clearer foreground action + midground evidence + background location,
- better candidate selection when multiple images are generated per frame.

## Files to replace

Copy these files into your existing project under `src/dce_vistory/`:

- `frame_director.py`
- `sdxl_cross_attention_generator.py`
- `evaluator.py`
- `pipeline_crossattn_butterfly.py`

Recommended config change:

```json
{
  "image_generator": {
    "num_candidates_per_frame": 4,
    "num_ending_candidates": 6,
    "ip_adapter_scale": 0.62,
    "guidance_scale": 7.5,
    "num_inference_steps": 36,
    "width": 1024,
    "height": 1024,
    "use_ip_adapter": true
  },
  "pipeline": {
    "emotion_retry": true,
    "story_alignment_threshold": 0.82,
    "event_grounding_threshold": 0.78,
    "evidence_visibility_threshold": 0.78,
    "emotion_visibility_threshold": 0.74,
    "colorfulness_threshold": 0.35
  }
}
```

If the protagonist is still unstable after V34, the next step should be stronger identity conditioning such as InstantID/PhotoMaker/LoRA or a generated canonical reference sheet. V34 remains training-free and only modifies prompt/control/ranking logic.
