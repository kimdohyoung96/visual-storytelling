# DCEE-CausalVerse v6 storyboard exact-count patch

Fixes:
`Storyboard length mismatch after strict API repair. got=1, expected=6`

The previous repair prompt showed a one-frame example, so the model copied one frame.
This patch generates an explicit required skeleton for frame_id 1..N and retries API repair up to 3 times.

Changed file:
- src/dce_vistory/planner.py
