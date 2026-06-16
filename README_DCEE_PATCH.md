# DCEE-BVC code patch

This patch changes the planning flow from DCE to DCEE:
Desire -> Conflict -> Event Chain -> Ending Emotion.

Files included:
- src/dce_vistory/prompts.py
- src/dce_vistory/planner.py
- src/dce_vistory/evaluator.py
- src/dce_vistory/pipeline_crossattn_butterfly.py

Notes:
- Existing class/file names are kept for compatibility.
- `DCEPlan.event_spine` is treated as the DCEE event chain.
- The pipeline also saves `dcee_plan.json` while preserving `dce_plan.json`.
- New evaluation fields include `event_grounding` and `event_emotion_causal_consistency`.
