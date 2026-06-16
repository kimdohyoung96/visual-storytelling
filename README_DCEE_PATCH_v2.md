# DCEE-BVC modified code patch v2

This patch updates the current DCE-BVC code to follow the final DCEE flow:

Desire -> Conflict -> Event Chain -> Ending Emotion

Modified files:
- src/dce_vistory/prompts.py
- src/dce_vistory/planner.py
- src/dce_vistory/butterfly_adapter.py
- src/dce_vistory/evaluator.py
- src/dce_vistory/pipeline_crossattn_butterfly.py
- src/dce_vistory/sdxl_cross_attention_generator.py

Notes:
- The public class names DCEPlanner and DCEPlan are kept for backward compatibility.
- DCEE fields are attached dynamically when schema.py does not contain them.
- The pipeline writes both dcee_plan.json and dce_plan.json.
- Event grounding, emotion-cause visibility, event-emotion causal consistency, and colorfulness are used for reranking/retry.
- The SDXL generator includes the previous fp16/fp32 VAE dtype fix and compact prompt logic to reduce CLIP 77-token truncation.

Apply by unzipping over your project root:
C:\Users\kdhms\Desktop\DET\dce_vistory_v2
