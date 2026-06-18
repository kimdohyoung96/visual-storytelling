# DCEE-CausalVerse final method/code patch

This patch upgrades the previous DCEE-BVC pipeline into DCEE-CausalVerse.

Key additions:
1. DCEE-Tree planning: samples multiple Desire->Conflict routes, generates alternative event chains, and selects the best event chain for the target ending emotion.
2. Entity/event canonicalization: rewrites vague story events into drawable event/evidence descriptions.
3. Causal Sink Memory: retrieves previous memory based on event, conflict, evidence, emotion-cause, and clue relevance.
4. Anchor Bank: stores character, object/evidence, and world anchors.
5. Evidence branch: extends Character/World/Emotion/Event control into Character/World/Emotion/Event/Evidence cross-attention tokens.
6. Event-evidence-aware reranking and retry.

Apply by overwriting the files under `src/dce_vistory/` and optionally using:
`configs/crossattn_butterfly_dcee_causalverse.yaml`.

Example:
```powershell
$env:PYTHONPATH="src"
python run_crossattn_butterfly_pipeline.py --config configs/crossattn_butterfly_dcee_causalverse.yaml --input examples/sad_1.json --out outputs/DCEE_CausalVerse_sad_1
```
