# DCEE-CausalVerse Final Submission Code v6

This final package addresses the user's three findings.

## 1. Conflict Level vs Emotion Arc

They are intentionally separated:
- Emotion Arc = `emotion` + `emotion_intensity`
- Conflict Level = narrative tension from DCEE event progression

They do not have to be equal. The final story now prints both.

## 2. Same protagonist identity

Every frame receives an `identity_lock`:
- name
- age_group
- gender
- face
- hair
- body
- outfit
- signature_items

The Butterfly controller adds this to the character control prompt. The protagonist's expression, pose, event, emotion, and background may change, but age/gender/face/outfit/body identity should remain stable.

## 3. Visual storytelling image selection

The evaluator prompt now prioritizes:
- event grounding
- evidence visibility
- visible cause of emotion
- target emotion

It should not over-reward fixed shot type or fixed pose. Stable outfit is treated as identity consistency only, not as the main objective.

## 4. Any target ending emotion

The planner repair prompts no longer hard-code sadness. They use target-specific guidance:
- happy/joy/relief: honesty rewarded, old axe returned, warm light, grateful smile
- sad/regret/grief: loss, empty hands, rain, kneeling body
- other emotions: use `target_ending_emotion`

Run:
```powershell
$env:PYTHONPATH="src"
python run_crossattn_butterfly_pipeline.py --config configs/crossattn_butterfly_dcee_causalverse.yaml --input examples/sad_1.json --out outputs/DCEE_v6_sad_1
python run_crossattn_butterfly_pipeline.py --config configs/crossattn_butterfly_dcee_causalverse.yaml --input examples/happy_1.json --out outputs/DCEE_v6_happy_1
```
