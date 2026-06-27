# DCEE-CausalVerse V20 Single-Image Story-Locked Patch

## 목적
1. 각 frame에 후보 이미지가 여러 장 생기는 문제를 제거
2. story에 없는 객체/인물/배경 hallucination을 줄여 sentence-frame 정합성을 높임

## 핵심
- frame당 정확히 1장만 생성
- ending frame도 1장만 생성
- retry 비활성화로 추가 후보 파일 생성 방지
- image prompt를 story-locked 방식으로 강화
- must_show inventory를 최소한의 object/background만 남기도록 축소

## 수정 파일
- src/dce_vistory/prompts.py
- src/dce_vistory/planner.py
- src/dce_vistory/pipeline_crossattn_butterfly.py

## 실행 예시
```powershell
$env:PYTHONPATH="src"
python run_crossattn_butterfly_pipeline.py --config configs/crossattn_butterfly_dcee_causalverse.yaml --input examples/sad_1_b1.json --out outputs/DCEE_v20_sad_1
```
