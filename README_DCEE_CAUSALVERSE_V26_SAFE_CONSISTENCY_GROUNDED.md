# DCEE-CausalVerse V26 Safe Consistency + Grounded Incremental Patch

## 핵심 방향
V26는 `Training-Free Consistent Text-to-Image Generation`의 방법론을 **그대로 억지로 연결하지 않고**, 현재 코드에 실제로 도움이 되는 부분만 남긴 버전입니다.

즉,
- **유지한 것**: 입력 이미지 기반 identity anchor, 프레임 간 continuity 의식
- **제거/완화한 것**: pairwise override, 과도한 stage-lock, 강한 스타일화/억지 consistency coupling

V19_1 / V20의 장점을 살려서 다시 **story-first**, **image-grounded**, **incremental generation**으로 돌아갔습니다.

---

## V26에서 바뀐 점

### 1) Story 생성 단계 개선
- story sentence를 생성할 때 다음 정보를 함께 사용합니다.
  - input image summary
  - 이전 story sentence
  - 이전 selected image summary
  - DCEE plan
- 각 step은 **single-scene**, **one main action**, **protagonist-only** 규칙을 따릅니다.
- `allowed_visual_terms`를 코드에서 구성하여, LLM이 내놓은 object/background를 후처리로 다시 필터링합니다.
- 즉, 다른 객체 제거를 **프롬프트만으로 처리하지 않고**, 코드 레벨에서 grounded inventory 기반으로 정리합니다.

### 2) Image 생성 단계 개선
- 현재 frame prompt에는 다음이 반영됩니다.
  - current story sentence
  - previous story sentence
  - previous selected image summary
  - visible cause/evidence
  - allowed grounded inventory only
- continuity는 텍스트 기반으로만 넣고, duplicate subject를 유발하는 강한 연결은 줄였습니다.
- 입력 이미지 identity anchor는 유지합니다.

### 3) Candidate selection 개선
- V25의 pairwise selector는 비활성화했습니다.
- 실제로 더 나은 후보를 pairwise 단계가 덮어쓰는 문제가 있을 수 있어서,
  V26은 기본 ranking에서 끝냅니다.
- ranking은 story/event/evidence/identity 가중치를 높였습니다.

### 4) 이전 결과 재사용 방지
- 같은 output 폴더를 재실행할 때 기존 frame candidate / contact_sheet / selected json을 지웁니다.
- 이전 결과가 남아 새 결과처럼 보이는 문제를 줄입니다.

### 5) 이전 선택 이미지 요약 사용
- 프레임이 하나 선택되면, 그 선택된 이미지를 `image_understanding.analyze()`로 다시 요약합니다.
- 이 요약이 다음 frame story step 및 image prompt continuity에 사용됩니다.
- 즉, **직전 story 문장 + 직전 실제 선택 이미지**를 보고 다음 이미지를 만들도록 바뀌었습니다.

---

## 수정 파일
- `src/dce_vistory/prompts.py`
- `src/dce_vistory/planner.py`
- `src/dce_vistory/pipeline_crossattn_butterfly.py`
- `src/dce_vistory/frame_director.py`
- `src/dce_vistory/evaluator.py`

---

## 적용 방법
```powershell
cd C:\Users\nkm11\visual-storytelling
Expand-Archive -Path .\DCEE_CausalVerse_V26_safe_consistency_grounded_patch.zip -DestinationPath . -Force
```

## 실행 예시
```powershell
$env:PYTHONPATH="src"
python run_crossattn_butterfly_pipeline.py --config configs/crossattn_butterfly_dcee_causalverse.yaml --input examples/sad_1_b3.json --out outputs/DCEE_v26_W_sad_1
```

---

## 확인 포인트
생성 후 아래 파일을 확인하세요.
- `generation_policy_V26.json`
- `candidate_manifest.json`
- `selected_images.json`
- `storyboard.json`
- `contact_sheet.png`
- `evaluation.json`

특히 `visual_control_packets.json` 안에 각 frame마다
- `previous_image_summary_for_next_frame`
가 기록되면, 직전 선택 이미지 summary가 다음 frame에 실제로 반영된 것입니다.
