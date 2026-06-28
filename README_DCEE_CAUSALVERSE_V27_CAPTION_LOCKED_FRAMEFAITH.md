# DCEE-CausalVerse V27 Caption-Locked Frame-Faith Patch

## 왜 V27이 필요한가
V26 결과를 보면 최종 `storyboard.json` / `full_story.md` 안의 프레임 caption은 비교적 그럴듯하게 생성되지만,
실제 SDXL 생성 이미지는 그 caption을 정확히 따르지 못했습니다.

즉, 핵심 문제는 다음입니다.

1. **caption/story sentence와 image 생성 prompt의 결속력이 약함**
2. **candidate selector가 caption-faithful 후보보다 다른 후보를 선택할 수 있음**
3. **required object / location / event evidence가 image 생성과 선택에서 충분히 강제되지 않음**
4. **generic portrait / unrelated human / unrelated object가 끼어들어도 선택될 수 있음**

V27은 이 문제를 해결하기 위해 **각 frame의 caption을 가장 우선적인 계약(contract)** 으로 보도록 수정한 버전입니다.

---

## V27 핵심 변경점

### 1) Caption-Locked Prompting
각 프레임 image prompt를 만들 때 다음 우선순위를 명확히 고정했습니다.

1. exact current frame caption
2. current event/action
3. visible cause/evidence
4. grounded required objects
5. location / weather / atmosphere
6. emotion

즉, **예쁜 그림**보다 **caption에 맞는 그림**이 우선되도록 바꿨습니다.

### 2) FrameFaith visual inventory 강화
`frame_director.py`에서 caption 안의 핵심 키워드(예: 꿀/강/숲/하늘/돌/음식)를 추출해
`required_objects`에 보강합니다.

예:
- `꿀`, `honey` -> `honey jar`
- `강`, `river` -> `riverbank`, `water`
- `숲`, `forest` -> `forest`, `trees`

즉, caption에 있는 핵심 대상과 배경이 prompt에 더 직접적으로 반영됩니다.

### 3) Candidate selection을 caption-faithful 방향으로 재조정
`evaluator.py`에서 다음을 강화했습니다.

- `caption_faithfulness_local`
- `required_object_coverage_local`
- local caption 안에 `person/man/woman/human/child`가 있으면 강한 penalty
- story / event / evidence 가중치 상향
- image quality / colorfulness 가중치 하향

즉, **예쁘지만 틀린 후보**보다 **정확히 맞는 후보**를 선택하게 했습니다.

### 4) Pairwise override 비활성화
기존 pairwise VLM override는 때때로 더 적합한 후보 대신
더 그럴듯해 보이는 다른 후보를 선택할 수 있었습니다.

V27은 이를 비활성화했습니다.

### 5) Retry 조건에 story_alignment 추가
기존에는 emotion / event / evidence 위주로 retry 했는데,
이제 `story_alignment`가 낮아도 retry 하도록 수정했습니다.

즉, caption과 안 맞는 이미지가 나오면 더 적극적으로 다시 생성합니다.

---

## 수정 파일
- `src/dce_vistory/planner.py`
- `src/dce_vistory/frame_director.py`
- `src/dce_vistory/sdxl_cross_attention_generator.py`
- `src/dce_vistory/evaluator.py`
- `src/dce_vistory/pipeline_crossattn_butterfly.py`

---

## 적용 방법
```powershell
cd C:\Users\nkm11\visual-storytelling
Expand-Archive -Path .\DCEE_CausalVerse_V27_caption_locked_framefaith_patch.zip -DestinationPath . -Force
```

## 실행 예시
```powershell
$env:PYTHONPATH="src"
python run_crossattn_butterfly_pipeline.py --config configs/crossattn_butterfly_dcee_causalverse.yaml --input examples/sad_1_b3.json --out outputs/DCEE_v27_W_sad_1
```

---

## 실행 후 확인 포인트
다음 파일을 꼭 확인하세요.

- `generation_policy_V27.json`
- `candidate_manifest.json`
- `selected_images.json`
- `storyboard.json`
- `evaluation.json`
- `contact_sheet.png`

특히 `selected_images.json` / `candidate_manifest.json`에서
- `story_alignment`
- `event_alignment`
- `evidence_visibility`
- `v27_selection_reason`
을 보면, 왜 그 이미지가 선택됐는지 더 잘 확인할 수 있습니다.
