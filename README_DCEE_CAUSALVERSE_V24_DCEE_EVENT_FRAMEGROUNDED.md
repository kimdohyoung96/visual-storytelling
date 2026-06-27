# DCEE-CausalVerse V24 DCEE Event-Frame Grounded Patch

## 목적

V23 결과에서 다음 문제가 발생했습니다.

1. Storyboard는 흰 곰 1마리라고 되어 있는데, frame 이미지에는 작은 두 번째 곰/사람/보조 객체가 생성됨
2. Frame 1~6의 이미지가 story event보다 캐릭터/배경 위주로 생성됨
3. 후보 이미지 중 story에 더 맞는 후보가 있는데도 다른 후보가 선택됨
4. Korean story sentence가 SDXL prompt에 그대로 들어가면서 action/evidence가 약하게 반영됨

V24는 이 문제를 해결하기 위해 DCEE event contract와 English SDXL visual prompt를 추가합니다.

## 논문 반영 방식

`Training-Free Consistent Text-to-Image Generation` / ConsiStory의 full SDSA, feature injection, DIFT correspondence는 U-Net attention hook과 batch-level feature sharing이 필요합니다. 현재 코드 구조에서 억지로 넣으면 더 불안정합니다.

따라서 V24는 도움이 되는 부분만 반영합니다.

- subject consistency는 유지
- background/layout mistake는 공유하지 않음
- input image는 subject identity anchor로만 사용
- frame별 prompt alignment를 우선
- 후보 간 비교에서 story/event/evidence를 우선

## 수정 파일

- `src/dce_vistory/prompts.py`
- `src/dce_vistory/planner.py`
- `src/dce_vistory/pipeline_crossattn_butterfly.py`
- `src/dce_vistory/frame_director.py`
- `src/dce_vistory/sdxl_cross_attention_generator.py`
- `src/dce_vistory/evaluator.py`

## 핵심 변경

### 1. DCEE event contract

각 frame은 아래 요소를 반드시 가져야 합니다.

```text
protagonist + action + cause/evidence + required objects + background + emotion
```

이제 "깨달음을 얻었다", "아무것도 없는 곳을 응시한다" 같은 추상 event는 image prompt로 바로 보내지 않고,
visible evidence/action 중심으로 변환합니다.

### 2. English SDXL prompt

LLM story는 한국어로 유지할 수 있지만, image generation에는 아래 English fields를 사용합니다.

- `image_sentence_en`
- `action_en`
- `action_pose_en`
- `visible_cause_en`
- `required_objects_en`
- `background_elements_en`
- `camera_composition_en`

### 3. Single protagonist hard constraint

Frame prompt와 negative prompt에 아래를 강하게 넣습니다.

```text
EXACTLY ONE white bear
no second white bear
no duplicate protagonist
no extra character
no split panel
no comic panel
```

### 4. Candidate selector 개선

V24는 후보 이미지를 개별 평가한 뒤, 마지막에 VLM pairwise selector로 후보들을 한 번에 비교합니다.

선택 기준:

1. duplicate protagonist / extra character / split panel이면 탈락
2. story event/action이 가장 잘 보이는 후보 우선
3. required objects/evidence가 잘 보이는 후보 우선
4. emotion cause가 보이는 후보 우선
5. image beauty는 마지막

### 5. Object inventory 정리

문장 조각이나 추상어는 object에서 제거합니다.

제거 예:
- `quietly on the`
- `water with a`
- `staring`
- `heavy`
- `the loss`
- `emotion`

유지 예:
- `white bear`
- `honey jar`
- `riverbank`
- `river`
- `rock`
- `snowy forest`
- `spilled honey`

## 실행

```powershell
cd C:\Users\nkm11\visual-storytelling
Expand-Archive -Path .\DCEE_CausalVerse_V24_dcee_event_framegrounded_patch.zip -DestinationPath . -Force

$env:PYTHONPATH="src"
python run_crossattn_butterfly_pipeline.py --config configs/crossattn_butterfly_dcee_causalverse.yaml --input examples/sad_1_b3.json --out outputs/DCEE_v24W_sad_1
```

## 확인 파일

- `outputs\DCEE_v24W_sad_1\generation_policy_V24.json`
- `outputs\DCEE_v24W_sad_1\candidate_manifest.json`
- `outputs\DCEE_v24W_sad_1\selected_images.json`
- `outputs\DCEE_v24W_sad_1\storyboard.json`
- `outputs\DCEE_v24W_sad_1\contact_sheet.png`

`candidate_manifest.json`에서 아래가 있으면 V24 selector가 적용된 것입니다.

```json
"v24_selection_reason": {
  "story_first_selection": true,
  "event_action_evidence_first": true,
  "image_quality_weight_minimal": true
}
```

VLM pairwise selector가 정상 작동하면 아래도 들어갑니다.

```json
"v24_pairwise_selection": {
  "best_candidate_id": 2,
  "rejected_candidate_ids": [0, 1]
}
```
