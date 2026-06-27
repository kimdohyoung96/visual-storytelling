# DCEE-CausalVerse V19 Protagonist-Only Incremental Patch

## 목적

V19는 다음 문제를 해결하기 위한 최종 구조입니다.

1. 생성된 story에 주인공이 아닌 다른 객체/캐릭터가 등장하면 이미지에서 누락되거나 주인공과 비슷하게 생성되는 문제
2. 입력 이미지가 강하게 작동할 때 보조 객체/보조 캐릭터가 주인공처럼 복제되는 문제
3. 판다만 입력했는데 woodcutter, axe, fairy, wild animal friends 같은 템플릿 story 요소가 침투하는 문제
4. 생성된 story 문장과 frame image가 약하게 연결되는 문제

## V19 핵심 정책

V19는 `protagonist_only_incremental` 모드입니다.

- 주인공만 active character로 사용합니다.
- secondary character, friends, animal friends, helpers, humans, woodcutters, fairies, villagers, crowds를 생성하지 않습니다.
- 스토리에 나오는 object는 prop/background로 제한합니다.
- object가 이미지 생성에서 혼동을 일으키면 story 단계에서 제거합니다.
- conflict는 다른 캐릭터가 아니라 환경, 잃어버린 물건, 날씨, 거리, 배고픔, 실수, 내적 감정에서 발생합니다.

## 생성 순서

V19는 전체 story를 먼저 만든 뒤 image를 생성하지 않습니다.

```text
input image/text
→ grounded seed
→ DCEE plan
→ sentence 1 생성
→ frame 1 이미지 생성
→ sentence 2 생성, frame 1 memory 반영
→ frame 2 이미지 생성
→ ...
→ sentence 6 생성
→ frame 6 ending image 생성
```

## 수정 파일

- `src/dce_vistory/prompts.py`
- `src/dce_vistory/planner.py`
- `src/dce_vistory/pipeline_crossattn_butterfly.py`

## 적용 방법

프로젝트 루트에서 압축을 풀고 덮어쓰기:

```powershell
cd C:\Users\nkm11\visual-storytelling
Expand-Archive -Path .\DCEE_CausalVerse_V19_protagonist_only_incremental_patch.zip -DestinationPath . -Force
```

## 실행

```powershell
$env:PYTHONPATH="src"
python run_crossattn_butterfly_pipeline.py --config configs/crossattn_butterfly_dcee_causalverse.yaml --input examples/sad_1_p2.json --out outputs/DCEE_V19_sad_1
```

## 확인 파일

```text
outputs\DCEE_V19_sad_1\generation_policy_V19.json
outputs\DCEE_V19_sad_1\seed.json
outputs\DCEE_V19_sad_1\full_story.json
outputs\DCEE_V19_sad_1\storyboard.json
outputs\DCEE_V19_sad_1\selected_images.json
```

`generation_policy_V19.json`에서 아래가 보여야 합니다.

```json
{
  "version": "V19",
  "mode": "protagonist_only_incremental",
  "protagonist_only": true,
  "no_secondary_characters": true
}
```

`seed.json`에는 현재 입력에 없는 템플릿 entity가 금지 목록으로 저장됩니다.

```json
"forbidden_ungrounded_entities": [
  "woodcutter",
  "lumberjack",
  "axe",
  "fairy",
  "golden axe",
  "silver axe",
  "friend",
  "friends",
  "wild animal friends"
]
```

## 기대 효과

- 판다 입력에서는 판다 중심 이야기만 생성
- wild animal friends / woodcutter / axe / fairy 제거
- 이미지 생성이 어려운 보조 캐릭터 제거
- 주인공의 행동, 감정, 배경, prop 중심으로 frame 생성
- sentence i → frame i alignment 강화
