# DCEE-CausalVerse V28: V20-based Caption-Grounded Patch

## 목적
V21~V27에서 추가했던 강한 subject-consistency/feature-sharing 계열 아이디어를 제거하고,
이미지 생성 결과가 더 안정적이었던 V20 계열 구조로 되돌린 뒤, 각 frame caption이 실제 image prompt의 중심이 되도록 개선한 패치입니다.

## 핵심 변경

### 1. V20 기반으로 복귀
- protagonist-only story generation 유지
- story/caption 기반 frame generation 유지
- 불필요한 consistency override / pairwise override 제거

### 2. 후보 생성 복원
V20.1에서 잘못 적용했던 `1 frame = 1 candidate image` 제한을 제거했습니다.
이제 다시 다음 구조로 동작합니다.

```text
1 frame -> multiple candidates -> caption/story/event/evidence 기반 best selection
```

설정값은 기존 config를 따릅니다.

```python
num_candidates = int(img_cfg.get("num_candidates_per_frame", 2))
num_ending_candidates = int(img_cfg.get("num_ending_candidates", 5))
```

### 3. Caption-Grounded image generation
`planner.py`와 `prompts.py`에서 각 frame마다 다음 필드를 생성하도록 했습니다.

- `sentence`: 한국어 frame caption
- `image_caption_en`: 같은 내용을 담은 영어 image-generation caption
- `action / action_en`
- `visible_cause / visible_cause_en`
- `required_objects / required_objects_en`
- `background_elements / background_elements_en`

SDXL prompt는 이제 `caption`을 1순위 계약으로 사용합니다.

### 4. Code-level object filtering
프롬프트만으로 다른 객체를 막지 않고, `planner.py`에서 `required_objects`와 `background_elements`를 다시 정리합니다.
다음은 제거됩니다.

- human / person / child / extra animal
- duplicate protagonist
- ungrounded agent
- story에 없는 object/prop

### 5. Candidate selector 개선
`evaluator.py`에서 caption과 관련 없는 후보를 더 강하게 감점합니다.

- `story_alignment`
- `event_alignment`
- `event_grounding`
- `evidence_visibility`
- `required_object_coverage`

반대로 다음은 감점됩니다.

- unrelated human/person/child
- duplicate protagonist
- generic portrait
- caption mismatch

## 수정 파일
- `src/dce_vistory/prompts.py`
- `src/dce_vistory/planner.py`
- `src/dce_vistory/pipeline_crossattn_butterfly.py`
- `src/dce_vistory/frame_director.py`
- `src/dce_vistory/sdxl_cross_attention_generator.py`
- `src/dce_vistory/evaluator.py`
- `src/dce_vistory/butterfly_adapter.py`
- `src/dce_vistory/causal_memory.py`
- `src/dce_vistory/story_graph.py`
- `src/dce_vistory/story_bible.py`

## 적용 방법
```powershell
cd C:\Users\nkm11\visual-storytelling
Expand-Archive -Path .\DCEE_CausalVerse_V28_v20_caption_grounded_patch.zip -DestinationPath . -Force
```

## 실행 예시
```powershell
$env:PYTHONPATH="src"
python run_crossattn_butterfly_pipeline.py --config configs/crossattn_butterfly_dcee_causalverse.yaml --input examples/sad_1_b3.json --out outputs/DCEE_v28_W_sad_1
```

## 확인 파일
- `generation_policy_V28.json`
- `storyboard.json`
- `visual_control_packets.json`
- `candidate_manifest.json`
- `selected_images.json`
- `contact_sheet.png`

`candidate_manifest.json` 안에 `v28_selection_reason`이 있으면 V28 selector가 적용된 것입니다.
