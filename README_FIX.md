# DCE-ViStory schema-compatible emotion/color patch

이 패치는 현재 `schema.py`에 `world_context`, `shot_type`, `color_palette` 같은 필드가 없어도 실행되도록 수정한 버전입니다.

교체 파일:
- src/dce_vistory/prompts.py
- src/dce_vistory/planner.py
- src/dce_vistory/sdxl_cross_attention_generator.py
- src/dce_vistory/evaluator.py
- src/dce_vistory/pipeline_crossattn_butterfly.py

핵심 수정:
- StorySeed(world_context=...) 에러 방지
- schema.py에 없는 필드는 객체 생성 후 setattr로 보존
- full-color 강제
- emotion visibility / emotion cause visibility 강화
- 감정 또는 컬러가 약하면 retry generation 수행
