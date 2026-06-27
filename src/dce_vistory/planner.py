# src/dce_vistory/planner.py
# Drop-in planner patch.
# Replace your old woodcutter/folk-tale-biased prompt builder with this input-grounded planner.
#
# Required external function:
#   llm_json(messages: list[dict]) -> dict
# It should call your OpenAI/OpenRouter/local LLM and return parsed JSON.

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List

from .guards import (
    ForbiddenEntityError,
    StoryConstraints,
    allowed_entities_text,
    assert_no_forbidden_terms,
    constraints_from_input,
    find_forbidden_terms,
)


def _system_prompt() -> str:
    return """You are a visual storytelling planner.

CRITICAL RULES:
1. Use ONLY the user-provided protagonist, input text, reference image description, allowed characters, allowed objects, and allowed locations.
2. Do NOT introduce any new character unless allow_new_characters=true.
3. Do NOT use woodcutter, lumberjack, axe, hunter, farmer, villager, random human, or any human antagonist unless explicitly listed in allowed_characters.
4. Do NOT copy examples from fairy tales or previous runs.
5. If conflict is needed, create it from allowed objects, nature, weather, loss, distance, misunderstanding, scarcity, or the protagonist's internal emotion.
6. The story must be visual: each sentence should describe a clear visible action, visible evidence object, location, and emotion.
7. Return STRICT JSON only. No markdown.
"""


def _planner_user_prompt(input_data: Dict[str, Any], constraints: StoryConstraints) -> str:
    num_frames = int(input_data.get("num_frames", 6))
    target_emotion = input_data.get("target_emotion", input_data.get("ending_emotion", "sad"))
    simple_content = input_data.get("content", input_data.get("description", ""))
    image_caption = input_data.get("image_caption", input_data.get("reference_image_caption", ""))

    return f"""
Create a DCEE visual storytelling plan from the following input.

[INPUT]
protagonist: {constraints.protagonist}
target_ending_emotion: {target_emotion}
num_frames: {num_frames}
simple_content: {simple_content}
reference_image_caption: {image_caption}
{allowed_entities_text(constraints)}

[OUTPUT JSON SCHEMA]
{{
  "protagonist": "...",
  "target_ending_emotion": "...",
  "story_abstract": "...",
  "desire": "...",
  "conflict": "...",
  "turning_point": "...",
  "ending_state": "...",
  "event_spine": [
    {{
      "event_id": 1,
      "event": "...",
      "causal_role": "desire_start | conflict_start | escalation | turning_point | ending",
      "visible_action": "...",
      "evidence_objects": ["..."],
      "location": "...",
      "emotion": "...",
      "emotion_cause": "..."
    }}
  ],
  "story_sentences": [
    "Frame 1 sentence.",
    "Frame 2 sentence.",
    "Frame 3 sentence.",
    "Frame 4 sentence.",
    "Frame 5 sentence.",
    "Frame 6 sentence."
  ],
  "character_bible": {{
    "name": "{constraints.protagonist}",
    "role": "protagonist",
    "identity_lock": "...",
    "appearance": "...",
    "outfit": "...",
    "color_palette": ["..."],
    "must_remain_consistent": true
  }},
  "world_bible": {{
    "main_location": "...",
    "weather": "...",
    "style": "cinematic storybook illustration",
    "allowed_locations": {constraints.allowed_locations}
  }}
}}

Requirements:
- story_sentences length must be exactly {num_frames}.
- Every sentence must focus on {constraints.protagonist}.
- Every event must be drawable as one frame.
- Use no forbidden entity.
- If only panda is allowed, the conflict should come from bamboo scarcity, river current, rain, wind, darkness, distance, or internal sadness—not from a woodcutter/human.
"""


def _repair_prompt(plan: Dict[str, Any], constraints: StoryConstraints, hits: List[str]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": _system_prompt()},
        {
            "role": "user",
            "content": f"""
The previous JSON contains forbidden terms: {hits}

Rewrite the JSON so that:
- The story uses only allowed characters: {constraints.allowed_characters}
- The protagonist remains: {constraints.protagonist}
- Forbidden terms are completely removed: {constraints.forbidden_terms}
- The causal structure remains DCEE.
- Keep exactly the same JSON schema.
- Return STRICT JSON only.

Previous JSON:
{json.dumps(plan, ensure_ascii=False)}
""",
        },
    ]


def generate_dcee_plan(
    input_data: Dict[str, Any],
    llm_json: Callable[[List[Dict[str, str]]], Dict[str, Any]],
    max_retries: int = 3,
) -> Dict[str, Any]:
    constraints = constraints_from_input(input_data)

    messages = [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": _planner_user_prompt(input_data, constraints)},
    ]

    last_plan: Dict[str, Any] | None = None
    last_hits: List[str] = []

    for attempt in range(max_retries):
        if attempt == 0:
            plan = llm_json(messages)
        else:
            plan = llm_json(_repair_prompt(last_plan or {}, constraints, last_hits))

        last_plan = plan
        last_hits = find_forbidden_terms(plan, constraints)
        if not last_hits:
            _validate_plan_shape(plan, input_data)
            return plan

    raise ForbiddenEntityError(
        f"Planner kept generating forbidden entities after {max_retries} retries: {last_hits}"
    )


def _validate_plan_shape(plan: Dict[str, Any], input_data: Dict[str, Any]) -> None:
    num_frames = int(input_data.get("num_frames", 6))
    sentences = plan.get("story_sentences", [])
    if not isinstance(sentences, list) or len(sentences) != num_frames:
        raise ValueError(f"story_sentences must have exactly {num_frames} items, got {len(sentences)}")

    required = [
        "protagonist",
        "target_ending_emotion",
        "story_abstract",
        "desire",
        "conflict",
        "turning_point",
        "ending_state",
        "event_spine",
        "story_sentences",
        "character_bible",
        "world_bible",
    ]
    missing = [k for k in required if k not in plan]
    if missing:
        raise ValueError(f"Missing planner keys: {missing}")


def build_storyboard_from_plan(plan: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create one frame spec for each story sentence.
    This function intentionally does NOT invent a new antagonist or new object.
    """
    constraints = constraints_from_input(input_data)
    assert_no_forbidden_terms(plan, constraints, where="DCEE plan")

    event_spine = plan.get("event_spine", [])
    sentences = plan["story_sentences"]
    frames = []

    for idx, sentence in enumerate(sentences, start=1):
        event = event_spine[min(idx - 1, len(event_spine) - 1)] if event_spine else {}
        frame = {
            "frame_id": idx,
            "story_sentence": sentence,
            "event": event.get("event", sentence),
            "visible_action": event.get("visible_action", sentence),
            "evidence_objects": event.get("evidence_objects", []),
            "emotion": event.get("emotion", ""),
            "emotion_cause": event.get("emotion_cause", ""),
            "location": event.get("location", plan.get("world_bible", {}).get("main_location", "")),
            "character_bible": plan.get("character_bible", {}),
            "world_bible": plan.get("world_bible", {}),
            "allowed_characters": constraints.allowed_characters,
            "allowed_objects": constraints.allowed_objects,
            "forbidden_terms": constraints.forbidden_terms,
        }
        assert_no_forbidden_terms(frame, constraints, where=f"storyboard frame {idx}")
        frames.append(frame)

    storyboard = {"frames": frames, "plan": plan}
    assert_no_forbidden_terms(storyboard, constraints, where="storyboard")
    return storyboard
