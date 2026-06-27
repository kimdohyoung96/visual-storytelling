# src/dce_vistory/sequential_pipeline.py
# Final autoregressive visual storytelling pipeline.
#
# Core behavior:
#   1) Input JSON + optional reference image -> clean DCEE plan/story sentences.
#   2) Generate frame 1 from sentence 1 + input/reference image.
#   3) Generate frame 2 from sentence 2 + previous generated frame 1 + initial anchor.
#   4) Generate frame k from sentence k + previous generated frame k-1 + compact story/character bible.
#
# Required external objects:
#   llm_json(messages) -> dict
#   image_generator.generate_frame(packet: dict) -> str
#
# image_generator.generate_frame should return saved image path.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .guards import (
    assert_no_forbidden_terms,
    constraints_from_input,
    hard_negative_prompt,
)
from .planner import build_storyboard_from_plan, generate_dcee_plan


def build_frame_packet(
    frame: Dict[str, Any],
    input_data: Dict[str, Any],
    previous_frame: Optional[Dict[str, Any]],
    initial_image_path: Optional[str],
    output_dir: Path,
) -> Dict[str, Any]:
    constraints = constraints_from_input(input_data)

    frame_id = int(frame["frame_id"])
    current_sentence = frame["story_sentence"]

    prev_sentence = ""
    prev_image_path = None
    if previous_frame:
        prev_sentence = previous_frame.get("story_sentence", "")
        prev_image_path = previous_frame.get("image_path")

    # Keep current sentence dominant. Previous frame is only for identity/style continuity.
    positive_prompt = f"""
cinematic storybook illustration, high quality, coherent visual storytelling frame

CURRENT FRAME {frame_id}:
{current_sentence}

VISIBLE ACTION:
{frame.get("visible_action", current_sentence)}

MAIN PROTAGONIST IDENTITY LOCK:
{json.dumps(frame.get("character_bible", {}), ensure_ascii=False)}

WORLD / STYLE LOCK:
{json.dumps(frame.get("world_bible", {}), ensure_ascii=False)}

EMOTION:
{frame.get("emotion", "")}
Cause: {frame.get("emotion_cause", "")}

LOCATION:
{frame.get("location", "")}

CONTINUITY RULE:
Use the previous generated frame only as reference for the same protagonist identity, color palette, and visual style.
Do not copy the previous event if it conflicts with the current sentence.
Previous sentence: {prev_sentence}

STRICT ENTITY RULE:
Allowed characters only: {constraints.allowed_characters}.
Allowed objects: {constraints.allowed_objects}.
Do not add any new character.
Do not add forbidden entities.
""".strip()

    negative_prompt = hard_negative_prompt(constraints)

    packet = {
        "frame_id": frame_id,
        "story_sentence": current_sentence,
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "reference_image_paths": [p for p in [initial_image_path, prev_image_path] if p],
        "previous_frame_path": prev_image_path,
        "initial_image_path": initial_image_path,
        "output_path": str(output_dir / "frames" / f"frame_{frame_id:03d}.png"),
        "seed": int(input_data.get("seed", 42)) + frame_id,
        "width": int(input_data.get("width", 1024)),
        "height": int(input_data.get("height", 1024)),
        "guidance_scale": float(input_data.get("guidance_scale", 6.0)),
        "num_inference_steps": int(input_data.get("num_inference_steps", 30)),
        # img2img/IP-Adapter users can use this:
        # Low/medium value preserves identity while allowing event changes.
        "reference_strength": float(input_data.get("reference_strength", 0.45 if frame_id > 1 else 0.35)),
        "allowed_characters": constraints.allowed_characters,
        "forbidden_terms": constraints.forbidden_terms,
    }

    assert_no_forbidden_terms(packet["positive_prompt"], constraints, where=f"frame {frame_id} positive prompt")
    return packet


def run_sequential_visual_storytelling(
    input_json_path: str | Path,
    output_dir: str | Path,
    llm_json: Callable[[List[Dict[str, str]]], Dict[str, Any]],
    image_generator: Any,
) -> Dict[str, Any]:
    input_json_path = Path(input_json_path)
    output_dir = Path(output_dir)
    (output_dir / "frames").mkdir(parents=True, exist_ok=True)

    input_data = json.loads(input_json_path.read_text(encoding="utf-8"))
    constraints = constraints_from_input(input_data)

    plan = generate_dcee_plan(input_data=input_data, llm_json=llm_json)
    assert_no_forbidden_terms(plan, constraints, where="final DCEE plan")
    (output_dir / "dcee_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    storyboard = build_storyboard_from_plan(plan, input_data)
    assert_no_forbidden_terms(storyboard, constraints, where="final storyboard")
    (output_dir / "storyboard.json").write_text(json.dumps(storyboard, ensure_ascii=False, indent=2), encoding="utf-8")

    initial_image_path = input_data.get("image_path") or input_data.get("reference_image_path")
    history: List[Dict[str, Any]] = []

    for frame in storyboard["frames"]:
        previous_frame = history[-1] if history else None

        packet = build_frame_packet(
            frame=frame,
            input_data=input_data,
            previous_frame=previous_frame,
            initial_image_path=initial_image_path,
            output_dir=output_dir,
        )

        (output_dir / "frames" / f"frame_{packet['frame_id']:03d}_packet.json").write_text(
            json.dumps(packet, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        image_path = image_generator.generate_frame(packet)
        frame_result = {
            **frame,
            "image_path": image_path,
            "packet_path": str(output_dir / "frames" / f"frame_{packet['frame_id']:03d}_packet.json"),
        }
        history.append(frame_result)

        # Save after each frame so an interrupted run can be inspected.
        (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    final_story = {
        "input": input_data,
        "plan": plan,
        "storyboard": storyboard,
        "frames": history,
    }
    (output_dir / "final_story.json").write_text(json.dumps(final_story, ensure_ascii=False, indent=2), encoding="utf-8")
    return final_story
