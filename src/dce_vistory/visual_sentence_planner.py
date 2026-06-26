from __future__ import annotations

from typing import Any, Dict, List
import re


ABSTRACT_WORDS = {
    "realizes", "realize", "understands", "understand", "knows", "learns", "learn",
    "honesty", "integrity", "choice", "moral", "importance", "heart", "fate",
    "truth", "greed", "perseverance", "desire", "loss", "weight", "meaning",
    "symbolizes", "symbolize", "bittersweet", "profound", "forever changed",
}

COMPLEX_CONNECTORS = [
    " as ", " while ", " when ", " because ", " although ", " but ", " however ",
    " realizing ", " understanding ", " feeling ", " that ", " which ",
]


def clean(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "").replace("\n", " ")).strip()


def words(x: str) -> List[str]:
    return re.findall(r"[A-Za-z가-힣0-9']+", clean(x))


def imageability_issues(sentence: str) -> List[str]:
    s = clean(sentence)
    low = s.lower()
    issues = []
    if len(words(s)) > 18:
        issues.append("too_many_words")
    if s.count(",") >= 2 or ";" in s or ":" in s:
        issues.append("too_many_clauses")
    if any(c in low for c in COMPLEX_CONNECTORS):
        issues.append("complex_temporal_or_abstract_clause")
    if any(w in low for w in ABSTRACT_WORDS):
        issues.append("abstract_or_internal_word")
    # Too many active agents usually confuses SDXL unless explicitly spatialized.
    agents = sum(1 for a in ["panda", "woodcutter", "fairy", "man", "woman", "bear", "child"] if a in low)
    if agents >= 3:
        issues.append("too_many_agents")
    return issues


def is_image_friendly_sentence(sentence: str) -> bool:
    return not imageability_issues(sentence)


def normalize_visual_rows(data: Any, num_frames: int) -> List[Dict[str, Any]]:
    if isinstance(data, dict):
        rows = data.get("sentences") or data.get("visual_sentences") or data.get("frames") or []
    elif isinstance(data, list):
        rows = data
    else:
        rows = []
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows[:num_frames]):
        if not isinstance(row, dict):
            row = {"sentence": str(row)}
        sentence = clean(row.get("sentence") or row.get("image_sentence") or row.get("caption") or "")
        subject = clean(row.get("subject") or row.get("protagonist") or "")
        action = clean(row.get("action") or row.get("visible_action") or row.get("event") or "")
        obj = clean(row.get("object") or row.get("target_object") or row.get("main_object") or "")
        location = clean(row.get("location") or row.get("background") or row.get("scene_location") or "")
        emotion = clean(row.get("emotion") or "")
        face = clean(row.get("facial_cue") or row.get("face") or "")
        body = clean(row.get("body_cue") or row.get("pose") or "")
        required = row.get("required_objects") or row.get("must_show") or []
        if isinstance(required, str):
            required = [required]
        if not isinstance(required, list):
            required = [str(required)]
        required = [clean(x) for x in required if clean(x)]
        out.append({
            "frame_id": int(row.get("frame_id", i + 1) or i + 1),
            "sentence": sentence,
            "image_sentence": sentence,
            "subject": subject,
            "action": action,
            "object": obj,
            "location": location,
            "emotion": emotion,
            "facial_cue": face,
            "body_cue": body,
            "required_objects": required[:5],
            "forbidden_objects": row.get("forbidden_objects", []),
            "imageability_issues": imageability_issues(sentence),
            "dcee_stage": clean(row.get("dcee_stage") or row.get("stage") or ""),
            "alignment_reason": clean(row.get("alignment_reason") or "This sentence is one drawable moment."),
        })
    return out


def visual_story_rewrite_prompt(seed: Dict[str, Any], dce_plan: Dict[str, Any], emotion_arc: Dict[str, Any], full_story: Dict[str, Any], num_frames: int) -> str:
    return f"""
Rewrite the generated story into an IMAGE-FRIENDLY 6-frame visual story for text-to-image generation.

INPUT SEED:
{seed}

DCEE PLAN:
{dce_plan}

EMOTION ARC:
{emotion_arc}

ORIGINAL FULL STORY:
{full_story}

Return JSON only:
{{
  "story_title": "short title",
  "sentences": [
    {{
      "frame_id": 1,
      "sentence": "one simple drawable sentence",
      "subject": "one main subject",
      "action": "one visible action",
      "object": "one main visible object",
      "location": "one visible background",
      "emotion": "visible emotion",
      "facial_cue": "visible face cue",
      "body_cue": "visible body cue",
      "required_objects": ["object1", "object2"],
      "forbidden_objects": ["object that must not appear"]
    }}
  ]
}}

HARD RULES:
- Return exactly {num_frames} sentences.
- Sentence i will become image frame i.
- Each sentence must be EASY for SDXL to draw.
- Each sentence must be 8 to 16 words.
- Each sentence must have exactly ONE visible action.
- Each sentence must have exactly ONE clear location/background.
- Each sentence must have exactly ONE visible emotion cue.
- Do not write abstract/internal ideas: honesty, integrity, realizes, understands, importance, heart, fate, moral, choice.
- Convert abstract meaning into visible objects/actions.
  Bad: "the panda realizes the importance of honesty"
  Good: "the panda kneels beside broken bamboo stumps in the rain"
- Avoid multiple simultaneous actions in one sentence.
- Avoid "while", "as", "because", "but", "although", "realizing", "understanding".
- Keep the protagonist identity consistent.
- If another character appears, state clear spatial roles:
  "woodcutter on the left raises axe; panda on the right watches".
- For emotion, use visible face/body cues, not abstract words.
- The final frame must be an ending image with visible cause of the target emotion.
""".strip()
