from __future__ import annotations

import argparse
import json
from pathlib import Path


GENERIC = [
    "resolve the central problem",
    "discovers the problem",
    "conflict becomes visible",
    "decisive event changes the outcome",
    "object or place that starts the story",
]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    required = [
        "abstract.txt",
        "dcee_plan.json",
        "storyboard.json",
        "selected_images.json",
        "visual_control_packets.json",
        "memory_log.json",
        "evaluation.json",
        "final_story.md",
        "contact_sheet.png",
    ]
    missing = [x for x in required if not (out / x).exists()]
    if missing:
        raise SystemExit(f"Missing required outputs: {missing}")

    abstract = (out / "abstract.txt").read_text(encoding="utf-8").strip()
    final_md = (out / "final_story.md").read_text(encoding="utf-8").strip()
    if not abstract:
        raise SystemExit("abstract.txt is empty")
    if not final_md:
        raise SystemExit("final_story.md is empty")

    dcee = read_json(out / "dcee_plan.json")
    story = read_json(out / "storyboard.json")

    blob = json.dumps({"dcee": dcee, "storyboard": story}, ensure_ascii=False).lower()
    bad = [g for g in GENERIC if g in blob]
    if bad:
        raise SystemExit(f"Generic fallback phrases detected: {bad}")

    chain = dcee.get("event_chain") or dcee.get("event_spine") or []
    if len(chain) < 3:
        raise SystemExit("DCEE event chain must have at least 3 events")

    for i, ev in enumerate(chain, start=1):
        for key in ["event", "causal_role", "visual_grounding", "key_objects", "evidence_objects"]:
            if not ev.get(key):
                raise SystemExit(f"Event {i} missing {key}: {ev}")

    for i, fr in enumerate(story, start=1):
        for key in ["event", "event_grounding", "emotion", "must_show"]:
            if not fr.get(key):
                raise SystemExit(f"Storyboard frame {i} missing {key}: {fr}")
        if not (fr.get("evidence_objects") or fr.get("emotion_evidence")):
            raise SystemExit(f"Storyboard frame {i} has no evidence objects/emotion evidence")

    print("DCEE output validation passed.")
    print("abstract length:", len(abstract))
    print("num events:", len(chain))
    print("num frames:", len(story))
    print("contact sheet:", out / "contact_sheet.png")


if __name__ == "__main__":
    main()
