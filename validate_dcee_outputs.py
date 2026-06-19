from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dce_vistory.llm import build_llm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="openrouter")
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "openai/gpt-4o-mini"))
    args = ap.parse_args()

    print("OPENAI_API_KEY exists:", bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")))
    print("OPENAI_BASE_URL:", os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1")
    print("MODEL:", args.model)

    llm = build_llm({"provider": args.provider, "model": args.model})
    out = llm.generate(
        "You are a strict DCEE visual storytelling planner.",
        "Return one sentence confirming API mode. Mention: woodcutter, old iron axe, river, sadness.",
        temperature=0.0,
        max_tokens=80,
    )
    print("\nAPI RESPONSE:\n", out)


if __name__ == "__main__":
    main()
