import argparse
from pathlib import Path

from dce_vistory.pipeline_butterfly import ButterflyDCEViStoryPipeline
from dce_vistory.utils import load_yaml, load_json, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    sample = load_json(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = ButterflyDCEViStoryPipeline(cfg)
    result = pipe.run(sample, out_dir)
    save_json(result.to_dict(), out_dir / "run_result.json")
    print(f"[DONE] Butterfly DCE-ViStory results saved to: {out_dir}")


if __name__ == "__main__":
    main()
