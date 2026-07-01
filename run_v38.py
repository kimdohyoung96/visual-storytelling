from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.dce_vistory.pipeline_crossattn_butterfly import CrossAttentionButterflyDCEViStoryPipeline


def main():
    ap = argparse.ArgumentParser(description='Run DCEE-CausalVerse V38 pipeline')
    ap.add_argument('--config', required=True, help='Path to config YAML')
    ap.add_argument('--input', required=True, help='Path to input JSON')
    ap.add_argument('--output', '--out', dest='output', required=True, help='Output directory')
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    sample = json.loads(Path(args.input).read_text(encoding='utf-8'))

    pipe = CrossAttentionButterflyDCEViStoryPipeline(cfg)
    result = pipe.run(sample=sample, out_dir=Path(args.output))

    summary = {
        'output_dir': str(Path(args.output)),
        'contact_sheet': str(getattr(result, 'contact_sheet_path', '')),
        'final_story': str(Path(args.output) / 'final_story.md'),
        'evaluation': str(Path(args.output) / 'evaluation.json'),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
