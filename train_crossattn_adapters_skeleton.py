import argparse
from pathlib import Path

import torch

from dce_vistory.adapters_pytorch import ButterflyAdapterStack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/adapter_init.pt")
    args = parser.parse_args()

    adapter = ButterflyAdapterStack(
        character_tokens=8,
        world_tokens=8,
        emotion_tokens=6,
        event_tokens=6,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(adapter.state_dict(), args.out)
    print(f"Saved initial adapter checkpoint to {args.out}")


if __name__ == "__main__":
    main()
