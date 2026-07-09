from __future__ import annotations

import argparse
import json
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.dce_vistory.pipeline_crossattn_butterfly import CrossAttentionButterflyDCEViStoryPipeline


def _parse_gpu_ids(raw: str | None) -> List[int]:
    if raw is None or str(raw).strip() == "":
        raw = os.environ.get("DCEE_GPU_IDS", "") or os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ids: List[int] = []
    for part in str(raw or "").replace(";", ",").split(","):
        part = part.strip()
        if part.isdigit():
            ids.append(int(part))
    out: List[int] = []
    for x in ids:
        if x not in out:
            out.append(x)
    return out


def _torch_gpu_report() -> Dict[str, Any]:
    try:
        import torch
        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if available else 0
        names = [torch.cuda.get_device_name(i) for i in range(count)] if available else []
        return {"cuda_available": available, "device_count": count, "device_names": names}
    except Exception as e:
        return {"cuda_available": False, "device_count": 0, "device_names": [], "error": f"{type(e).__name__}: {e}"}


def _print_nvidia_smi() -> None:
    if not shutil.which("nvidia-smi"):
        print("[GPU] nvidia-smi not found on PATH")
        return
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        print("[GPU] nvidia-smi summary:")
        print(out.strip())
    except Exception as e:
        print(f"[GPU] nvidia-smi failed: {type(e).__name__}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run DCEE-CausalVerse V64.5 with OOM-safe SDXL candidate-parallel multi-GPU support")
    ap.add_argument("--config", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", "--out", dest="output", required=True)
    ap.add_argument("--gpu-ids", default=None, help="Visible CUDA GPU ids to use, e.g. 0,1,2,3. These are after CUDA_VISIBLE_DEVICES remapping.")
    ap.add_argument("--require-gpus", type=int, default=0, help="Fail early if PyTorch sees fewer than this many GPUs.")
    ap.add_argument("--disable-multi-gpu", action="store_true", help="Force single-GPU generation even if config enables multi_gpu.")
    ap.add_argument("--max-parallel-generators", type=int, default=None, help="Max SDXL pipeline replicas to run in parallel. Use 4 for A100 x4.")
    ap.add_argument("--tensor-parallel-size", type=int, default=None, help="Accepted for compatibility; for SDXL this is treated as max parallel candidate generators, not tensor parallelism.")
    ap.add_argument("--min-free-memory-gb", type=float, default=28.0, help="Skip GPUs with less free memory than this before creating SDXL worker replicas. V64.5 default is 28GB to avoid half-busy A100 crashes.")
    ap.add_argument("--safe-generation-size", type=int, default=768, help="Generate SDXL candidates at this resolution first, then resize to configured output size. Prevents OOM/illegal-address after full-res attempts.")
    ap.add_argument("--safe-num-inference-steps", type=int, default=34, help="Cap diffusion steps in CUDA-failsafe multi-GPU mode.")
    ap.add_argument("--no-force-safe-lowres", action="store_true", help="Disable preemptive low-res generation. Not recommended for 4x SDXL replicas.")
    ap.add_argument("--no-skip-busy-gpus", action="store_true", help="Do not skip GPUs that are already busy. Not recommended for SDXL.")
    ap.add_argument("--strict-worker-failures", action="store_true", help="Crash if one GPU worker fails instead of continuing with remaining generated candidates.")
    ap.add_argument("--disable-process-isolated-multigpu", action="store_true", help="Use old thread-based multi-GPU mode. Not recommended; CUDA illegal-address errors can poison the process.")
    ap.add_argument("--print-nvidia-smi", action="store_true")
    args = ap.parse_args()

    # V64.5 CUDA-safe defaults. Must be set early, before heavy CUDA allocations.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    try:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    except Exception:
        pass

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    sample = json.loads(Path(args.input).read_text(encoding="utf-8"))
    sample.setdefault("image_path", "")
    sample.setdefault("protagonist_reference_paths", [])
    sample.setdefault("canonical_reference_sheet_path", "")

    report = _torch_gpu_report()
    print("[GPU] torch report:")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.print_nvidia_smi:
        _print_nvidia_smi()
    if args.require_gpus and int(report.get("device_count", 0)) < args.require_gpus:
        raise RuntimeError(f"Expected at least {args.require_gpus} CUDA GPUs, but PyTorch sees {report.get('device_count', 0)}.")

    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    if not gpu_ids and int(report.get("device_count", 0)) > 0:
        gpu_ids = list(range(int(report.get("device_count", 0))))

    img_cfg = cfg.setdefault("image_generator", {})
    if args.disable_multi_gpu:
        img_cfg["multi_gpu"] = False
    else:
        img_cfg["multi_gpu"] = bool(len(gpu_ids) > 1)
    if gpu_ids:
        img_cfg["gpu_ids"] = gpu_ids
    max_parallel = args.max_parallel_generators or args.tensor_parallel_size or len(gpu_ids) or 1
    img_cfg["max_parallel_generators"] = int(max_parallel)
    img_cfg["min_free_memory_gb"] = float(args.min_free_memory_gb)
    img_cfg["skip_busy_gpus"] = not bool(args.no_skip_busy_gpus)
    img_cfg["continue_on_worker_failure"] = not bool(args.strict_worker_failures)
    img_cfg["force_safe_lowres_generation"] = not bool(args.no_force_safe_lowres)
    img_cfg["safe_generation_width"] = int(args.safe_generation_size)
    img_cfg["safe_generation_height"] = int(args.safe_generation_size)
    img_cfg["safe_num_inference_steps"] = int(args.safe_num_inference_steps)
    img_cfg["disable_fusion_in_multigpu_safe"] = True
    img_cfg.setdefault("oom_safe_generation", True)
    img_cfg.setdefault("enable_vae_tiling", True)
    img_cfg.setdefault("enable_vae_slicing", True)
    img_cfg.setdefault("enable_attention_slicing", True)

    print("[GPU] effective image_generator multi-GPU config:")
    print(json.dumps({
        "device": img_cfg.get("device", "cuda"),
        "multi_gpu": img_cfg.get("multi_gpu", False),
        "gpu_ids": img_cfg.get("gpu_ids", []),
        "max_parallel_generators": img_cfg.get("max_parallel_generators", 1),
        "note": "SDXL uses candidate-parallel generation across GPUs; --tensor-parallel-size is not vLLM tensor parallelism here.",
    }, ensure_ascii=False, indent=2))

    pipe = CrossAttentionButterflyDCEViStoryPipeline(cfg)
    pipe.run(sample=sample, out_dir=Path(args.output))
    print(json.dumps({
        "output_dir": str(Path(args.output)),
        "final_story": str(Path(args.output) / "final_story.md"),
        "storyboard": str(Path(args.output) / "storyboard.json"),
        "evaluation": str(Path(args.output) / "evaluation.json"),
        "multi_gpu_enabled": img_cfg.get("multi_gpu", False),
        "gpu_ids": img_cfg.get("gpu_ids", []),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
