#!/usr/bin/env python3
"""Run benchmark samples with LingBot-Video's native FSDP pipeline."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
from PIL import Image

from lingbot_video import runner as lingbot_runner
from lingbot_video.pipeline_lingbot_video import DEFAULT_NEGATIVE_PROMPT
from lingbot_video.utils import num_frames_from_duration


def is_rank_zero() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def scalar(value: Any) -> str:
    if isinstance(value, list):
        value = value[0] if value else ""
    return "" if value is None else str(value).strip()


def load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "items", "samples", "tasks", "records"):
            if isinstance(data.get(key), list):
                return data[key]
    raise TypeError(f"Unsupported summary structure: {path}")


def task_episode(item: dict[str, Any]) -> tuple[str, str]:
    task = scalar(item.get("task_name") or item.get("task"))
    episode = scalar(item.get("episode_name") or item.get("episode"))
    if task and episode:
        return task.replace("\\", "/").split("/")[-1], episode.replace("\\", "/").split("/")[-1]
    gt_path = scalar(item.get("gt_path"))
    parts = Path(gt_path.replace("\\", "/")).parts
    if len(parts) >= 3:
        return parts[-3], parts[-2]
    raise ValueError(f"Cannot resolve task and episode: {item}")


def image_path(
    item: dict[str, Any], project_root: Path, data_root: Path,
    dataset_root: Path, task: str, episode: str,
) -> Path:
    candidates = [
        dataset_root / "gt_data" / task / episode / "prompt" / "init_frame.png",
        dataset_root / "gt_data" / task / episode / "prompt" / "init_frame.jpg",
    ]
    raw = scalar(
        item.get("image") or item.get("image_path")
        or item.get("init_frame") or item.get("init_frame_path")
    )
    if raw:
        path = Path(os.path.expandvars(raw)).expanduser()
        candidates = ([path] if path.is_absolute() else [project_root / path, dataset_root / path, data_root / path]) + candidates
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"No conditioning image for {dataset_root.name}/{task}/{episode}")


def sample_seed(base_seed: int, sample_id: str) -> int:
    offset = int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:8], 16)
    return (base_seed + offset) % 2_147_483_647


def build_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for dataset in args.datasets:
        dataset_root = args.data_root / dataset
        summary = dataset_root / "summary.json"
        if not summary.is_file():
            raise FileNotFoundError(f"Missing summary: {summary}")
        for item in load_rows(summary):
            task, episode = task_episode(item)
            prompt = scalar(item.get(args.prompt_key))
            if not prompt:
                raise ValueError(f"Empty {args.prompt_key}: {dataset}/{task}/{episode}")
            output_dir = dataset_root / "generated_data" / args.run_name / task / episode / str(args.attempt)
            output = output_dir / f"{task}_{episode}.mp4"
            if not args.overwrite and output.is_file() and output.stat().st_size >= args.min_existing_bytes:
                continue
            samples.append({
                "dataset": dataset, "task": task, "episode": episode,
                "sample_id": f"{dataset}/{task}/{episode}", "prompt": prompt,
                "image": image_path(item, args.project_root, args.data_root, dataset_root, task, episode),
                "output_dir": output_dir, "output": output,
            })
    return samples


def load_pipeline(model_dir: Path):
    dtypes = {
        "default": torch.bfloat16, "transformer": torch.bfloat16,
        "text_encoder": torch.bfloat16, "vae": torch.float32,
    }
    load_args = SimpleNamespace(
        model_dir=str(model_dir), engine="diffusers", mode="ti2v",
        transformer_subfolder="transformer",
    )
    pipe, engine = lingbot_runner._load_pipe(
        load_args, dtypes, defer_transformer_to_device=True,
    )
    mesh = lingbot_runner.init_fsdp_inference_mesh()
    fsdp_info = lingbot_runner._apply_fsdp_inference_if_requested(pipe, True, mesh)
    lingbot_runner._configure_pipeline_logs(pipe)
    return pipe, engine, fsdp_info


def generate(pipe: Any, sample: dict[str, Any], args: argparse.Namespace) -> None:
    seed = sample_seed(args.seed, sample["sample_id"])
    generator = torch.Generator(device=torch.cuda.current_device()).manual_seed(seed)
    with Image.open(sample["image"]) as source:
        image = source.convert("RGB")
    result = pipe(
        prompt=sample["prompt"], image=image,
        negative_prompt=args.negative_prompt,
        height=args.height, width=args.width,
        num_frames=num_frames_from_duration(args.duration, args.fps),
        num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
        shift=args.shift, generator=generator, output_type="np",
        cfg_parallel_group=None, batch_cfg=False, null_cond_clone_zero=False,
    )
    if is_rank_zero():
        output_dir = sample["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        partial = sample["output"].with_suffix(".partial.mp4")
        frames = lingbot_runner._extract_frames(result)
        lingbot_runner._save_frames(frames, "ti2v", partial, args.fps)
        if not partial.is_file() or partial.stat().st_size == 0:
            raise RuntimeError(f"Empty generated video: {partial}")
        partial.replace(sample["output"])
        prompt_dir = output_dir / "prompt"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        (prompt_dir / "prompt.txt").write_text(sample["prompt"] + "\n", encoding="utf-8")
        image.save(prompt_dir / "init_frame.png")
        config = {
            "model": "LingBot-Video-MoE-30B-A3B", "mode": "ti2v",
            "backend": "diffusers", "fsdp": True, "base_only": True,
            "prompt_rewriter": False, "auto_negative": False, "refiner": False,
            "prompt_key": args.prompt_key, "seed": seed, "duration": args.duration,
            "fps": args.fps, "height": args.height, "width": args.width,
            "num_frames": num_frames_from_duration(args.duration, args.fps),
            "steps": args.steps, "guidance_scale": args.guidance_scale, "shift": args.shift,
        }
        (output_dir / "generation_config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8",
        )
        print(f"[DONE] {sample['output']}", flush=True)
    del result, image, generator
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--min-existing-bytes", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.project_root = args.project_root.expanduser().resolve()
    args.data_root = (args.project_root / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root.resolve()
    args.model_dir = args.model_dir.expanduser().resolve()
    return args


def main() -> None:
    args = parse_args()
    lingbot_runner._init_parallel(cfg_degree=1, context_degree=1, enable_fsdp_inference=True)
    try:
        payload = [None]
        if is_rank_zero():
            try:
                payload[0] = {"ok": True, "samples": build_samples(args)}
            except Exception:
                payload[0] = {"ok": False, "error": traceback.format_exc()}
        dist.broadcast_object_list(payload, src=0)
        if not payload[0]["ok"]:
            raise RuntimeError("Manifest construction failed:\n" + payload[0]["error"])
        samples = payload[0]["samples"]
        if not samples:
            if is_rank_zero():
                print("Nothing to generate.")
            return
        pipe, engine, fsdp_info = load_pipeline(args.model_dir)
        if is_rank_zero():
            print(f"Loaded engine={engine}, fsdp={fsdp_info}, samples={len(samples)}")
        dist.barrier()
        for sample in samples:
            generate(pipe, sample, args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
