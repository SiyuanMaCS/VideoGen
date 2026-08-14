#!/usr/bin/env python3
"""Build PF_Wan's prompt@@first_frame manifest from dataset summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any


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
        return task, episode
    parts = Path(scalar(item.get("gt_path")).replace("\\", "/")).parts
    if len(parts) >= 3:
        return parts[-3], parts[-2]
    raise ValueError(f"Cannot resolve task and episode: {item}")


def stable_score(*parts: Any) -> float:
    digest = hashlib.sha256("||".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "unknown"


def find_image(
    item: dict[str, Any], project_root: Path, data_root: Path,
    dataset_root: Path, task: str, episode: str,
) -> Path:
    candidates = [dataset_root / "gt_data" / task / episode / "prompt" / "init_frame.png"]
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--keep-prob", type=float, default=1.0)
    parser.add_argument("--sample-seed", type=int, default=20260610)
    parser.add_argument("--sample-model-key", default="pf_wan14b")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--expected-count", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.keep_prob <= 1.0:
        parser.error("--keep-prob must be in [0, 1]")
    project_root = args.project_root.expanduser().resolve()
    data_root = args.data_root.expanduser()
    data_root = (project_root / data_root).resolve() if not data_root.is_absolute() else data_root.resolve()
    work_root = args.work_root.expanduser()
    work_root = (project_root / work_root).resolve() if not work_root.is_absolute() else work_root.resolve()
    run_root = work_root / args.name
    if run_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Work directory exists; pass --overwrite to replace it: {run_root}")
        shutil.rmtree(run_root)
    images = run_root / "images"
    images.mkdir(parents=True)

    mapping: list[dict[str, Any]] = []
    manifest: list[str] = []
    for dataset in args.datasets:
        dataset_root = data_root / dataset
        summary = dataset_root / "summary.json"
        if not summary.is_file():
            raise FileNotFoundError(f"Missing summary: {summary}")
        for item in load_rows(summary):
            task, episode = task_episode(item)
            sample_id = scalar(item.get("gt_path")) or f"{dataset}/{task}/{episode}"
            score = stable_score(args.sample_seed, args.sample_model_key, args.prompt_key, sample_id)
            if score >= args.keep_prob:
                continue
            prompt = " ".join(scalar(item.get(args.prompt_key)).split())
            if not prompt:
                raise ValueError(f"Empty {args.prompt_key}: {dataset}/{task}/{episode}")
            if "@@" in prompt:
                raise ValueError(f"Prompt contains reserved '@@': {dataset}/{task}/{episode}")
            source = find_image(item, project_root, data_root, dataset_root, task, episode)
            stem = f"{len(mapping):06d}--{safe_name(dataset)}--{safe_name(task)}--{safe_name(episode)}"
            linked = images / f"{stem}{source.suffix.lower() or '.png'}"
            try:
                linked.symlink_to(source)
            except OSError:
                shutil.copy2(source, linked)
            # Do not resolve this path: PF_Wan names outputs from this unique basename.
            manifest.append(f"{prompt}@@{linked.absolute()}")
            mapping.append({
                "engine_stem": stem, "dataset": dataset, "task": task,
                "episode": episode, "sample_id": sample_id, "prompt_key": args.prompt_key,
                "prompt": prompt, "source_image": str(source), "score": score,
                "run_name": args.run_name,
            })

    if not mapping:
        raise RuntimeError("No samples selected")
    (run_root / "manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    (run_root / "mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (run_root / "selection.jsonl").open("w", encoding="utf-8") as handle:
        for item in mapping:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    if args.expected_count >= 0 and len(mapping) != args.expected_count:
        raise RuntimeError(f"Expected {args.expected_count} samples, selected {len(mapping)}")
    print(f"Selected {len(mapping)} samples; manifest: {run_root / 'manifest.txt'}")


if __name__ == "__main__":
    main()
