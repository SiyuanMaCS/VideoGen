#!/usr/bin/env python3
"""Collect PF_Wan raw videos into the repository's generated_data layout."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def link_or_copy(source: Path, target: Path, overwrite: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"Destination exists: {target}")
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-name", default="", help="Override the run_name stored in mapping.json")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    mapping = json.loads(args.mapping.expanduser().read_text(encoding="utf-8"))
    raw_dir = args.raw_dir.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    packaged = 0
    missing: list[Path] = []
    for item in mapping:
        source_video = raw_dir / f"{item['engine_stem']}.mp4"
        if not source_video.is_file():
            missing.append(source_video)
            continue
        run_name = args.run_name or item.get("run_name") or item.get("dest_run_name")
        if not run_name:
            raise ValueError("No run name; pass --run-name or regenerate the mapping")
        sample_root = data_root / item["dataset"] / "generated_data" / run_name / item["task"] / item["episode"] / "1"
        link_or_copy(source_video, sample_root / f"{item['task']}_{item['episode']}.mp4", args.overwrite)
        link_or_copy(Path(item["source_image"]), sample_root / "prompt" / "init_frame.png", args.overwrite)
        (sample_root / "prompt" / "prompt.txt").write_text(item["prompt"] + "\n", encoding="utf-8")
        packaged += 1
    print(f"expected={len(mapping)}, packaged={packaged}, missing={len(missing)}")
    if missing and not args.allow_missing:
        raise RuntimeError("Missing raw videos, first entries:\n" + "\n".join(map(str, missing[:20])))


if __name__ == "__main__":
    main()
