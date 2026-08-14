#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path


def stable_uniform_0_1(*parts):
    text = "||".join(str(x) for x in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_prompt(value):
    if isinstance(value, list):
        if not value:
            raise ValueError("empty prompt list")
        return str(value[0])
    if isinstance(value, str):
        return value
    raise TypeError("unsupported prompt value type: %s" % type(value).__name__)


def resolve_task_episode(item):
    if item.get("task_name") and item.get("episode_name"):
        return str(item["task_name"]), str(item["episode_name"])
    gt_path = item.get("gt_path")
    if gt_path:
        parts = Path(gt_path).parts
        if len(parts) >= 3:
            return parts[-3], parts[-2]
    standard_name = str(item.get("standard_name", "sample_00000"))
    if "_" in standard_name:
        return standard_name.rsplit("_", 1)
    return "unknown_task", standard_name


def sample_id(item, dataset):
    return str(item.get("gt_path") or item.get("image") or "%s/%s" % (dataset, resolve_task_episode(item)))


def should_keep(item, dataset, prompt_key, args):
    keep_prob = float(args.sample_keep_prob)
    if keep_prob >= 1.0:
        return True, 0.0
    if keep_prob <= 0.0:
        return False, 1.0
    score = stable_uniform_0_1(args.sample_seed, args.sample_model_key, prompt_key, sample_id(item, dataset))
    return score < keep_prob, score


def discover_datasets(data_root, requested):
    if requested:
        return requested
    datasets = []
    for summary in sorted(Path(data_root).glob("*/summary.json")):
        datasets.append(summary.parent.name)
    return datasets


def test_name_for(prompt_key, args):
    if prompt_key == "prompt_prefix":
        suffix = "prefix"
    elif prompt_key == "prompt_rewrite":
        suffix = "rewrite"
    else:
        suffix = prompt_key
    return args.test_name_template.format(prompt_key=prompt_key, suffix=suffix, model=args.sample_model_key)


def maybe_link_or_copy(src, dst):
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
    except FileNotFoundError:
        pass
    try:
        os.link(str(src), str(dst))
    except OSError:
        try:
            os.symlink(os.path.relpath(str(src), str(dst.parent)), str(dst))
        except OSError:
            shutil.copy2(str(src), str(dst))


def copy_prompt_bundle(prompt_text, image_path, out_dir, mirror_dir):
    prompt_dir = Path(out_dir) / "prompt"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_txt = prompt_dir / "prompt.txt"
    prompt_txt.write_text(prompt_text, encoding="utf-8")
    init_dst = prompt_dir / "init_frame.png"
    if Path(image_path).exists() and not init_dst.exists():
        shutil.copy2(str(image_path), str(init_dst))

    if mirror_dir:
        mirror_prompt_dir = Path(mirror_dir) / "prompt"
        mirror_prompt_dir.mkdir(parents=True, exist_ok=True)
        maybe_link_or_copy(prompt_txt, mirror_prompt_dir / "prompt.txt")
        if init_dst.exists():
            maybe_link_or_copy(init_dst, mirror_prompt_dir / "init_frame.png")


def append_jsonl(path, record):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_pipe(args):
    import torch
    from diffusers import Cosmos3OmniPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside this job")

    t0 = time.time()
    device_map = getattr(args, "device_map", None) or None
    print("[Cosmos3] loading pipeline %s  device_map=%s" % (args.model_id, device_map), flush=True)
    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        safety_checker=None,
        enable_safety_checker=False,
        token=os.environ.get("HF_TOKEN") or None,
    )
    if device_map:
        load_kwargs["device_map"] = device_map
    if getattr(args, "load_in_8bit", False):
        from diffusers import PipelineQuantizationConfig
        load_kwargs["quantization_config"] = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_8bit",
            quant_kwargs={"load_in_8bit": True},
            components_to_quantize=["transformer"],
        )
    pipe = Cosmos3OmniPipeline.from_pretrained(args.model_id, **load_kwargs)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=args.flow_shift)
    if not device_map:
        if getattr(args, "sequential_offload", False):
            pipe.enable_sequential_cpu_offload()
        elif args.cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(args.device)
    if getattr(args, "attention_slicing", False):
        pipe.enable_attention_slicing(1)
    print("[Cosmos3] pipeline loaded in %.1fs" % (time.time() - t0), flush=True)
    return pipe, torch


def generate_one(pipe, torch, prompt_text, image_path, output_path, args, seed):
    from diffusers.utils import export_to_video, load_image

    generator = torch.Generator(device=args.device).manual_seed(seed)
    image = load_image(str(image_path))

    t0 = time.time()
    result = pipe(
        prompt=prompt_text,
        negative_prompt=args.negative_prompt,
        image=image,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        fps=args.fps,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        enable_sound=False,
        add_resolution_template=False,
        add_duration_template=False,
        generator=generator,
    )
    tmp_path = output_path.with_name("%s.tmp%s" % (output_path.stem, output_path.suffix))
    if tmp_path.exists():
        tmp_path.unlink()
    export_to_video(result.video, str(tmp_path), fps=args.fps, macro_block_size=1)
    os.replace(str(tmp_path), str(output_path))
    return time.time() - t0


def build_work_items(args):
    data_root = Path(args.data_root)
    items = []
    counts = {}
    summary_fname = getattr(args, "summary_name", "summary.json") or "summary.json"
    for dataset in discover_datasets(data_root, args.datasets):
        summary_path = data_root / dataset / summary_fname
        if not summary_path.exists():
            # fall back to default summary.json if custom name not found
            summary_path = data_root / dataset / "summary.json"
        if not summary_path.exists():
            print("[Warn] missing summary: %s" % summary_path, flush=True)
            continue
        tasks = read_json(summary_path)
        counts.setdefault(dataset, {"total": len(tasks), "kept": {}})
        for prompt_key in args.prompt_keys:
            kept_for_key = 0
            for idx, item in enumerate(tasks):
                if prompt_key not in item:
                    continue
                keep, score = should_keep(item, dataset, prompt_key, args)
                if not keep:
                    continue
                kept_for_key += 1
                items.append(
                    {
                        "dataset": dataset,
                        "prompt_key": prompt_key,
                        "index": idx,
                        "score": score,
                        "item": item,
                    }
                )
            counts[dataset]["kept"][prompt_key] = kept_for_key
    return items, counts


def run(args):
    project_root = Path(args.project_root).resolve()
    os.chdir(str(project_root))

    all_items, counts = build_work_items(args)
    sharded_items = all_items[args.rank :: args.world_size] if args.world_size > 1 else all_items
    if args.max_items > 0:
        sharded_items = sharded_items[: args.max_items]

    print("=" * 80, flush=True)
    print("Cosmos3 zero-shot image-to-video", flush=True)
    print("project_root      : %s" % project_root, flush=True)
    print("data_root         : %s" % args.data_root, flush=True)
    print("output_root       : %s" % args.output_root, flush=True)
    print("model             : %s" % args.model_id, flush=True)
    print("prompt_keys       : %s" % ",".join(args.prompt_keys), flush=True)
    print("sample_keep_prob  : %.3f" % args.sample_keep_prob, flush=True)
    print("rank/world_size   : %s/%s" % (args.rank, args.world_size), flush=True)
    print("items total/shard : %s/%s" % (len(all_items), len(sharded_items)), flush=True)
    print("video             : %sx%s %s frames @ %s fps, steps=%s" % (args.width, args.height, args.num_frames, args.fps, args.num_inference_steps), flush=True)
    print("=" * 80, flush=True)
    print(json.dumps(counts, indent=2, ensure_ascii=False), flush=True)

    if args.dry_run:
        return
    if not sharded_items:
        print("[Done] no work for this rank", flush=True)
        return

    pipe, torch = load_pipe(args)
    log_path = Path(args.log_dir) / ("cosmos3_progress_rank%s.jsonl" % args.rank)

    done = 0
    skipped = 0
    failed = 0
    started_at = time.time()

    for local_idx, rec in enumerate(sharded_items, start=1):
        dataset = rec["dataset"]
        prompt_key = rec["prompt_key"]
        item = rec["item"]
        task_name, episode_name = resolve_task_episode(item)
        test_name = test_name_for(prompt_key, args)
        attempt_dir = Path(args.data_root) / dataset / "generated_data" / test_name / task_name / episode_name / "1"
        final_mp4 = attempt_dir / ("%s_%s.mp4" % (task_name, episode_name))
        rel_attempt_dir = Path(dataset) / "generated_data" / test_name / task_name / episode_name / "1"
        mirror_dir = Path(args.output_root) / rel_attempt_dir if args.output_root else None
        mirror_mp4 = mirror_dir / final_mp4.name if mirror_dir else None

        try:
            prompt_text = resolve_prompt(item[prompt_key])
            image_path = Path(item["image"])
            if not image_path.is_absolute():
                image_path = project_root / image_path
            if not image_path.exists():
                raise FileNotFoundError("missing init image: %s" % image_path)

            attempt_dir.mkdir(parents=True, exist_ok=True)
            copy_prompt_bundle(prompt_text, image_path, attempt_dir, mirror_dir)

            if final_mp4.exists() and final_mp4.stat().st_size > 0:
                skipped += 1
                if mirror_mp4:
                    maybe_link_or_copy(final_mp4, mirror_mp4)
                append_jsonl(log_path, {"event": "skip", "dataset": dataset, "prompt_key": prompt_key, "task": task_name, "episode": episode_name, "path": str(final_mp4), "time": time.time()})
                print("[%s/%s] skip %s %s/%s" % (local_idx, len(sharded_items), prompt_key, task_name, episode_name), flush=True)
                continue

            seed = args.seed + int(rec["index"]) + (100000 if prompt_key == "prompt_rewrite" else 0)
            print("[%s/%s] start %s %s/%s -> %s" % (local_idx, len(sharded_items), prompt_key, dataset, task_name, final_mp4), flush=True)
            seconds = generate_one(pipe, torch, prompt_text, image_path, final_mp4, args, seed)
            fail_marker = Path(str(final_mp4) + ".failed.txt")
            if fail_marker.exists():
                fail_marker.unlink()
            if mirror_mp4:
                maybe_link_or_copy(final_mp4, mirror_mp4)
            done += 1
            append_jsonl(
                log_path,
                {
                    "event": "done",
                    "dataset": dataset,
                    "prompt_key": prompt_key,
                    "task": task_name,
                    "episode": episode_name,
                    "path": str(final_mp4),
                    "mirror_path": str(mirror_mp4) if mirror_mp4 else "",
                    "seconds": seconds,
                    "time": time.time(),
                },
            )
            avg = (time.time() - started_at) / max(1, done + skipped + failed)
            remaining = len(sharded_items) - local_idx
            print("[%s/%s] done in %.1fs | avg %.1fs/item | rank eta %.1fh" % (local_idx, len(sharded_items), seconds, avg, remaining * avg / 3600.0), flush=True)
        except Exception as exc:
            failed += 1
            err_path = Path(str(final_mp4) + ".failed.txt")
            err_path.parent.mkdir(parents=True, exist_ok=True)
            err_path.write_text("%s\n\n%s" % (repr(exc), traceback.format_exc()), encoding="utf-8")
            append_jsonl(
                log_path,
                {
                    "event": "failed",
                    "dataset": dataset,
                    "prompt_key": prompt_key,
                    "task": task_name,
                    "episode": episode_name,
                    "path": str(final_mp4),
                    "error": repr(exc),
                    "time": time.time(),
                },
            )
            print("[%s/%s] failed %s %s/%s: %s" % (local_idx, len(sharded_items), prompt_key, task_name, episode_name, exc), flush=True)
            if args.stop_on_error:
                raise

    print("[Done] rank=%s generated=%s skipped=%s failed=%s elapsed=%.1fh" % (args.rank, done, skipped, failed, (time.time() - started_at) / 3600.0), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Cosmos3-Nano zero-shot image-to-video over MLLM-as-World-Judge summaries.")
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--output_root", default="output_video")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--prompt_keys", nargs="+", default=["prompt_rewrite", "prompt_prefix"], choices=["prompt", "prompt_prefix", "prompt_rewrite"])
    parser.add_argument("--test_name_template", default="cosmos3_nano_{suffix}")
    parser.add_argument("--model_id", default="nvidia/Cosmos3-Nano")
    parser.add_argument("--sample_keep_prob", type=float, default=0.4)
    parser.add_argument("--sample_seed", type=int, default=20260609)
    parser.add_argument("--sample_model_key", default="cosmos3_nano")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--max_items", type=int, default=-1, help="Limit work items after sharding. Useful for smoke tests.")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--num_inference_steps", type=int, default=35)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--flow_shift", type=float, default=10.0)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--sequential_offload", action="store_true",
                        help="Use enable_sequential_cpu_offload (layer-by-layer, fits huge models in <80GB GPU)")
    parser.add_argument("--attention_slicing", action="store_true",
                        help="Use enable_attention_slicing(1) to reduce peak attention memory")
    parser.add_argument("--device_map", default=None,
                        help="Pass device_map to from_pretrained (e.g. 'auto' to split across all visible GPUs)")
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="Load model in INT8 via bitsandbytes (~half VRAM, fits Super on single 80GB GPU)")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--stop_on_error", action="store_true")
    parser.add_argument("--summary_name", default="summary.json",
                        help="Name of the summary file to read from each dataset dir (default: summary.json)")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
