"""
run_api_parallel.py — parallel API-based video inference dispatcher.

Reconstructed to drive the API video generators (Wan2.6 / Wan2.6-Flash via
Alibaba DashScope) that the scripts in scripts_inference/ expect. The local
checkpoint path is handled by run_local.py; this file handles the API path.

Reads a dataset summary.json, and for each task submits an image+prompt to the
API, downloads the returned video URL, and writes results in the same on-disk
layout as run_local.py so the evaluator can consume them:

    pred_root/{task_name}/{episode_name}/{attempt}/
        prompt/prompt.txt
        prompt/init_frame.png
        video/frame_00000.jpg ...        (best-effort, if opencv is available)
        {task_name}_{episode_name}.mp4

Usage (matches scripts_inference/run_wan26flash.sh):
    python run_api_parallel.py \
        --n_proc 4 --model wan2.6-i2v-flash \
        --gt_root data/<DATASET>/summary.json \
        --pred_root data/<DATASET>/generated_data/<TEST_NAME> \
        --api_key <DASHSCOPE_KEY> --duration 5 --n_attempts 1 \
        --prompt_key prompt_rewrite
"""
import argparse
import hashlib
import json
import os
import time
import shutil
import urllib.request
from pathlib import Path
from multiprocessing import Pool


# Defaults for deterministic sampling, kept in lock-step with run_local.py.
# Setting --sample_keep_prob 0.1 (or env KEEP_RATE=0.1) selects ~10% per dataset
# with the same model+seed+prompt_key giving the same subset across runs.
DEFAULT_KEEP_PROB = float(os.environ.get("KEEP_RATE", "0.1"))
DEFAULT_SAMPLE_SEED = int(os.environ.get("SAMPLE_SEED", "20260609"))


# --------------------------------------------------------------------------- #
# Task / prompt / path helpers (mirrors run_local.py)
# --------------------------------------------------------------------------- #
def get_tasks_from_json(json_path: str):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"summary json not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_tasks(tasks, rank: int, world_size: int):
    return tasks if world_size <= 1 else tasks[rank::world_size]


# --------------------------------------------------------------------------- #
# Deterministic sampling — kept identical to run_local.py so the same
# (sample_seed, model_key, prompt_key, sample_id) yields the same keep/skip.
# --------------------------------------------------------------------------- #
def stable_uniform_0_1(*parts) -> float:
    text = "||".join(str(x) for x in parts)
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(h[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def resolve_dataset_name(gt_root: str) -> str:
    p = Path(gt_root)
    if p.is_dir():
        return p.name
    return p.parent.name if p.name == "summary.json" else p.stem


def resolve_sample_id(task_info, dataset_name: str) -> str:
    gt_path = task_info.get("gt_path")
    if gt_path:
        return str(gt_path)
    task_name, episode_name = resolve_task_episode(task_info)
    return f"{dataset_name}/{task_name}/{episode_name}"


def should_keep_task(task_info, args, dataset_name: str):
    keep_prob = float(args.sample_keep_prob)
    if keep_prob >= 1.0:
        return True, 0.0, resolve_sample_id(task_info, dataset_name)
    if keep_prob <= 0.0:
        return False, 1.0, resolve_sample_id(task_info, dataset_name)

    sample_id = resolve_sample_id(task_info, dataset_name)
    model_key = (args.sample_model_key or "").strip() or args.model
    score = stable_uniform_0_1(args.sample_seed, model_key, args.prompt_key, sample_id)
    return score < keep_prob, score, sample_id


def filter_tasks_by_sampling(tasks, args, dataset_name: str):
    kept, skipped = [], []
    for item in tasks:
        keep, score, sample_id = should_keep_task(item, args, dataset_name)
        if keep:
            kept.append(item)
        else:
            task_name, episode_name = resolve_task_episode(item)
            skipped.append({
                "dataset": dataset_name,
                "model": args.model,
                "sample_model_key": (args.sample_model_key or "").strip() or args.model,
                "sample_seed": args.sample_seed,
                "sample_keep_prob": args.sample_keep_prob,
                "sample_id": sample_id,
                "task_name": task_name,
                "episode_name": episode_name,
                "score": score,
                "keep": False,
            })
    return kept, skipped


def resolve_prompt(task_info, prompt_key: str = "prompt"):
    if prompt_key not in task_info:
        raise KeyError(
            f"prompt_key '{prompt_key}' not in task_info. Available: {list(task_info.keys())}"
        )
    prompt = task_info.get(prompt_key, "")
    if isinstance(prompt, list):
        if not prompt:
            raise ValueError(f"task_info['{prompt_key}'] is an empty list")
        return str(prompt[0])
    if isinstance(prompt, str):
        return prompt
    raise ValueError(f"Unsupported prompt type for key '{prompt_key}': {type(prompt)}")


def resolve_task_episode(task_info):
    if "task_name" in task_info and "episode_name" in task_info:
        return str(task_info["task_name"]), str(task_info["episode_name"])
    gt_path = task_info.get("gt_path")
    if gt_path:
        parts = Path(gt_path).parts
        if len(parts) >= 3:
            return parts[-3], parts[-2]
    standard_name = str(task_info.get("standard_name", "sample_00000"))
    if "_" in standard_name:
        return tuple(standard_name.rsplit("_", 1))
    return "unknown_task", standard_name


def resolve_image_path(task_info, gt_root: str):
    """Resolve the init-frame path. The 'image'/'gt_path' fields in summary.json are
    relative to the project root (the dir that contains data/). Try, in order:
    absolute, CWD-relative, project-root-relative, dataset-dir-relative."""
    gp = Path(gt_root)
    roots = []
    if gp.name == "summary.json" and len(gp.parents) >= 3:
        roots.append(gp.parents[2])   # project root containing data/<dataset>/summary.json
    roots += [Path.cwd(), gp.parent, gp.parent.parent]
    cand = task_info.get("image") or ""
    if cand:
        p = Path(cand).expanduser()
        if p.is_absolute() and p.exists():
            return str(p)
        for r in roots:
            cp = r / cand
            if cp.exists():
                return str(cp.resolve())
    gt_path = task_info.get("gt_path")  # points at .../video.mp4; init_frame is sibling
    if gt_path:
        for r in roots:
            ip = (r / gt_path).parent / "prompt" / "init_frame.png"
            if ip.exists():
                return str(ip.resolve())
    raise FileNotFoundError(f"Could not resolve init_frame for task {task_info.get('task_name')}")


def save_prompt_bundle(prompt_text: str, img_p: str, prompt_dir: str):
    os.makedirs(prompt_dir, exist_ok=True)
    with open(os.path.join(prompt_dir, "prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt_text)
    if os.path.exists(img_p):
        shutil.copy2(img_p, os.path.join(prompt_dir, "init_frame.png"))


def download_video(url: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as resp, open(out_path, "wb") as f:
        shutil.copyfileobj(resp, f)
    if not (os.path.exists(out_path) and os.path.getsize(out_path) > 0):
        raise RuntimeError(f"Downloaded video empty: {out_path}")
    return out_path


def try_extract_frames(video_path: str, frames_dir: str):
    """Best-effort frame extraction; skipped silently if opencv is unavailable."""
    try:
        import cv2
    except Exception:
        return False
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(frames_dir, f"frame_{idx:05d}.jpg"), frame)
        idx += 1
    cap.release()
    return idx > 0


def ensure_dashscope_dims(img_p: str, min_side: int = 240, max_side: int = 8000, target_min: int = 480):
    """DashScope rejects images whose side is <240 or >8000 px (InvalidParameter).
    Some datasets (e.g. droid 320x180, egoscaler_human 224x224) are under-sized.
    If out of range, write an upscaled/downscaled copy (aspect preserved) to a
    cache dir and return that path; otherwise return the original path."""
    try:
        from PIL import Image
    except Exception:
        return img_p
    try:
        with Image.open(img_p) as im:
            w, h = im.size
            lo, hi = min(w, h), max(w, h)
            if lo >= min_side and hi <= max_side:
                return img_p
            if lo < min_side:
                scale = target_min / float(lo)
            else:  # hi > max_side
                scale = max_side / float(hi)
            nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
            # Key cache by target size so different model minimums (e.g. 240 vs 300)
            # don't collide on the same basename.
            cache = os.path.join(os.path.dirname(img_p), f".dashscope_resized_{target_min}")
            os.makedirs(cache, exist_ok=True)
            out = os.path.join(cache, os.path.basename(img_p))
            if not (os.path.exists(out) and os.path.getsize(out) > 0):
                im.convert("RGB").resize((nw, nh), Image.LANCZOS).save(out)
            return out
    except Exception:
        return img_p


# --------------------------------------------------------------------------- #
# Generator factory (API models)
# --------------------------------------------------------------------------- #
def build_api_generator(model: str, api_key: str):
    m = model.lower()
    if m.startswith("wan2.6") or m.startswith("wan2.1"):
        from inference.wan_generator import Wan26Generator
        return Wan26Generator(api_key=api_key, model=model)
    if m.startswith("veo"):
        from inference.veo_generator import VeoVideoGenerator
        return VeoVideoGenerator(api_key=api_key, model_name=model)
    if m.startswith("happyhorse"):
        from inference.dashscope_extra_generator import DashScopeVideoGenerator
        return DashScopeVideoGenerator(api_key=api_key, model=model, resolution="720P",
                                       media_type="first_frame")
    if m.startswith("vidu"):
        from inference.dashscope_extra_generator import DashScopeVideoGenerator
        # Vidu is deployed on a workspace-specific MaaS endpoint (not the default
        # dashscope.aliyuncs.com). Override via VIDU_BASE_URL if it ever changes.
        vidu_base = os.environ.get(
            "VIDU_BASE_URL",
            "https://ws-c8sw7d257aos18yg.cn-beijing.maas.aliyuncs.com/api/v1",
        )
        return DashScopeVideoGenerator(api_key=api_key, model=model, resolution="540P",
                                       base_url=vidu_base, media_type="image")
    raise ValueError(
        f"Unsupported API model '{model}'. "
        f"Supported prefixes: wan2.6-*, wan2.1-*, veo-*, happyhorse-*, vidu-*"
    )


def is_veo_model(model: str) -> bool:
    return model.lower().startswith("veo")


def is_wan_model(model: str) -> bool:
    m = model.lower()
    return m.startswith("wan2.6") or m.startswith("wan2.1")


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
def run_one_task(task_info, args):
    try:
        img_p = resolve_image_path(task_info, args.gt_root)
        # HappyHorse requires images >=300x300; Wan/Vidu accept >=240. Upscale accordingly.
        _m = (args.model or "").lower()
        _min_side = 300 if _m.startswith("happyhorse") else 240
        _target_min = 512 if _m.startswith("happyhorse") else 480
        img_p = ensure_dashscope_dims(img_p, min_side=_min_side, target_min=_target_min)
        prompt_text = resolve_prompt(task_info, args.prompt_key)
        task_name, episode_name = resolve_task_episode(task_info)
    except Exception as e:
        return f"SKIP {task_info.get('task_name', '?')}: {e}"

    gen = build_api_generator(args.model, args.api_key)
    pid = os.getpid()
    results = []
    for attempt in range(1, args.n_attempts + 1):
        attempt_dir = os.path.join(args.pred_root, task_name, episode_name, str(attempt))
        final_video_path = os.path.join(attempt_dir, f"{task_name}_{episode_name}.mp4")
        frames_dir = os.path.join(attempt_dir, "video")
        prompt_dir = os.path.join(attempt_dir, "prompt")

        if os.path.exists(final_video_path) and os.path.getsize(final_video_path) > 0:
            results.append(f"A{attempt}:Skip")
            continue
        os.makedirs(attempt_dir, exist_ok=True)
        try:
            print(f"[PID {pid}] 🚀 {task_name}/{episode_name} A{attempt} | {args.model}", flush=True)
            os.makedirs(os.path.dirname(final_video_path), exist_ok=True)
            if is_veo_model(args.model):
                # Veo: returns a Generated-video SDK object, save via .save_video()
                video_obj = gen.generate(prompt=prompt_text, img_path=img_p, duration=args.duration)
                gen.save_video(video_obj, final_video_path)
            else:
                # Wan2.6 / Wan2.1: returns a downloadable video URL
                video_url = gen.generate(prompt=prompt_text, img_path=img_p, duration=args.duration)
                download_video(video_url, final_video_path)
            save_prompt_bundle(prompt_text, img_p, prompt_dir)
            try_extract_frames(final_video_path, frames_dir)
            results.append(f"A{attempt}:OK")
        except Exception as e:
            print(f"[PID {pid}] ❌ {task_name}/{episode_name} A{attempt}: {e}", flush=True)
            results.append(f"A{attempt}:Fail")
        time.sleep(0.2)
    return f"{task_name}/{episode_name} -> {' '.join(results)}"


def _worker(packed):
    task_info, args = packed
    return run_one_task(task_info, args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_proc", type=int, default=4)
    parser.add_argument("--model", type=str, required=True, help="e.g. wan2.6-i2v-flash, wan2.6-i2v")
    parser.add_argument("--gt_root", type=str, required=True, help="path to dataset summary.json")
    parser.add_argument("--pred_root", type=str, required=True)
    parser.add_argument("--api_key", type=str, default="",
                        help="API key. Falls back to DASHSCOPE_API_KEY (Wan) or "
                             "GOOGLE_API_KEY/GEMINI_API_KEY (Veo) env vars when empty.")
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument("--n_attempts", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--max_tasks", type=int, default=-1, help="smoke test: only first N tasks")
    parser.add_argument("--prompt_key", type=str, default="prompt",
                        choices=["prompt", "prompt_prefix", "prompt_rewrite"])
    parser.add_argument("--sample_keep_prob", type=float, default=DEFAULT_KEEP_PROB,
                        help="Deterministic per-task keep probability for API runs. "
                             "Defaults to 0.1 (10%%) or env KEEP_RATE. Set 1.0 to disable.")
    parser.add_argument("--sample_seed", type=int, default=DEFAULT_SAMPLE_SEED,
                        help="Sampling seed; same seed+model+prompt_key+sample_id is stable across runs.")
    parser.add_argument("--sample_model_key", type=str, default="",
                        help="Optional stable model identifier used in the hash. "
                             "Use the same value across runs that should share a sampled subset.")
    args = parser.parse_args()

    if not args.api_key:
        if is_veo_model(args.model):
            args.api_key = (
                os.environ.get("GOOGLE_API_KEY")
                or os.environ.get("GEMINI_API_KEY")
                or ""
            )
        else:
            args.api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not args.api_key:
        raise SystemExit(
            "❌ --api_key is empty. Provide a key, or set "
            "DASHSCOPE_API_KEY (Wan) / GOOGLE_API_KEY (Veo) env var."
        )

    tasks = get_tasks_from_json(args.gt_root)
    total_loaded = len(tasks)
    tasks = split_tasks(tasks, args.rank, args.world_size)

    dataset_name = resolve_dataset_name(args.gt_root)
    if args.sample_keep_prob < 1.0:
        kept, skipped = filter_tasks_by_sampling(tasks, args, dataset_name)
        print(
            f"🎯 sample_keep_prob={args.sample_keep_prob} "
            f"(seed={args.sample_seed}, model_key={(args.sample_model_key or args.model)}, "
            f"prompt_key={args.prompt_key}) -> kept {len(kept)} / sharded {len(tasks)} "
            f"(loaded {total_loaded}, skipped {len(skipped)})",
            flush=True,
        )
        tasks = kept

    if args.max_tasks > 0:
        tasks = tasks[:args.max_tasks]
    os.makedirs(args.pred_root, exist_ok=True)
    print(f"🚀 API inference | model={args.model} | tasks={len(tasks)} | n_proc={args.n_proc} "
          f"| prompt_key={args.prompt_key} | duration={args.duration}s", flush=True)

    packed = [(t, args) for t in tasks]
    if args.n_proc <= 1:
        out = [_worker(p) for p in packed]
    else:
        with Pool(processes=args.n_proc) as pool:
            out = pool.map(_worker, packed)

    ok = sum(1 for r in out if "OK" in r)
    fail = sum(1 for r in out if "Fail" in r)
    skip = sum(1 for r in out if r.startswith("SKIP") or "Skip" in r)
    print(f"✅ DONE | ok={ok} fail={fail} skip={skip} total={len(out)} -> {args.pred_root}", flush=True)


if __name__ == "__main__":
    main()
