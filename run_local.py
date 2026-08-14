import argparse
import hashlib
import json
import os
import shutil
import shlex
import time
from pathlib import Path

import cv2


def get_tasks_from_json(json_path: str):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"summary json not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_tasks(tasks, rank: int, world_size: int):
    return tasks if world_size <= 1 else tasks[rank::world_size]


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
    # Prefer gt_path because it usually contains dataset/task/episode and is stable across runs.
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
    model_key = args.sample_model_key.strip() or args.model

    score = stable_uniform_0_1(args.sample_seed, model_key, args.prompt_key, sample_id)
    return score < keep_prob, score, sample_id


def filter_tasks_by_sampling(tasks, args, dataset_name: str):
    kept = []
    skipped = []

    for item in tasks:
        keep, score, sample_id = should_keep_task(item, args, dataset_name)
        task_name, episode_name = resolve_task_episode(item)
        record = {
            "dataset": dataset_name,
            "model": args.model,
            "sample_model_key": args.sample_model_key.strip() or args.model,
            "sample_seed": args.sample_seed,
            "sample_keep_prob": args.sample_keep_prob,
            "sample_id": sample_id,
            "task_name": task_name,
            "episode_name": episode_name,
            "score": score,
            "keep": keep,
        }
        if keep:
            kept.append(item)
        else:
            skipped.append(record)

    return kept, skipped


# def resolve_prompt(task_info):
#     prompt = task_info.get("prompt", "")
#     if isinstance(prompt, list):
#         if not prompt:
#             raise ValueError("task_info['prompt'] is an empty list")
#         return str(prompt[0])
#     if isinstance(prompt, str):
#         return prompt
#     raise ValueError(f"Unsupported prompt type: {type(prompt)}")


def resolve_prompt(task_info, prompt_key: str = "prompt"):
    if prompt_key not in task_info:
        raise KeyError(f"prompt_key '{prompt_key}' not found in task_info. Available keys: {list(task_info.keys())}")

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
        p = Path(gt_path)
        parts = p.parts
        if len(parts) >= 3:
            return parts[-3], parts[-2]

    standard_name = str(task_info.get("standard_name", "sample_00000"))
    if "_" in standard_name:
        return standard_name.rsplit("_", 1)
    return "unknown_task", standard_name


def build_generator(args):
    if args.model == "wow":
        from inference.wow_generator import WoWGenerator
        return WoWGenerator(
            wow_root=args.wow_root,
            checkpoint_folder=args.checkpoint_folder,
            custom_checkpoint=args.custom_checkpoint,
            gpu=args.gpu,
            persistent_param_gb=args.persistent_param_gb,
            steps=args.steps,
            num_frames=args.num_frames,
            tiled=args.tiled,
            seed=args.seed,
        )

    if args.model == "cosmos":
        from inference.cosmos_generator import CosmosGenerator
        return CosmosGenerator(
            cosmos_root=args.cosmos_root,
            checkpoint_dir=args.checkpoint_folder,
            model=args.cosmos_model,
            inference_type=args.cosmos_inference_type,
            seed=args.seed,
            num_frames=None if args.cosmos_num_frames < 0 else args.cosmos_num_frames,
            steps=None if args.cosmos_steps < 0 else args.cosmos_steps,
        )

    if args.model == "cosmos3":
        from inference.cosmos3_generator import Cosmos3Generator
        return Cosmos3Generator(
            cosmos_framework_root=args.cosmos3_root,
            checkpoint_path=args.cosmos3_checkpoint,
            seed=args.seed,
            num_gpus=args.cosmos3_num_gpus,
            parallelism_preset=args.cosmos3_parallelism,
            extra_cli_args=shlex.split(args.cosmos3_extra_args) if args.cosmos3_extra_args else None,
        )

    if args.model == "cogvideo":
        from inference.cogvideo_generator import CogVideoGenerator
        return CogVideoGenerator(
            checkpoint_dir=args.checkpoint_folder,
            gpu=args.gpu,
            seed=args.seed,
            num_inference_steps=args.cogvideo_steps,
            num_frames=args.cogvideo_num_frames,
            guidance_scale=args.cogvideo_guidance_scale,
            fps=args.cogvideo_fps,
            use_model_cpu_offload=args.cogvideo_model_cpu_offload,
            use_vae_tiling=not args.cogvideo_disable_vae_tiling,
            use_vae_slicing=not args.cogvideo_disable_vae_slicing,
            width=args.cogvideo_width if args.cogvideo_width > 0 else None,
            height=args.cogvideo_height if args.cogvideo_height > 0 else None,
        )

    if args.model == "gigaworld":
        from inference.gigaworld_generator import GigaWorldGenerator
        return GigaWorldGenerator(
            gigaworld_root=args.gigaworld_root,
            checkpoint_dir=args.checkpoint_folder,
            gpu=args.gpu,
            seed=args.seed,
            extra_infer_args=shlex.split(args.gigaworld_extra_args),
        )

    if args.model == "wan22":
        from inference.wan22_generator import Wan22Generator
        return Wan22Generator(
            wan_root=args.wan_root,
            checkpoint_dir=args.checkpoint_folder,
            gpu=args.gpu,
            gpu_ids=args.wan_gpu_ids,
            task=args.wan_task,
            size=args.wan_size,
            python_bin=args.wan_python_bin,
            use_torchrun=args.wan_use_torchrun,
            torchrun_bin=args.wan_torchrun_bin,
            nproc_per_node=args.wan_nproc_per_node,
            t5_fsdp=args.wan_t5_fsdp,
            dit_fsdp=args.wan_dit_fsdp,
            ulysses_size=args.wan_ulysses_size,
            offload_model=args.wan_offload_model,
            convert_model_dtype=args.wan_convert_model_dtype,
            t5_cpu=args.wan_t5_cpu,
            frame_num=args.wan_frame_num,
            sample_solver=args.wan_sample_solver,
            sample_steps=args.wan_sample_steps,
            sample_shift=args.wan_sample_shift,
            sample_guide_scale=args.wan_sample_guide_scale,
            use_prompt_extend=args.wan_use_prompt_extend,
            prompt_extend_method=args.wan_prompt_extend_method,
            prompt_extend_model=args.wan_prompt_extend_model,
            prompt_extend_target_lang=args.wan_prompt_extend_target_lang,
            extra_args=args.wan_extra_args,
        )

    if args.model == "abot_physworld":
        from inference.abot_physworld_generator import ABotPhysWorldGenerator
        return ABotPhysWorldGenerator(
            abot_root=args.abot_root,
            checkpoint_path=args.abot_checkpoint_path,
            gpu=args.gpu,
            height=args.abot_height,
            width=args.abot_width,
            num_frames=args.abot_num_frames,
            num_inference_steps=args.abot_num_inference_steps,
            cfg_scale=args.abot_cfg_scale,
            cache_dir=args.abot_cache_dir if args.abot_cache_dir else None,
            no_tiled=args.abot_no_tiled,
        )

    raise ValueError(f"Unsupported model: {args.model}")


def extract_frames(video_path: str, frames_dir: str):
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
    if idx == 0:
        raise RuntimeError(f"No frames extracted from {video_path}")


def run_abot_physworld_batch(tasks, gen, args):
    from pathlib import Path
    import shutil
    import shlex
    import math
    import traceback

    pending = []

    for idx, item in enumerate(tasks):
        task_name = item.get("task_name")
        episode_name = item.get("episode_name")

        if not task_name or not episode_name:
            gt_path = Path(item["gt_path"])
            # data/<dataset>/gt_data/<task>/<episode>/video.mp4
            task_name = gt_path.parts[-3]
            episode_name = gt_path.parts[-2]

        out_dir = Path(args.pred_root) / task_name / episode_name / "1"
        out_dir.mkdir(parents=True, exist_ok=True)

        final_mp4 = out_dir / f"{task_name}_{episode_name}.mp4"

        if final_mp4.exists():
            print(f"[Skip] {task_name}/{episode_name} -> exists", flush=True)
            continue

        prompt = resolve_prompt(item, args.prompt_key)

        image = str(Path(item["image"]).expanduser().resolve())

        pending.append(
            {
                "task_name": task_name,
                "episode_name": episode_name,
                "image": image,
                "prompt": prompt,
                "unique_id": f"{task_name}__{episode_name}__{idx:06d}",
                "final_output_path": str(final_mp4),
            }
        )

        # 复制 prompt 文件夹，保持输出结构一致
        src_prompt_dir = Path(item["image"]).expanduser().resolve().parent
        dst_prompt_dir = out_dir / "prompt"
        if src_prompt_dir.exists() and not dst_prompt_dir.exists():
            try:
                shutil.copytree(src_prompt_dir, dst_prompt_dir)
            except Exception:
                pass

    total = len(pending)
    if total == 0:
        print("[ABot batch] nothing to run", flush=True)
        return

    batch_size = max(1, int(getattr(args, "abot_batch_size", 1)))
    num_chunks = math.ceil(total / batch_size)

    print(f"[ABot batch] pending samples: {total}", flush=True)
    print(f"[ABot batch] batch_size={batch_size}, num_chunks={num_chunks}", flush=True)

    success_total = 0
    fail_total = 0

    for chunk_idx, start in enumerate(range(0, total, batch_size), start=1):
        chunk = pending[start:start + batch_size]
        left = start + 1
        right = start + len(chunk)

        print(
            f"[ABot batch] START chunk {chunk_idx}/{num_chunks} | "
            f"samples {left}-{right}/{total}",
            flush=True
        )

        try:
            gen.generate_many(chunk, seed=args.seed)
            for s in pending:
                final_mp4 = Path(s["final_output_path"])
                attempt_dir = final_mp4.parent
                frames_dir = attempt_dir / "video"

                if final_mp4.exists() and (not frames_dir.exists() or not any(frames_dir.iterdir())):
                    try:
                        extract_frames(str(final_mp4), str(frames_dir))
                        print(f"[ABot batch] extracted frames for {final_mp4}", flush=True)
                    except Exception as e:
                        print(f"[ABot batch] failed to extract frames for {final_mp4}: {e}", flush=True)
        except Exception as e:
            print(
                f"[ABot batch] CHUNK FAILED {chunk_idx}/{num_chunks} | "
                f"samples {left}-{right}/{total} | err={e}",
                flush=True
            )

            # 如果一整个 chunk 失败：
            # - 若 batch_size=1，直接标记这条失败
            # - 若 batch_size>1，自动降级逐条重试，避免一条坏样本拖死这一批
            if len(chunk) == 1:
                s = chunk[0]
                fail_log = s["final_output_path"] + ".failed.txt"
                with open(fail_log, "w", encoding="utf-8") as ff:
                    ff.write(f"ABot generation failed:\n{repr(e)}\n")
                    ff.write(traceback.format_exc())
                fail_total += 1
                print(
                    f"[ABot batch] FAIL {s['task_name']}/{s['episode_name']}",
                    flush=True
                )
            else:
                print(
                    f"[ABot batch] fallback to per-sample retry for chunk {chunk_idx}",
                    flush=True
                )
                for s in chunk:
                    try:
                        print(
                            f"[ABot batch] RETRY single {s['task_name']}/{s['episode_name']}",
                            flush=True
                        )
                        gen.generate_many([s], seed=args.seed)
                        if Path(s["final_output_path"]).exists():
                            success_total += 1
                            print(
                                f"[ABot batch] OK {s['task_name']}/{s['episode_name']}",
                                flush=True
                            )
                        else:
                            fail_log = s["final_output_path"] + ".failed.txt"
                            with open(fail_log, "w", encoding="utf-8") as ff:
                                ff.write("ABot single retry finished but no final mp4 found.\n")
                            fail_total += 1
                            print(
                                f"[ABot batch] FAIL(no mp4) {s['task_name']}/{s['episode_name']}",
                                flush=True
                            )
                    except Exception as ee:
                        fail_log = s["final_output_path"] + ".failed.txt"
                        with open(fail_log, "w", encoding="utf-8") as ff:
                            ff.write(f"ABot single retry failed:\n{repr(ee)}\n")
                            ff.write(traceback.format_exc())
                        fail_total += 1
                        print(
                            f"[ABot batch] FAIL {s['task_name']}/{s['episode_name']} | err={ee}",
                            flush=True
                        )
            print(
                f"[ABot batch] PROGRESS after failed chunk: "
                f"success={success_total}, failed={fail_total}, done={success_total + fail_total}/{total}",
                flush=True
            )
            continue

        # chunk 正常返回后，统计这批实际成功多少
        chunk_success = 0
        chunk_fail = 0
        for s in chunk:
            if Path(s["final_output_path"]).exists():
                chunk_success += 1
                print(
                    f"[ABot batch] OK {s['task_name']}/{s['episode_name']}",
                    flush=True
                )
            else:
                chunk_fail += 1
                fail_log = s["final_output_path"] + ".failed.txt"
                with open(fail_log, "w", encoding="utf-8") as ff:
                    ff.write("ABot batch call returned, but no final mp4 was found.\n")
                print(
                    f"[ABot batch] FAIL(no mp4) {s['task_name']}/{s['episode_name']}",
                    flush=True
                )

        success_total += chunk_success
        fail_total += chunk_fail

        print(
            f"[ABot batch] END chunk {chunk_idx}/{num_chunks} | "
            f"chunk_success={chunk_success}, chunk_fail={chunk_fail} | "
            f"total_success={success_total}, total_fail={fail_total}, "
            f"done={success_total + fail_total}/{total}",
            flush=True
        )

    print(
        f"[ABot batch] ALL DONE | success={success_total}, failed={fail_total}, total={total}",
        flush=True
    )


def save_prompt_bundle(prompt_text: str, img_p: str, prompt_dir: str):
    os.makedirs(prompt_dir, exist_ok=True)
    with open(os.path.join(prompt_dir, "prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt_text)
    if os.path.exists(img_p):
        shutil.copy2(img_p, os.path.join(prompt_dir, "init_frame.png"))


def run_one_task(gen, task_info, args):
    img_p = str(Path(task_info["image"]).expanduser().resolve())
    prompt_text = resolve_prompt(task_info, args.prompt_key)
    task_name, episode_name = resolve_task_episode(task_info)
    pid = os.getpid()
    results = []

    for attempt in range(1, args.n_attempts + 1):
        attempt_dir = os.path.join(args.pred_root, task_name, episode_name, str(attempt))
        prompt_dir = os.path.join(attempt_dir, "prompt")
        frames_dir = os.path.join(attempt_dir, "video")
        final_video_path = os.path.join(attempt_dir, f"{task_name}_{episode_name}.mp4")

        if os.path.exists(final_video_path) and os.path.isdir(frames_dir) and len(os.listdir(frames_dir)) > 0:
            results.append(f"A{attempt}:Skip")
            continue

        os.makedirs(attempt_dir, exist_ok=True)

        try:
            print(f"[PID {pid}] 🚀 {task_name}/{episode_name} attempt {attempt} | model={args.model}", flush=True)
            video_path = gen.generate(
                prompt=prompt_text,
                img_path=img_p,
                output_path=final_video_path,
                seed=args.seed + attempt - 1,
            )

            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Generator returned missing video path: {video_path}")

            save_prompt_bundle(prompt_text, img_p, prompt_dir)
            extract_frames(video_path, frames_dir)
            results.append(f"A{attempt}:OK")
        except Exception as e:
            print(f"[PID {pid}] ❌ {task_name}/{episode_name} attempt {attempt} failed: {e}", flush=True)
            results.append(f"A{attempt}:Fail")

        time.sleep(0.2)

    return f"{task_name}/{episode_name} -> {' '.join(results)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["wow", "cosmos", "cosmos3", "cogvideo", "gigaworld", "wan22", "abot_physworld"])
    parser.add_argument("--gt_root", type=str, required=True)
    parser.add_argument("--pred_root", type=str, required=True)
    parser.add_argument("--n_attempts", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_folder", type=str, required=True)
    parser.add_argument("--max_tasks", type=int, default=-1, help="For smoke tests: only run the first N sharded tasks if > 0.")
    parser.add_argument("--sample_keep_prob", type=float, default=1.0,
                        help="Probability of keeping a sample for generation. Use 0.4 to generate 40%% and skip 60%%.")
    parser.add_argument("--sample_seed", type=int, default=20260609,
                        help="Deterministic sampling seed. Same seed/model/sample gives the same keep/skip decision.")
    parser.add_argument("--sample_model_key", type=str, default="",
                        help="Stable model identifier used for sampling. Use the same value for prefix/rewrite if they should share the same sampled subset.")
    parser.add_argument(
        "--prompt_key",
        type=str,
        default="prompt",
        choices=["prompt", "prompt_prefix", "prompt_rewrite"],
        help="Which prompt field in summary.json to use for generation.",
    )

    parser.add_argument("--wow_root", type=str, default="")
    parser.add_argument("--custom_checkpoint", type=str, default="WoW_video_dit.pt")
    parser.add_argument("--persistent_param_gb", type=int, default=70)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--num_frames", type=int, default=41)
    parser.add_argument("--tiled", action="store_true")

    parser.add_argument("--cosmos_root", type=str, default="")
    parser.add_argument("--cosmos_model", type=str, default="2B/post-trained")
    parser.add_argument("--cosmos_inference_type", type=str, default="image2world")
    parser.add_argument("--cosmos_num_frames", type=int, default=-1)
    parser.add_argument("--cosmos_steps", type=int, default=-1)

    parser.add_argument("--cosmos3_root", type=str, default="")
    parser.add_argument("--cosmos3_checkpoint", type=str, default="Cosmos3-Nano",
                        help="Cosmos3-Nano or Cosmos3-Super")
    parser.add_argument("--cosmos3_num_gpus", type=int, default=1,
                        help="Number of GPUs for Cosmos3 (1 for Nano, 4-8 for Super)")
    parser.add_argument("--cosmos3_parallelism", type=str, default="throughput",
                        choices=["throughput", "latency"])
    parser.add_argument("--cosmos3_extra_args", type=str, default="",
                        help="Extra raw args passed to cosmos_framework.scripts.inference")

    parser.add_argument("--cogvideo_steps", type=int, default=50)
    parser.add_argument("--cogvideo_num_frames", type=int, default=81)
    parser.add_argument("--cogvideo_guidance_scale", type=float, default=6.0)
    parser.add_argument("--cogvideo_fps", type=int, default=16)
    parser.add_argument("--cogvideo_model_cpu_offload", action="store_true")
    parser.add_argument("--cogvideo_disable_vae_tiling", action="store_true")
    parser.add_argument("--cogvideo_disable_vae_slicing", action="store_true")
    parser.add_argument("--cogvideo_width", type=int, default=-1)
    parser.add_argument("--cogvideo_height", type=int, default=-1)

    parser.add_argument("--gigaworld_root", type=str, default="")
    parser.add_argument("--gigaworld_extra_args", type=str, default="", help="Extra raw args passed to GigaWorld scripts/inference.py.")

    # Wan2.2 local inference options. Defaults target Wan2.2-I2V-A14B on one H100.
    parser.add_argument("--wan_root", type=str, default="")
    parser.add_argument("--wan_gpu_ids", type=str, default="", help="Optional CUDA_VISIBLE_DEVICES string, e.g. '0' or '0,1,2,3'. Overrides --gpu.")
    parser.add_argument("--wan_task", type=str, default="i2v-A14B")
    parser.add_argument("--wan_size", type=str, default="1280*720")
    parser.add_argument("--wan_python_bin", type=str, default="")
    parser.add_argument("--wan_use_torchrun", action="store_true")
    parser.add_argument("--wan_torchrun_bin", type=str, default="torchrun")
    parser.add_argument("--wan_nproc_per_node", type=int, default=1)
    parser.add_argument("--wan_t5_fsdp", action="store_true")
    parser.add_argument("--wan_dit_fsdp", action="store_true")
    parser.add_argument("--wan_ulysses_size", type=int, default=1)
    parser.add_argument("--wan_offload_model", action="store_true")
    parser.add_argument("--wan_convert_model_dtype", action="store_true")
    parser.add_argument("--wan_t5_cpu", action="store_true")
    parser.add_argument("--wan_frame_num", type=int, default=-1)
    parser.add_argument("--wan_sample_solver", type=str, default="")
    parser.add_argument("--wan_sample_steps", type=int, default=-1)
    parser.add_argument("--wan_sample_shift", type=float, default=-1.0)
    parser.add_argument("--wan_sample_guide_scale", type=float, default=-1.0)
    parser.add_argument("--wan_use_prompt_extend", action="store_true")
    parser.add_argument("--wan_prompt_extend_method", type=str, default="")
    parser.add_argument("--wan_prompt_extend_model", type=str, default="")
    parser.add_argument("--wan_prompt_extend_target_lang", type=str, default="")
    parser.add_argument("--wan_extra_args", type=str, default="", help="Extra raw args passed to Wan2.2 generate.py, parsed by shlex.")

    parser.add_argument("--abot_root", type=str, default="")
    parser.add_argument("--abot_checkpoint_path", type=str, default="")
    parser.add_argument("--abot_height", type=int, default=480)
    parser.add_argument("--abot_width", type=int, default=832)
    parser.add_argument("--abot_num_frames", type=int, default=81)
    parser.add_argument("--abot_num_inference_steps", type=int, default=50)
    parser.add_argument("--abot_cfg_scale", type=float, default=5.0)
    parser.add_argument("--abot_cache_dir", type=str, default="")
    parser.add_argument("--abot_no_tiled", action="store_true")
    parser.add_argument("--abot_batch_size", type=int, default=1,
                    help="How many samples to process per ABot batch call. "
                         "1 means save each sample immediately; larger values reduce init overhead.")

    args = parser.parse_args()

    json_path = os.path.join(args.gt_root, "summary.json") if os.path.isdir(args.gt_root) else args.gt_root
    tasks = get_tasks_from_json(json_path)
    dataset_name = resolve_dataset_name(args.gt_root)

    shard = split_tasks(tasks, args.rank, args.world_size)
    shard_before_sampling = len(shard)

    shard, sampling_skipped = filter_tasks_by_sampling(shard, args, dataset_name)

    if args.max_tasks > 0:
        shard = shard[:args.max_tasks]

    os.makedirs("logs", exist_ok=True)

    # Record sampled-out items for traceability. This is not required for resume;
    # deterministic hash sampling is enough, but the log helps auditing.
    if args.sample_keep_prob < 1.0:
        sample_log = os.path.join(
            "logs",
            f"sampling_{dataset_name}_{args.model}_{args.sample_model_key.strip() or args.model}_"
            f"seed{args.sample_seed}_p{args.sample_keep_prob}_rank{args.rank}.jsonl"
        )
        with open(sample_log, "a", encoding="utf-8") as f:
            for rec in sampling_skipped:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[Sampling] before={shard_before_sampling}, kept={len(shard)}, skipped={len(sampling_skipped)}, log={sample_log}", flush=True)
    print("=" * 80)
    print(f"Model      : {args.model}")
    print(f"JSON       : {json_path}")
    print(f"Pred root  : {args.pred_root}")
    print(f"GPU        : {args.gpu}")
    print(f"Rank       : {args.rank}/{args.world_size}")
    print(f"Num tasks  : {len(shard)} / total {len(tasks)}")
    print(f"Sampling   : keep_prob={args.sample_keep_prob}, seed={args.sample_seed}, model_key={args.sample_model_key.strip() or args.model}")
    if args.model == "wan22":
        print(f"Wan root   : {args.wan_root}")
        print(f"Wan task   : {args.wan_task}")
        print(f"Wan size   : {args.wan_size}")
        print(f"Wan ckpt   : {args.checkpoint_folder}")
    print("=" * 80)

    if len(shard) == 0:
        print("[Sampling] No tasks kept for this rank; exit before model initialization.", flush=True)
        return

    gen = build_generator(args)

    if args.model == "abot_physworld":
        run_abot_physworld_batch(shard, gen, args)
        with open("logs/inference_detail.log", "a", encoding="utf-8") as f:
            f.write(
                f"[{time.ctime()}] rank={args.rank} gpu={args.gpu} "
                f"ABot batch mode finished: pending/generated under {args.pred_root}\n"
            )
    else:
        for idx, task_info in enumerate(shard, start=1):
            info = run_one_task(gen, task_info, args)
            print(f"[{idx}/{len(shard)}] {info}", flush=True)
            with open("logs/inference_detail.log", "a", encoding="utf-8") as f:
                f.write(f"[{time.ctime()}] rank={args.rank} gpu={args.gpu} {info}\n")


if __name__ == "__main__":
    main()
