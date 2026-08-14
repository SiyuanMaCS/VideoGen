import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import time


class ABotPhysWorldGenerator:
    def __init__(
        self,
        abot_root: str,
        checkpoint_path: str,
        gpu: int = 0,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        cfg_scale: float = 5.0,
        cache_dir: str | None = None,
        no_tiled: bool = False,
    ):
        self.abot_root = abot_root
        self.checkpoint_path = checkpoint_path
        self.gpu = gpu
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.num_inference_steps = num_inference_steps
        self.cfg_scale = cfg_scale
        self.cache_dir = cache_dir
        self.no_tiled = no_tiled

    def _build_cmd(self, jsonl_path: str, output_dir: str, seed: int) -> list[str]:
        cmd = [
            sys.executable,
            os.path.join(self.abot_root, "inference", "inference.py"),
            "--jsonl_path", jsonl_path,
            "--output_dir", output_dir,
            "--checkpoint_path", self.checkpoint_path,
            "--height", str(self.height),
            "--width", str(self.width),
            "--num_frames", str(self.num_frames),
            "--num_inference_steps", str(self.num_inference_steps),
            "--cfg_scale", str(self.cfg_scale),
            "--seed", str(seed),
            "--gpu_id", str(self.gpu),
            "--num_samples", "1000000",
        ]
        if self.cache_dir:
            cmd += ["--cache_dir", self.cache_dir]
        if self.no_tiled:
            cmd += ["--no_tiled"]
        return cmd

    def generate_many(self, samples: list[dict], seed: int = 42) -> None:
        """
        samples: list of dicts with keys:
        - image
        - prompt
        - unique_id
        - final_output_path
        """
        if not samples:
            return

        tmp_dir = tempfile.mkdtemp(prefix="abotpw_batch_")
        jsonl_path = os.path.join(tmp_dir, "samples.jsonl")
        out_dir = os.path.join(tmp_dir, "outputs")
        tmp_inputs_dir = os.path.join(tmp_dir, "inputs")
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(tmp_inputs_dir, exist_ok=True)

        # 1) 先准备唯一输入文件，并真正写出 JSONL
        prepared_samples = []
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for s in samples:
                src_image = str(Path(s["image"]).expanduser().resolve())
                suffix = Path(src_image).suffix if Path(src_image).suffix else ".png"
                unique_input = os.path.join(tmp_inputs_dir, f"{s['unique_id']}{suffix}")

                try:
                    if os.path.lexists(unique_input):
                        os.remove(unique_input)
                    os.symlink(src_image, unique_input)
                except Exception:
                    shutil.copy2(src_image, unique_input)

                row = {
                    "video": unique_input,
                    "prompt": s["prompt"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

                prepared = dict(s)
                prepared["tmp_input"] = unique_input
                prepared_samples.append(prepared)

        # 2) 关键变量必须在这里定义
        cmd = self._build_cmd(jsonl_path=jsonl_path, output_dir=out_dir, seed=seed)
        batch_log = os.path.join(out_dir, "abot_batch.log")
        env = os.environ.copy()

        print(f"[ABot batch] launch subprocess for {len(prepared_samples)} samples", flush=True)
        print(f"[ABot batch] tmp_dir={tmp_dir}", flush=True)

        moved = set()
        success_count = 0
        fail_count = 0
        last_report = time.time()

        with open(batch_log, "w", encoding="utf-8") as logf:
            logf.write("CMD: " + " ".join(cmd) + "\n")
            logf.flush()

            proc = subprocess.Popen(
                cmd,
                cwd=self.abot_root,
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
            )

            while True:
                # ABot 当前行为：每完成一个样本，log 里会多一次 "Saving video: 100%"
                complete_count = 0
                if os.path.exists(batch_log):
                    try:
                        with open(batch_log, "r", encoding="utf-8", errors="ignore") as rf:
                            log_text = rf.read()
                        complete_count = log_text.count("Saving video: 100%")
                    except Exception:
                        complete_count = 0

                # 如果完成数增加了，就按顺序把当前 rolling mp4 拷贝出来
                while success_count < complete_count and success_count < len(prepared_samples):
                    s = prepared_samples[success_count]

                    mp4s = sorted(
                        [
                            os.path.join(out_dir, x)
                            for x in os.listdir(out_dir)
                            if x.endswith(".mp4")
                        ],
                        key=lambda p: os.path.getmtime(p),
                    )

                    if not mp4s:
                        break

                    src_mp4 = mp4s[-1]
                    final_output_path = str(s["final_output_path"])
                    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)

                    # 注意这里必须 copy，不能 move，因为 outputs 里的 rolling 文件后面还会继续被覆盖使用
                    shutil.copy2(src_mp4, final_output_path)

                    per_sample_log = final_output_path + ".abot.log"
                    with open(per_sample_log, "w", encoding="utf-8") as ff:
                        ff.write(f"Batch log: {batch_log}\n")
                        ff.write(f"Temporary batch dir: {tmp_dir}\n")
                        ff.write(f"Copied from rolling output: {src_mp4}\n")
                        ff.write(f"Completion index: {success_count + 1}\n")

                    moved.add(success_count)
                    success_count += 1

                    print(
                        f"[ABot batch] saved {success_count}/{len(prepared_samples)} -> {final_output_path}",
                        flush=True
                    )

                ret = proc.poll()
                now = time.time()

                if now - last_report > 60:
                    print(
                        f"[ABot batch] heartbeat: completed={complete_count}, "
                        f"saved={success_count}/{len(prepared_samples)}, "
                        f"still_running={ret is None}",
                        flush=True
                    )
                    last_report = now

                if ret is not None:
                    break

                time.sleep(2)

        # 3) 子进程结束后，再补收一次
        if os.path.exists(batch_log):
            try:
                with open(batch_log, "r", encoding="utf-8", errors="ignore") as rf:
                    log_text = rf.read()
                complete_count = log_text.count("Saving video: 100%")
            except Exception:
                complete_count = success_count
        else:
            complete_count = success_count

        while success_count < complete_count and success_count < len(prepared_samples):
            s = prepared_samples[success_count]

            mp4s = sorted(
                [
                    os.path.join(out_dir, x)
                    for x in os.listdir(out_dir)
                    if x.endswith(".mp4")
                ],
                key=lambda p: os.path.getmtime(p),
            )

            if not mp4s:
                break

            src_mp4 = mp4s[-1]
            final_output_path = str(s["final_output_path"])
            os.makedirs(os.path.dirname(final_output_path), exist_ok=True)

            shutil.copy2(src_mp4, final_output_path)

            per_sample_log = final_output_path + ".abot.log"
            with open(per_sample_log, "w", encoding="utf-8") as ff:
                ff.write(f"Batch log: {batch_log}\n")
                ff.write(f"Temporary batch dir: {tmp_dir}\n")
                ff.write(f"Copied from rolling output: {src_mp4}\n")
                ff.write(f"Completion index: {success_count + 1}\n")

            moved.add(success_count)
            success_count += 1

            print(
                f"[ABot batch] saved {success_count}/{len(prepared_samples)} -> {final_output_path}",
                flush=True
            )

        # 4) 其余没收成功的样本记成 failed
        results_json = os.path.join(out_dir, "results.json")
        results = []
        if os.path.exists(results_json):
            try:
                with open(results_json, "r", encoding="utf-8") as f:
                    results = json.load(f)
            except Exception:
                results = []

        for idx, s in enumerate(prepared_samples):
            if idx in moved:
                continue

            final_output_path = str(s["final_output_path"])
            fail_log = final_output_path + ".failed.txt"

            with open(fail_log, "w", encoding="utf-8") as ff:
                ff.write(f"ABot did not produce final mp4 for sample index {idx}\n")
                ff.write(f"Batch log: {batch_log}\n")
                ff.write(f"Batch dir: {tmp_dir}\n")
                if isinstance(results, list) and idx < len(results):
                    ff.write("results.json entry:\n")
                    ff.write(json.dumps(results[idx], ensure_ascii=False, indent=2) + "\n")

            fail_count += 1
            print(
                f"[ABot batch] failed {idx+1}/{len(prepared_samples)} -> {final_output_path}",
                flush=True
            )

        print(
            f"[ABot batch] finished: success={success_count}, failed={fail_count}, "
            f"returncode={proc.returncode}",
            flush=True
        )

        if success_count == 0 and proc.returncode != 0:
            raise RuntimeError(
                f"ABot-PhysWorld batch inference failed with return code {proc.returncode}. "
                f"See log: {batch_log}"
            )

    # 兼容旧接口；单样本时仍可用
    def generate(self, prompt: str, img_path: str, output_path: str, seed: int = 42) -> str:
        sample = {
            "image": img_path,
            "prompt": prompt,
            "unique_id": Path(output_path).stem,
            "final_output_path": output_path,
        }
        self.generate_many([sample], seed=seed)
        return output_path
