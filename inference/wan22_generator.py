"""Wan2.2 local generator wrapper for AtomicBench inference.

This wrapper calls the official Wan2.2 `generate.py` script through subprocess,
so it does not import Wan2.2 modules inside the benchmark runner. This keeps the
runner lightweight and makes environment/debugging easier.
"""

import os
import shlex
import sys
import subprocess
from pathlib import Path
from typing import Optional


class Wan22Generator:
    def __init__(
        self,
        wan_root: str,
        checkpoint_dir: str,
        gpu: int = 0,
        gpu_ids: str = "",
        task: str = "i2v-A14B",
        size: str = "1280*720",
        python_bin: str = "",
        use_torchrun: bool = False,
        torchrun_bin: str = "torchrun",
        nproc_per_node: int = 1,
        t5_fsdp: bool = False,
        dit_fsdp: bool = False,
        ulysses_size: int = 1,
        offload_model: bool = False,
        convert_model_dtype: bool = False,
        t5_cpu: bool = False,
        frame_num: int = -1,
        sample_solver: str = "",
        sample_steps: int = -1,
        sample_shift: float = -1.0,
        sample_guide_scale: float = -1.0,
        use_prompt_extend: bool = False,
        prompt_extend_method: str = "",
        prompt_extend_model: str = "",
        prompt_extend_target_lang: str = "",
        extra_args: str = "",
    ):
        self.wan_root = Path(wan_root).expanduser().resolve()
        self.checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        self.gpu = gpu
        self.gpu_ids = gpu_ids.strip()
        self.task = task
        self.size = size
        self.python_bin = python_bin.strip() or sys.executable
        self.use_torchrun = use_torchrun
        self.torchrun_bin = torchrun_bin
        self.nproc_per_node = int(nproc_per_node)
        self.t5_fsdp = t5_fsdp
        self.dit_fsdp = dit_fsdp
        self.ulysses_size = int(ulysses_size)
        self.offload_model = offload_model
        self.convert_model_dtype = convert_model_dtype
        self.t5_cpu = t5_cpu
        self.frame_num = int(frame_num)
        self.sample_solver = sample_solver.strip()
        self.sample_steps = int(sample_steps)
        self.sample_shift = float(sample_shift)
        self.sample_guide_scale = float(sample_guide_scale)
        self.use_prompt_extend = use_prompt_extend
        self.prompt_extend_method = prompt_extend_method.strip()
        self.prompt_extend_model = prompt_extend_model.strip()
        self.prompt_extend_target_lang = prompt_extend_target_lang.strip()
        self.extra_args = extra_args.strip()

        self._validate_static_paths()

    def _validate_static_paths(self):
        if not self.wan_root.exists():
            raise FileNotFoundError(f"Wan2.2 repo not found: {self.wan_root}")
        generate_py = self.wan_root / "generate.py"
        if not generate_py.exists():
            raise FileNotFoundError(f"generate.py not found under Wan2.2 repo: {generate_py}")
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"Wan2.2 checkpoint dir not found: {self.checkpoint_dir}")

    def _build_command(self, prompt: str, img_path: str, output_path: str, seed: int):
        if self.use_torchrun:
            cmd = [
                self.torchrun_bin,
                "--nproc_per_node",
                str(self.nproc_per_node),
                "generate.py",
            ]
        else:
            cmd = [self.python_bin, "generate.py"]

        cmd += [
            "--task", self.task,
            "--size", self.size,
            "--ckpt_dir", str(self.checkpoint_dir),
            "--image", str(img_path),
            "--prompt", prompt if prompt is not None else "",
            "--save_file", str(output_path),
            "--base_seed", str(seed),
        ]

        if self.offload_model:
            cmd += ["--offload_model", "True"]
        if self.convert_model_dtype:
            cmd += ["--convert_model_dtype"]
        if self.t5_cpu:
            cmd += ["--t5_cpu"]
        if self.t5_fsdp:
            cmd += ["--t5_fsdp"]
        if self.dit_fsdp:
            cmd += ["--dit_fsdp"]
        if self.ulysses_size > 1:
            cmd += ["--ulysses_size", str(self.ulysses_size)]
        if self.frame_num > 0:
            cmd += ["--frame_num", str(self.frame_num)]
        if self.sample_solver:
            cmd += ["--sample_solver", self.sample_solver]
        if self.sample_steps > 0:
            cmd += ["--sample_steps", str(self.sample_steps)]
        if self.sample_shift >= 0:
            cmd += ["--sample_shift", str(self.sample_shift)]
        if self.sample_guide_scale >= 0:
            cmd += ["--sample_guide_scale", str(self.sample_guide_scale)]
        if self.use_prompt_extend:
            cmd += ["--use_prompt_extend"]
        if self.prompt_extend_method:
            cmd += ["--prompt_extend_method", self.prompt_extend_method]
        if self.prompt_extend_model:
            cmd += ["--prompt_extend_model", self.prompt_extend_model]
        if self.prompt_extend_target_lang:
            cmd += ["--prompt_extend_target_lang", self.prompt_extend_target_lang]
        if self.extra_args:
            cmd += shlex.split(self.extra_args)

        return cmd

    def generate(self, prompt: str, img_path: str, output_path: str, seed: int = 42):
        img_path = Path(img_path).expanduser().resolve()
        output_path = Path(output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not img_path.exists():
            raise FileNotFoundError(f"Input image not found: {img_path}")

        cmd = self._build_command(prompt=prompt, img_path=str(img_path), output_path=str(output_path), seed=seed)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.gpu_ids if self.gpu_ids else str(self.gpu)
        old_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(self.wan_root) + (os.pathsep + old_pythonpath if old_pythonpath else "")

        log_path = output_path.with_suffix(output_path.suffix + ".wan22.log")
        print(f"[Wan2.2] cwd={self.wan_root}", flush=True)
        print(f"[Wan2.2] CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}", flush=True)
        print("[Wan2.2] cmd=" + " ".join(shlex.quote(x) for x in cmd), flush=True)
        print(f"[Wan2.2] log={log_path}", flush=True)

        with open(log_path, "w", encoding="utf-8") as log_f:
            log_f.write("CMD: " + " ".join(shlex.quote(x) for x in cmd) + "\n")
            log_f.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.wan_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                log_f.write(line)
                log_f.flush()
            ret = proc.wait()

        if ret != 0:
            raise RuntimeError(f"Wan2.2 generation failed with return code {ret}. See log: {log_path}")
        if not output_path.exists():
            raise FileNotFoundError(f"Wan2.2 finished but output video is missing: {output_path}")
        if output_path.stat().st_size == 0:
            raise RuntimeError(f"Wan2.2 output video is empty: {output_path}")

        return str(output_path)
