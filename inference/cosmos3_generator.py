import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .base_generator import BaseVideoGenerator


class Cosmos3Generator(BaseVideoGenerator):
    """Wrapper around cosmos-framework for Cosmos3-Nano / Cosmos3-Super (image2video)."""

    def __init__(
        self,
        cosmos_framework_root: str,
        checkpoint_path: str = "Cosmos3-Nano",
        seed: int = 42,
        num_gpus: int = 1,
        parallelism_preset: str = "throughput",
        python_bin: str = "",
        extra_cli_args=None,
    ):
        self.cosmos_framework_root = os.path.abspath(cosmos_framework_root)
        self.checkpoint_path = checkpoint_path
        self.seed = seed
        self.num_gpus = num_gpus
        self.parallelism_preset = parallelism_preset
        self.python_bin = python_bin or sys.executable
        self.extra_cli_args = list(extra_cli_args or [])

        pyproject = os.path.join(self.cosmos_framework_root, "pyproject.toml")
        if not os.path.exists(pyproject):
            raise FileNotFoundError(
                f"cosmos-framework not found at: {self.cosmos_framework_root}"
            )

    def _find_generated_mp4(self, out_dir: str) -> str:
        mp4s = sorted(
            Path(out_dir).rglob("*.mp4"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not mp4s:
            raise FileNotFoundError(f"No generated mp4 found under: {out_dir}")
        return str(mp4s[0])

    def generate(self, prompt: str, img_path: str, output_path: str, seed: int = None, **kwargs) -> str:
        output_path = os.path.abspath(output_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        actual_seed = seed if seed is not None else self.seed

        with tempfile.TemporaryDirectory(prefix="cosmos3_gen_") as tmpdir:
            sample_json = os.path.join(tmpdir, "sample.json")
            out_dir = os.path.join(tmpdir, "outputs")
            os.makedirs(out_dir, exist_ok=True)

            sample = {
                "model_mode": "image2video",
                "prompt": prompt,
                "vision_path": os.path.abspath(img_path),
            }
            with open(sample_json, "w", encoding="utf-8") as f:
                json.dump(sample, f, indent=2, ensure_ascii=False)

            env = os.environ.copy()
            env["PYTHONPATH"] = (
                self.cosmos_framework_root
                + os.pathsep
                + env.get("PYTHONPATH", "")
            )

            if self.num_gpus > 1:
                cmd = [
                    "torchrun",
                    f"--nproc-per-node={self.num_gpus}",
                    "-m", "cosmos_framework.scripts.inference",
                ]
            else:
                cmd = [self.python_bin, "-m", "cosmos_framework.scripts.inference"]

            cmd += [
                f"--parallelism-preset={self.parallelism_preset}",
                "-i", sample_json,
                "-o", out_dir,
                f"--checkpoint-path={self.checkpoint_path}",
                f"--seed={actual_seed}",
            ]
            cmd.extend(self.extra_cli_args)

            proc = subprocess.run(
                cmd,
                cwd=self.cosmos_framework_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Cosmos3 inference failed (exit {proc.returncode}).\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"Output:\n{proc.stdout[-2000:]}"
                )

            generated_mp4 = self._find_generated_mp4(out_dir)
            shutil.copy2(generated_mp4, output_path)
            return output_path
