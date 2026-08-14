import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .base_generator import BaseVideoGenerator


class CosmosGenerator(BaseVideoGenerator):
    """Thin wrapper around Cosmos-Predict2.5 official examples/inference.py."""

    def __init__(
        self,
        cosmos_root: str,
        checkpoint_dir: str,
        model: str = "2B/post-trained",
        inference_type: str = "image2world",
        seed: int = 42,
        extra_cli_args=None,
    ):
        self.cosmos_root = os.path.abspath(cosmos_root)
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        self.model = model
        self.inference_type = inference_type
        self.seed = seed
        self.extra_cli_args = list(extra_cli_args or [])

        self.entry_script = os.path.join(self.cosmos_root, "examples", "inference.py")
        if not os.path.exists(self.entry_script):
            raise FileNotFoundError(f"Cosmos entry script not found: {self.entry_script}")

    def _ensure_repo_checkpoint_link(self):
        repo_ckpt = os.path.join(self.cosmos_root, "checkpoints")
        if os.path.islink(repo_ckpt):
            return
        if os.path.exists(repo_ckpt) and not os.path.islink(repo_ckpt):
            return
        try:
            os.symlink(self.checkpoint_dir, repo_ckpt)
        except FileExistsError:
            pass

    def _find_generated_mp4(self, out_dir: str) -> str:
        mp4s = sorted(Path(out_dir).rglob("*.mp4"), key=lambda p: p.stat().st_size, reverse=True)
        if not mp4s:
            raise FileNotFoundError(f"No generated mp4 found under: {out_dir}")
        return str(mp4s[0])

    def generate(self, prompt: str, img_path: str, output_path: str, seed: int = 42, **kwargs) -> str:
        output_path = os.path.abspath(output_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        self._ensure_repo_checkpoint_link()
        actual_seed = self.seed if seed is None else seed

        env = os.environ.copy()
        env["PYTHONPATH"] = self.cosmos_root + os.pathsep + env.get("PYTHONPATH", "")

        with tempfile.TemporaryDirectory(prefix="cosmos_atomicbench_") as tmpdir:
            bundle_path = os.path.join(tmpdir, "sample.json")
            out_dir = os.path.join(tmpdir, "outputs")
            os.makedirs(out_dir, exist_ok=True)

            bundle = {
                "inference_type": self.inference_type,
                "name": Path(output_path).stem,
                "prompt": prompt,
                "input_path": os.path.abspath(img_path),
            }
            with open(bundle_path, "w", encoding="utf-8") as f:
                json.dump(bundle, f, indent=2, ensure_ascii=False)

            cmd = [
                sys.executable,
                self.entry_script,
                "-i", bundle_path,
                "-o", out_dir,
                "--inference-type", self.inference_type,
                "--model", self.model,
                "--seed", str(actual_seed),
            ]
            cmd.extend(self.extra_cli_args)

            proc = subprocess.run(
                cmd,
                cwd=self.cosmos_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    "Cosmos inference failed.\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"Output:\n{proc.stdout}"
                )

            generated_mp4 = self._find_generated_mp4(out_dir)
            shutil.copy2(generated_mp4, output_path)
            return output_path
