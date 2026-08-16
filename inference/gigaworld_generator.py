import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .base_generator import BaseVideoGenerator


class GigaWorldGenerator(BaseVideoGenerator):
    """Thin wrapper around official giga-world-0/scripts/inference.py."""

    def __init__(
        self,
        gigaworld_root: str,
        checkpoint_dir: str,
        gpu: int = 0,
        seed: int = 42,
        extra_infer_args=None,
    ):
        self.gigaworld_root = os.path.abspath(gigaworld_root)
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        self.gpu = gpu
        self.seed = seed
        self.extra_infer_args = list(extra_infer_args or [])

        self.infer_script = os.path.join(self.gigaworld_root, "scripts", "inference.py")
        if not os.path.exists(self.infer_script):
            raise FileNotFoundError(f"GigaWorld inference.py not found: {self.infer_script}")

        self.transformer_model_path = os.path.join(self.checkpoint_dir, "transformer")
        self.text_encoder_model_path = os.path.join(self.checkpoint_dir, "text_encoder")
        self.vae_model_path = os.path.join(self.checkpoint_dir, "vae")
        for label, component_path in (
            ("transformer", self.transformer_model_path),
            ("text encoder", self.text_encoder_model_path),
            ("VAE", self.vae_model_path),
        ):
            if not os.path.isdir(component_path):
                raise FileNotFoundError(f"GigaWorld {label} directory not found: {component_path}")

    def _find_generated_mp4(self, out_dir: str) -> str:
        mp4s = sorted(Path(out_dir).rglob("*.mp4"), key=lambda p: p.stat().st_size, reverse=True)
        if not mp4s:
            raise FileNotFoundError(f"No generated mp4 found under: {out_dir}")
        return str(mp4s[0])

    def _extract_generated_panel(self, source_path: str, output_path: str) -> None:
        """Remove the conditioning-image panel added by upstream inference.py."""
        import cv2

        capture = cv2.VideoCapture(source_path)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open GigaWorld visualization video: {source_path}")
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS) or 16.0
        generated_width = (frame_width - 2) // 2
        if generated_width <= 0 or frame_height <= 0:
            capture.release()
            raise ValueError(f"Unexpected GigaWorld video size: {frame_width}x{frame_height}")

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (generated_width, frame_height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Cannot create cropped GigaWorld video: {output_path}")

        frame_count = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                # Upstream uses concat_images_grid(..., cols=2, pad=2) with
                # [conditioning_image, generated_image]. Keep the right panel.
                writer.write(frame[:, -generated_width:])
                frame_count += 1
        finally:
            capture.release()
            writer.release()

        if frame_count == 0 or not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError(f"Failed to extract generated GigaWorld video: {source_path}")

    def generate(self, prompt: str, img_path: str, output_path: str, seed: int = 42, **kwargs) -> str:
        output_path = os.path.abspath(output_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        img_path = os.path.abspath(img_path)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"GigaWorld conditioning image not found: {img_path}")

        env = os.environ.copy()
        env["PYTHONPATH"] = self.gigaworld_root + os.pathsep + env.get("PYTHONPATH", "")

        with tempfile.TemporaryDirectory(prefix="gigaworld_worldjudge_") as tmpdir:
            data_json = os.path.join(tmpdir, "infer.json")
            out_dir = os.path.join(tmpdir, "vis_results")
            os.makedirs(out_dir, exist_ok=True)

            payload = [{
                "name": Path(output_path).stem,
                "prompt": prompt,
                "image": img_path,
            }]
            with open(data_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            cmd = [
                sys.executable,
                self.infer_script,
                "--data-path", data_json,
                "--save-dir", out_dir,
                "--transformer-model-path", self.transformer_model_path,
                "--text-encoder-model-path", self.text_encoder_model_path,
                "--vae-model-path", self.vae_model_path,
                "--gpu-ids", str(self.gpu),
                "--seed", str(self.seed if seed is None else seed),
            ]
            cmd.extend(self.extra_infer_args)

            proc = subprocess.run(
                cmd,
                cwd=self.gigaworld_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    "GigaWorld inference failed.\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"Output:\n{proc.stdout}"
                )

            generated_mp4 = self._find_generated_mp4(out_dir)
            self._extract_generated_panel(generated_mp4, output_path)
            return output_path
