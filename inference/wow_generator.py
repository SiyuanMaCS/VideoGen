# /inspire/hdd/project/socialsimulation/dongshuai-FWXS252025/atomic/AtomicBench-Evaluation/inference/wow_generator.py

import os
import sys
import tempfile
from pathlib import Path
from PIL import Image

from .base_generator import BaseVideoGenerator


class WoWGenerator(BaseVideoGenerator):
    def __init__(
        self,
        wow_root: str,
        checkpoint_folder: str,
        custom_checkpoint: str = "WoW_video_dit.pt",
        gpu: int = 0,
        persistent_param_gb: int = 20,
        steps: int = 20,
        num_frames: int = 17,
        tiled: bool = True,
        seed: int = 42,
        negative_prompt: str = "low quality, distorted, ugly, bad anatomy",
    ):
        self.wow_root = os.path.abspath(wow_root)
        self.checkpoint_folder = os.path.abspath(checkpoint_folder)
        self.custom_checkpoint = custom_checkpoint
        self.gpu = gpu
        self.persistent_param_gb = persistent_param_gb
        self.steps = steps
        self.num_frames = num_frames
        self.tiled = tiled
        self.seed = seed
        self.negative_prompt = negative_prompt

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # 让 Python 能 import WoW demo 里的 build_pipeline
        demo_dir = os.path.join(self.wow_root, "demo")
        if demo_dir not in sys.path:
            sys.path.insert(0, demo_dir)

        from wan_infer_demo import build_pipeline
        from diffsynth import save_video

        self._save_video = save_video
        self.pipe = build_pipeline(
            gpu_id=self.gpu,
            checkpoint_folder=self.checkpoint_folder,
            custom_checkpoint_name=self.custom_checkpoint,
            enable_vram_management=True,
            persistent_param_gb=self.persistent_param_gb,
        )

    def _load_image(self, img_path: str):
        return Image.open(img_path).convert("RGB")

    def generate(self, prompt, img_path, output_path=None, seed=None):
        input_image = self._load_image(img_path)

        actual_seed = self.seed if seed is None else seed
        video = self.pipe(
            prompt=prompt,
            negative_prompt=self.negative_prompt,
            input_image=input_image,
            num_inference_steps=self.steps,
            seed=actual_seed,
            tiled=self.tiled,
            num_frames=self.num_frames,
        )

        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix=".mp4", prefix="wow_")
            os.close(fd)

        output_path = os.path.abspath(output_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        self._save_video(video, output_path, fps=15, quality=5)
        return output_path