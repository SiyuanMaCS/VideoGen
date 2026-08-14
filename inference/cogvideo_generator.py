import os
from pathlib import Path

import torch
from PIL import Image

from .base_generator import BaseVideoGenerator


class CogVideoGenerator(BaseVideoGenerator):
    def __init__(
        self,
        checkpoint_dir: str,
        gpu: int = 0,
        seed: int = 42,
        num_inference_steps: int = 50,
        num_frames: int = 81,
        guidance_scale: float = 6.0,
        fps: int = 16,
        use_model_cpu_offload: bool = False,
        use_vae_tiling: bool = True,
        use_vae_slicing: bool = True,
        width: int | None = None,
        height: int | None = None,
    ):
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        self.gpu = gpu
        self.seed = seed
        self.num_inference_steps = num_inference_steps
        self.num_frames = num_frames
        self.guidance_scale = guidance_scale
        self.fps = fps
        self.use_model_cpu_offload = use_model_cpu_offload
        self.use_vae_tiling = use_vae_tiling
        self.use_vae_slicing = use_vae_slicing
        self.width = width
        self.height = height

        from diffusers import CogVideoXImageToVideoPipeline
        from diffusers.utils import export_to_video

        self._export_to_video = export_to_video
        self.pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            self.checkpoint_dir,
            torch_dtype=torch.bfloat16,
        )

        if self.use_model_cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe = self.pipe.to(f"cuda:{self.gpu}")

        if self.use_vae_tiling:
            self.pipe.vae.enable_tiling()
        if self.use_vae_slicing:
            self.pipe.vae.enable_slicing()

    def _resolve_resolution(self, img_path: str) -> tuple[int, int]:
        if self.width is not None and self.height is not None:
            return self.width, self.height
        with Image.open(img_path) as im:
            w, h = im.size
        if w >= h:
            return 1360, 768
        return 768, 1360

    def generate(self, prompt: str, img_path: str, output_path: str, seed: int = 42, **kwargs) -> str:
        from diffusers.utils import load_image

        output_path = os.path.abspath(output_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        generator = torch.Generator(device="cpu").manual_seed(self.seed if seed is None else seed)
        width, height = self._resolve_resolution(img_path)
        image = load_image(img_path)

        result = self.pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=self.num_inference_steps,
            num_frames=self.num_frames,
            guidance_scale=self.guidance_scale,
            generator=generator,
            width=width,
            height=height,
        )
        frames = result.frames[0]
        self._export_to_video(frames, output_path, fps=self.fps)
        return output_path
