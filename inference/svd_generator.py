
import os

from .base_generator import BaseVideoGenerator


class SVDGenerator(BaseVideoGenerator):
    def __init__(
        self,
        checkpoint_dir: str,
        gpu: int = 0,
        num_frames: int = 25,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: int = 8,
        width: int = 1024,
        height: int = 576,
        variant: str = "fp16",
    ):
        import torch
        from diffusers import StableVideoDiffusionPipeline

        self.device = f"cuda:{gpu}"
        self.num_frames = num_frames
        self.fps = fps
        self.motion_bucket_id = motion_bucket_id
        self.noise_aug_strength = noise_aug_strength
        self.decode_chunk_size = decode_chunk_size
        self.width = width
        self.height = height

        # checkpoint_dir = a diffusers SVD dir (unet/vae/image_encoder/scheduler + model_index.json)
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            checkpoint_dir, torch_dtype=torch.float16, variant=variant)
        self.pipe.to(self.device)

    def generate(self, prompt: str, img_path: str, output_path: str, seed: int = 42, **kwargs) -> str:
        import torch
        import PIL.Image
        from diffusers.utils import export_to_video

        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # SVD ignores `prompt` — conditions on the init frame only
        image = PIL.Image.open(img_path).convert("RGB").resize((self.width, self.height))
        generator = torch.manual_seed(seed)
        frames = self.pipe(
            image,
            decode_chunk_size=self.decode_chunk_size,
            generator=generator,
            num_frames=self.num_frames,
            motion_bucket_id=self.motion_bucket_id,
            noise_aug_strength=self.noise_aug_strength,
            fps=self.fps,
        ).frames[0]
        export_to_video(frames, output_path, fps=self.fps)
        return output_path
