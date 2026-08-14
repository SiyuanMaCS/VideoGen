
import os

from .base_generator import BaseVideoGenerator

# Official demo negative prompt (i2v)
DEFAULT_NEG = ("Bright tones, overexposed, static, blurred details, subtitles, style, works, "
               "paintings, images, static, overall gray, worst quality, low quality, JPEG compression "
               "residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, "
               "deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, "
               "three legs, many people in the background, walking backwards")


class LongCatGenerator(BaseVideoGenerator):
    def __init__(self, checkpoint_dir, gpu=0, resolution="480p", num_frames=93,
                 num_inference_steps=50, guidance_scale=4.0, fps=15, negative_prompt=None):
        import torch
        import torch.distributed as dist
        from transformers import AutoTokenizer, UMT5EncoderModel
        from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
        from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
        from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
        from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
        from longcat_video.context_parallel import context_parallel_util
        from longcat_video.context_parallel.context_parallel_util import init_context_parallel

        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", str(gpu))
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        self.device = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.device)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        init_context_parallel(context_parallel_size=1, global_rank=0, world_size=1)
        cp_size = context_parallel_util.get_cp_size()
        cp_split_hw = context_parallel_util.get_optimal_split(cp_size)

        self.resolution = resolution
        self.num_frames = num_frames
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.fps = fps
        self.negative_prompt = negative_prompt if negative_prompt is not None else DEFAULT_NEG

        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, subfolder="tokenizer", torch_dtype=torch.bfloat16)
        text_encoder = UMT5EncoderModel.from_pretrained(checkpoint_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16)
        vae = AutoencoderKLWan.from_pretrained(checkpoint_dir, subfolder="vae", torch_dtype=torch.bfloat16)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_dir, subfolder="scheduler", torch_dtype=torch.bfloat16)
        dit = LongCatVideoTransformer3DModel.from_pretrained(checkpoint_dir, subfolder="dit",
                                                             cp_split_hw=cp_split_hw, torch_dtype=torch.bfloat16)
        self.pipe = LongCatVideoPipeline(tokenizer=tokenizer, text_encoder=text_encoder, vae=vae,
                                         scheduler=scheduler, dit=dit)
        self.pipe.to(self.device)

    def generate(self, prompt, img_path, output_path, seed=42, **kwargs) -> str:
        import numpy as np, torch, PIL.Image
        from torchvision.io import write_video

        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        image = PIL.Image.open(img_path).convert("RGB")
        target_size = image.size   # resize output back to the init-frame size
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)

        out = self.pipe.generate_i2v(
            image=image, prompt=prompt, negative_prompt=self.negative_prompt,
            resolution=self.resolution, num_frames=self.num_frames,
            num_inference_steps=self.num_inference_steps, guidance_scale=self.guidance_scale,
            generator=gen)[0]   # numpy (T,H,W,3) in [0,1]

        pil = [PIL.Image.fromarray((out[k] * 255).astype(np.uint8)) for k in range(out.shape[0])]
        pil = [f.resize(target_size, PIL.Image.BICUBIC) for f in pil]
        arr = torch.from_numpy(np.array(pil))   # (T,H,W,3) uint8
        write_video(output_path, arr, fps=self.fps, video_codec="libx264", options={"crf": "18"})
        return output_path
