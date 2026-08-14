
import os

from .base_generator import BaseVideoGenerator


def _build_hunyuan_args(model_path, resolution, num_inference_steps, video_length,
                        enable_step_distill, offloading, aspect_ratio, negative_prompt):
    
    import types
    return types.SimpleNamespace(
        prompt="", negative_prompt=negative_prompt, resolution=resolution,
        model_path=model_path, aspect_ratio=aspect_ratio,
        num_inference_steps=num_inference_steps, video_length=video_length,
        sr=False, save_pre_sr_video=False, rewrite=False,
        cfg_distilled=False, enable_step_distill=enable_step_distill, sparse_attn=False,
        offloading=offloading, group_offloading=None, overlap_group_offloading=True,
        dtype="bf16", seed=123, image_path=None, output_path=None,
        use_sageattn=False, sage_blocks_range="0-53",
        enable_torch_compile=False, enable_cache=False, cache_type="deepcache",
        no_cache_block_id="53", cache_start_step=11, cache_end_step=45,
        total_steps=50, cache_step_interval=4, save_generation_config=False,
        checkpoint_path=None, lora_path=None,
        use_fp8_gemm=False, quant_type="fp8-per-token-sgl", include_patterns="double_blocks",
    )


class HunyuanGenerator(BaseVideoGenerator):
    def __init__(self, checkpoint_dir, gpu=0, resolution="480p", num_inference_steps=12,
                 guidance_scale=1.0, flow_shift=7.0, video_length=121,
                 enable_step_distill=True, offloading=False, aspect_ratio="16:9",
                 negative_prompt="", fps=24):
        import torch

        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", str(gpu))
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline
        from hyvideo.commons.parallel_states import initialize_parallel_state
        from hyvideo.commons.infer_state import initialize_infer_state

        initialize_parallel_state(sp=int(os.environ["WORLD_SIZE"]))
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

        self.fps = fps
        self.guidance_scale = guidance_scale
        self.flow_shift = flow_shift
        self.args = _build_hunyuan_args(checkpoint_dir, resolution, num_inference_steps,
                                        video_length, enable_step_distill, offloading,
                                        aspect_ratio, negative_prompt)
        infer_state = initialize_infer_state(self.args)
        transformer_version = HunyuanVideo_1_5_Pipeline.get_transformer_version(
            resolution, "i2v", self.args.cfg_distilled, self.args.enable_step_distill,
            self.args.sparse_attn)
        enable_offloading = self.args.offloading
        off_cfg = HunyuanVideo_1_5_Pipeline.get_offloading_config()
        enable_group_offloading = (off_cfg["enable_group_offloading"]
                                   if self.args.group_offloading is None else self.args.group_offloading)
        device = torch.device("cpu") if enable_offloading else torch.device("cuda")
        transformer_init_device = torch.device("cpu") if enable_group_offloading else device

        self.pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
            pretrained_model_name_or_path=checkpoint_dir,
            transformer_version=transformer_version, create_sr_pipeline=False,
            transformer_dtype=torch.bfloat16, device=device,
            transformer_init_device=transformer_init_device)
        self.pipe.apply_infer_optimization(
            infer_state=infer_state, enable_offloading=enable_offloading,
            enable_group_offloading=enable_group_offloading,
            overlap_group_offloading=self.args.overlap_group_offloading)

    def generate(self, prompt, img_path, output_path, seed=42, **kwargs) -> str:
        import torch, einops, imageio

        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        out = self.pipe(
            enable_sr=False, prompt=prompt, aspect_ratio=self.args.aspect_ratio,
            num_inference_steps=self.args.num_inference_steps, sr_num_inference_steps=None,
            video_length=self.args.video_length, negative_prompt=self.args.negative_prompt,
            guidance_scale=self.guidance_scale, flow_shift=self.flow_shift,
            seed=seed, output_type="pt", prompt_rewrite=False,
            return_pre_sr_video=False, reference_image=img_path)

        video = out.videos
        if video.ndim == 5:
            video = video[0]
        vid = (video * 255).clamp(0, 255).to(torch.uint8)
        vid = einops.rearrange(vid, "c f h w -> f h w c").cpu().numpy()
        imageio.mimwrite(output_path, vid, fps=self.fps)
        return output_path
