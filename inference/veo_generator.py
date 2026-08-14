import time
import os
from google import genai
from google.genai import types

class VeoVideoGenerator:
    """
    Google Veo 3.1 极简生成器
    默认：720p, 横屏 (16:9), 5s 时长
    """
    
    def __init__(self, api_key=None, model_name="veo-3.1-lite-generate-preview"):
        """
        可选模型列表:
        1. "veo-3.1-lite-generate-preview" (For debug)
        2. "veo-3.1-fast-generate-preview" 
        3. "veo-3.1-generate-preview"      (For Evaluation)
        """
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt, img_path=None, duration=5, seed=None, **kwargs):
        """
        执行生成逻辑. seed 可选——传入则固定/指定该视频的随机种子。
        """
        print(f"🚀 [Veo 3.1] Task submitted to: {self.model_name}")

        # 按照需求：固定 720p，固定 16:9，指定时长
        cfg_kwargs = dict(
            aspect_ratio="16:9",
            resolution="720p",
            duration_seconds=duration,
        )
        if seed is not None:
            cfg_kwargs["seed"] = int(seed)
        config = types.GenerateVideosConfig(**cfg_kwargs)

        # 处理首帧图输入 — SDK 要求 types.Image(imageBytes, mimeType), 不能直接传 PIL.Image
        image_input = None
        if img_path:
            with open(img_path, "rb") as _f:
                _bytes = _f.read()
            _ext = img_path.lower().rsplit(".", 1)[-1]
            _mime = "image/png" if _ext == "png" else "image/jpeg"
            image_input = types.Image(image_bytes=_bytes, mime_type=_mime)

        try:
            # 提交异步生成请求
            operation = self.client.models.generate_videos(
                model=self.model_name,
                prompt=prompt,
                image=image_input,
                config=config
            )

            # 轮询直至完成
            while not operation.done:
                print(f"⌛ [Veo] Generating... ({self.model_name})")
                time.sleep(10)
                operation = self.client.operations.get(operation)

            if operation.response and operation.response.generated_videos:
                return operation.response.generated_videos[0]
            else:
                raise Exception("Veo return empty response.")

        except Exception as e:
            raise Exception(f"Veo API Error: {str(e)}")

    def save_video(self, video_obj, save_path):
        """下载并保存视频文件"""
        try:
            self.client.files.download(file=video_obj.video)
            video_obj.video.save(save_path)
            print(f"✅ Saved to {save_path}")
        except Exception as e:
            print(f"❌ Download Error: {e}")