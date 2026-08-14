import dashscope
from http import HTTPStatus
from dashscope import VideoSynthesis
from .base_generator import BaseVideoGenerator
import os
import subprocess
import shutil
from .base_generator import BaseVideoGenerator

import time
from dashscope import VideoSynthesis
from http import HTTPStatus


class Wan26Generator:
    """Wan 2.6 类：严格适配 Wan 2.6 参数格式并加入耗时统计"""
    def __init__(self, api_key, model="wan2.6-i2v"):
        self.api_key = api_key
        self.model = model
        dashscope.api_key = api_key
    def generate(self, prompt, img_path, duration=10, seed=None, **kwargs):
        start_time = time.time()

        # 2.6 的参数：使用 size 代替 resolution, 加入 shot_type
        call_kwargs = dict(
            model=self.model,
            prompt=prompt,
            img_url=f"file://{img_path}",
            resolution="720P",
            duration=duration,        # Wan 2.6 通常支持 10s
            prompt_extend=False,
            shot_type="single",
            audio=False,
        )
        if seed is not None:
            call_kwargs["seed"] = int(seed)
        rsp = VideoSynthesis.async_call(**call_kwargs)

        if rsp.status_code == HTTPStatus.OK:
            task_id = rsp.output.task_id
            print(f"✅ [Wan2.6 Submit] Task ID: {task_id}")
        else:
            raise Exception(f"❌ [Wan2.6 Submit Error] Code: {rsp.code}, Msg: {rsp.message}")

        # 阻塞等待渲染
        rsp = VideoSynthesis.wait(rsp)

        if rsp.status_code == HTTPStatus.OK:
            if hasattr(rsp.output, 'video_url') and rsp.output.video_url:
                elapsed = time.time() - start_time
                print(f"✨ [Wan2.6 Success] Task {task_id} done! ⏱️ 耗时: {elapsed:.1f}s")
                return rsp.output.video_url
            else:
                raise Exception(f"⚠️ [Wan2.6 Output Error] Task {task_id} success but NO URL.")
        else:
            raise Exception(f"❌ [Wan2.6 Gen Error] Task {task_id} Failed! Code: {rsp.code}, Msg: {rsp.message}")