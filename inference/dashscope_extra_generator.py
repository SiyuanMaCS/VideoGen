"""
Generic DashScope VideoSynthesis image-to-video generator for models that use
the same async API as Wan (HappyHorse, Vidu, ...). Returns a downloadable
video_url, like Wan26Generator, so run_api_parallel.py's URL-download path works.
"""
import time
import random
import dashscope
from dashscope import VideoSynthesis
from http import HTTPStatus


class DashScopeVideoGenerator:
    def __init__(self, api_key, model, resolution="720P", default_duration=5,
                 base_url=None, media_type="image"):
        self.api_key = api_key
        self.model = model
        self.resolution = resolution
        self.default_duration = default_duration
        self.base_url = base_url
        # media[0].type differs per model: Vidu wants 'image', HappyHorse wants 'first_frame'.
        self.media_type = media_type
        dashscope.api_key = api_key
        # Some third-party models (e.g. Vidu) are deployed on a workspace-specific
        # MaaS endpoint and are NOT reachable via the default dashscope.aliyuncs.com.
        if base_url:
            dashscope.base_http_api_url = base_url

    def generate(self, prompt, img_path, duration=None, **kwargs):
        dur = int(duration or self.default_duration)
        start = time.time()
        # Random per-video seed so outputs are not deterministic (siyuan's requirement).
        # random is auto-seeded from OS entropy per worker process, so seeds differ
        # across the multiprocessing pool.
        seed = random.randint(0, 2147483647)
        # HappyHorse / Vidu expect the first-frame image in input.media (not img_url).
        # The SDK uploads the local file:// and rewrites the url.
        rsp = VideoSynthesis.async_call(
            model=self.model,
            prompt=prompt,
            media=[{"type": self.media_type, "url": f"file://{img_path}"}],
            resolution=self.resolution,
            duration=dur,
            seed=seed,
        )
        if rsp.status_code != HTTPStatus.OK:
            raise Exception(f"[{self.model} submit] Code: {rsp.code}, Msg: {rsp.message}")
        task_id = rsp.output.task_id
        print(f"✅ [{self.model} Submit] {task_id}", flush=True)
        rsp = VideoSynthesis.wait(rsp)
        if rsp.status_code == HTTPStatus.OK and getattr(rsp.output, "video_url", None):
            print(f"✨ [{self.model} Success] {task_id} ⏱️ {time.time()-start:.1f}s", flush=True)
            return rsp.output.video_url
        raise Exception(f"[{self.model}] no video_url: code={getattr(rsp,'code',None)} msg={getattr(rsp,'message',None)}")
