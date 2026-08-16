# VideoGen — Video Generation Inference

Isolated, self-contained inference codebase for embodied-video generation benchmarks. Extracted from [MLLM-as-Embodied-World-Judge](https://github.com/SiyuanMaCS/MLLM-as-Embodied-World-Judge).

## Supported Models

| Route | Model | Entry point | Status |
| --- | --- | --- | --- |
| API batch | Veo 3.1 Lite | `scripts_inference/run_veo31lite.sh` | ✅ |
| API batch | Wan2.6 | `scripts_inference/run_wan26.sh` | ✅ |
| API batch | Wan2.6 Flash | `scripts_inference/run_wan26flash.sh` | ✅ |
| API batch | Kling | `scripts_inference/run_kling.sh` | ✅ |
| API batch | HappyHorse | `scripts_inference/run_happyhorse.sh` | ✅ |
| API batch | Vidu | `scripts_inference/run_vidu.sh` | ✅ |
| Unified local | Wan2.2 TI2V-5B | `scripts_inference/run_wan22.sh` | ✅ |
| Unified local | Cosmos-Predict2.5 | `scripts_inference/run_cosmos.sh` | ✅ |
| Unified local | Cosmos3 (framework) | `scripts_inference/run_cosmos3.sh` | ✅ |
| Unified local | CogVideoX | `scripts_inference/run_cogvideo.sh` | ✅ |
| Unified local | WoW | `scripts_inference/run_wow.sh` | ✅ |
| Unified local | GigaWorld | `scripts_inference/run_gigaworld.sh` | ✅ |
| Unified local | HunyuanVideo 1.5 | `scripts_inference/run_hunyuan.sh` | ✅ |
| Unified local | LongCat-Video | `scripts_inference/run_longcat.sh` | ✅ |
| Unified local | SVD-xt | `scripts_inference/run_svd.sh` | ✅ |
| Standalone | Cosmos3 (Diffusers) | `scripts_inference/run_cosmos3_diffusers.py` | ✅ |
| Native distributed | LingBot-Video (FSDP) | `scripts_inference/lingbot/` | ✅ |
| Native distributed | PF_Wan (torchrun) | `scripts_inference/pf_wan/` | ✅ |

All 18 models have end-to-end runnable scripts.

## Repository Layout

```text
VideoGen/
├── inference/                  # Model adapters (one per model family)
│   ├── wan22_generator.py
│   ├── cosmos_generator.py
│   ├── cosmos3_generator.py
│   ├── cogvideo_generator.py
│   ├── wow_generator.py
│   ├── abot_physworld_generator.py
│   ├── hunyuan_generator.py
│   ├── longcat_generator.py
│   ├── svd_generator.py
│   ├── wan_generator.py        # DashScope API Wan
│   ├── veo_generator.py        # Veo API adapter
│   └── dashscope_extra_generator.py
├── scripts_inference/          # Launch scripts
│   ├── run_*.sh               # Per-model launchers
│   ├── lingbot/               # LingBot-Video native FSDP
│   └── pf_wan/                # PF_Wan native torchrun + manifest
├── envs/                       # Pinned environment specs
│   ├── cosmos3_diffusers.txt
│   ├── lingbot_video.txt
│   └── pf_wan.txt
├── run_local.py                # Unified local model entry point
├── run_api_parallel.py         # API model parallel entry point
├── data/                       # Dataset mount point (git-ignored)
├── third_party/                # Upstream repos (git-ignored)
└── checkpoints/                # Model weights (git-ignored)
```

## Quick Start

### 1. Clone and prepare directories

```bash
git clone https://github.com/SiyuanMaCS/VideoGen.git
cd VideoGen
mkdir -p third_party checkpoints data logs
```

### 2. Environment setup

**Common requirements** (all models):

```bash
# Python 3.10+ recommended
pip install opencv-python-headless numpy Pillow
pip install -U "huggingface_hub[cli]"
ffmpeg -version   # ensure ffmpeg is available
```

**Per-model environments** — use a separate venv/conda per model to avoid dependency conflicts:

| Model | Environment |
| --- | --- |
| Wan2.2 | `pip install -r envs/wan22.txt` |
| Cosmos-Predict2.5 | Follow `third_party/cosmos-predict2.5` upstream |
| Cosmos3 (Diffusers) | `pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r envs/cosmos3_diffusers.txt` |
| Cosmos3 (framework) | `bash scripts_inference/setup_cosmos3.sh` |
| CogVideoX | `pip install -r envs/cogvideo.txt` |
| WoW | `pip install -r envs/wow.txt` |
| GigaWorld | `pip install -r envs/gigaworld.txt` |
| LingBot-Video | `pip install -r envs/lingbot_video.txt && pip install -e third_party/lingbot-video` |
| PF_Wan | `pip install -r envs/pf_wan.txt` (Python 3.11, PyTorch 2.7.1, CUDA 12.8, FlashAttn 2.8.3) |
| HunyuanVideo 1.5 | `pip install -r third_party/HunyuanVideo-1.5/requirements.txt` + matching flash-attn wheel |
| LongCat-Video | `pip install -r third_party/LongCat-Video/requirements.txt` + matching flash-attn wheel |
| SVD-xt | `pip install diffusers transformers accelerate` (standard diffusers stack) |
| API models | `pip install openai dashscope requests` |

**Notes:**
- Flash-attn: install with `pip install flash-attn --no-build-isolation` after torch is available.
- Proxy: if behind a firewall, set `http_proxy` / `https_proxy` and `HF_TOKEN` for model downloads.
- GLIBC: on GLIBC 2.31 hosts, use compatible manylinux flash-attn wheels (2.32+ wheels won't load).

### 3. Dataset layout

Place datasets under `data/<dataset>/` with a `summary.json`:

```json
[
  {
    "gt_path": "data/open_x_embodiment/gt_data/task_0001/episode_0001/video.mp4",
    "image": "data/open_x_embodiment/gt_data/task_0001/episode_0001/prompt/init_frame.png",
    "prompt": ["Put the object into the container."],
    "task_name": "task_0001",
    "episode_name": "episode_0001",
    "duration": 5.0
  }
]
```

Prompt fields: `prompt`, `prompt_prefix`, `prompt_rewrite`. Select one with `--prompt_key`.

### 3. Output contract

All backends produce:

```text
data/<dataset>/generated_data/<run_name>/<task>/<episode>/1/
├── <task>_<episode>.mp4
└── prompt/
    ├── init_frame.png
    └── prompt.txt
```

Resumable: existing non-empty videos are skipped on rerun.

## Running Models

### API Models

```bash
export DASHSCOPE_API_KEY="<key>"
export GOOGLE_API_KEY="<key>"

python run_api_parallel.py \
    --n_proc 4 --model wan2.6-i2v-flash \
    --gt_root data/open_x_embodiment/summary.json \
    --pred_root data/open_x_embodiment/generated_data/wan26_flash \
    --duration 5 --n_attempts 1 --prompt_key prompt_prefix --sample_keep_prob 1.0
```

### Unified Local Models

All use the same pattern: `bash scripts_inference/run_<model>.sh <DATASET> <TEST_NAME> [N_PROC] [DATA_ROOT] [GPU_IDS] [N_ATTEMPTS]`

```bash
# Wan2.2
bash scripts_inference/run_wan22.sh open_x_embodiment wan22_ti2v_5b 1 ./data 0 1

# Cosmos-Predict2.5
bash scripts_inference/run_cosmos.sh open_x_embodiment cosmos2.5_2b 2B/post-trained 1 ./data 0 1 49 50

# CogVideoX
bash scripts_inference/run_cogvideo.sh open_x_embodiment cogvideo CogVideoX1.5-5B-I2V 1 ./data 0 1

# WoW
bash scripts_inference/run_wow.sh open_x_embodiment wow 1 ./data 0 1 49 50

# GigaWorld
bash scripts_inference/run_gigaworld.sh open_x_embodiment gigaworld GigaWorld 1 ./data 0 1

# HunyuanVideo 1.5 (480p i2v step-distilled)
bash scripts_inference/run_hunyuan.sh open_x_embodiment hunyuan15 1 ./data 0 1

# LongCat-Video (base i2v, 480p, ~10 min/video on H100)
bash scripts_inference/run_longcat.sh open_x_embodiment longcat 1 ./data 0 1

# SVD-xt (image-only, no text prompt)
bash scripts_inference/run_svd.sh open_x_embodiment svd_xt 1 ./data 0 1
```

### Cosmos3 (Diffusers — Benchmark Reproduction Path)

```bash
python -m venv .venv-cosmos3-diffusers
source .venv-cosmos3-diffusers/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r envs/cosmos3_diffusers.txt

python scripts_inference/run_cosmos3_diffusers.py \
    --data_root ./data --output_root ./output_video \
    --datasets open_x_embodiment --prompt_keys prompt_prefix \
    --test_name_template 'cosmos3_nano_{suffix}' --sample_keep_prob 1.0 \
    --model_id nvidia/Cosmos3-Nano \
    --rank 0 --world_size 1 \
    --width 832 --height 480 --fps 24 --num_frames 121 \
    --num_inference_steps 35 --guidance_scale 6.0 --flow_shift 10.0
```

Multi-GPU: start one process per GPU with different `--rank` and same `--world_size`.

### LingBot-Video (Native 4-GPU FSDP)

```bash
python -m venv .venv-lingbot
source .venv-lingbot/bin/activate
pip install -r envs/lingbot_video.txt
pip install -e third_party/lingbot-video

GPU_COUNT=4 bash scripts_inference/lingbot/run_lingbot.sh open_x_embodiment lingbot_video_moe_base prompt_prefix
```

Checkpoint: `checkpoints/LingBot-Video/lingbot-video-moe-30b-a3b-base/`

### PF_Wan (Native torchrun, 4×H100)

```bash
python -m venv .venv-pf-wan
source .venv-pf-wan/bin/activate
pip install -r envs/pf_wan.txt

# Step 1: prepare manifest
python scripts_inference/pf_wan/prepare_pf_wan_manifest.py \
    --data-root data --work-root outputs/pf_wan_inputs --name common50 \
    --datasets open_x_embodiment robotwin --prompt-key prompt_prefix \
    --run-name pf_wan14b_f81_common50

# Step 2: run
GPU_COUNT=4 NUM_FRAMES=81 FPS=16 SEED=42 \
    bash scripts_inference/pf_wan/run_pf_wan.sh common50 outputs/pf_wan_inputs/common50/manifest.txt 4

# Step 3: collect results
python scripts_inference/pf_wan/collect_pf_wan_results.py \
    --mapping outputs/pf_wan_inputs/common50/mapping.json \
    --raw-dir outputs/pf_wan_raw/common50 --data-root data
```

## Checkpoint Locations

| Model | Path |
| --- | --- |
| Wan2.2 TI2V-5B | `checkpoints/Wan2.2/Wan2.2-TI2V-5B/` |
| Cosmos-Predict2.5 | `checkpoints/Cosmos-Predict2.5/<variant>/` |
| CogVideoX | `checkpoints/CogVideo/CogVideoX1.5-5B-I2V/` |
| WoW | `checkpoints/WoW/<checkpoint>/` |
| GigaWorld | `checkpoints/GigaWorld/<checkpoint>/` |
| Cosmos3 Diffusers | HF model ID `nvidia/Cosmos3-Nano` (cached via `HF_HOME`) |
| Cosmos3 framework | Set `COSMOS3_ROOT` + launcher checkpoint arg |
| LingBot-Video | `checkpoints/LingBot-Video/lingbot-video-moe-30b-a3b-base/` |
| PF_Wan | `checkpoints/PhysisForcing/PF_Wan/` |
| HunyuanVideo 1.5 | `checkpoints/HunyuanVideo-1.5/` |
| LongCat-Video | `checkpoints/LongCat-Video/` |
| SVD-xt | `checkpoints/SVD/stable-video-diffusion-img2vid-xt/` |

## Third-Party Repos

```bash
git clone https://github.com/Wan-Video/Wan2.2.git third_party/Wan2.2
git clone https://github.com/nvidia-cosmos/cosmos-predict2.5.git third_party/cosmos-predict2.5
git clone https://github.com/zai-org/CogVideo.git third_party/CogVideo
git clone https://github.com/wow-world-model/wow-world-model.git third_party/WoW
git clone https://github.com/open-gigaai/giga-world-0.git third_party/GigaWorld
git clone https://github.com/Robbyant/lingbot-video.git third_party/lingbot-video
git clone https://github.com/DAGroup-PKU/PhysisForcing.git third_party/PhysisForcing
```

## Deterministic Sampling

Use `--sample_keep_prob` to run a deterministic subset (e.g. 0.4 = 40%). The same `--sample_seed` + `--sample_model_key` + sample ID always produces the same keep/skip decision across reruns.

## Validation Checklist

Before evaluation, verify:
- Sample count matches generated video count
- No zero-byte or `.partial.mp4` files
- Unique `<task>/<episode>` paths
- Correct prompt field, frame count, fps, seed
