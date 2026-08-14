# Video Generation Guide

This is the single source of truth for video generation in this repository. It covers dataset preparation, API models, local open-source models, the native distributed LingBot-Video and PF_Wan backends, output layout, and resume behavior.

## 1. Choose an inference route

The inference routes intentionally remain separate:

| Route | Models | Entry point |
| --- | --- | --- |
| API batch | Veo 3.1, Wan2.6, Wan2.6 Flash, Kling, HappyHorse, Vidu | `run_api_parallel.py` or `scripts_inference/run_*.sh` |
| Unified local | Wan2.2, Cosmos-Predict2.5, Cosmos3 (`cosmos-framework`), CogVideoX, WoW, GigaWorld, ABot-PhysWorld | `run_local.py` through a model launcher |
| Standalone reproduced | Cosmos3 (`Cosmos3OmniPipeline`) | `scripts_inference/run_cosmos3_diffusers.py` |
| Native distributed | LingBot-Video, PhysisForcing PF_Wan | Dedicated adapters under `scripts_inference/lingbot/` and `scripts_inference/pf_wan/` |
| Adapter only | HunyuanVideo 1.5, LongCat-Video, Stable Video Diffusion | Generator classes under `inference/`; not yet wired to `run_local.py` or launchers |

LingBot-Video uses native multi-GPU FSDP. PF_Wan uses native `torchrun`/Ulysses and a manifest. Neither should be forced into `inference/*_generator.py`.

`inference/hunyuan_generator.py`, `inference/longcat_generator.py`, and `inference/svd_generator.py` are currently low-level adapters only. They require their respective upstream code, dependencies, and checkpoints, but this repository does not yet expose end-to-end CLI arguments or launch scripts for them. Treat them as development integrations, not runnable benchmark recipes.

## 2. Repository and data layout

Keep third-party repositories, checkpoints, generated videos, and logs outside Git tracking:

```text
MLLM-as-Embodied-World-Judge/
├── data/<dataset>/
│   ├── summary.json
│   ├── gt_data/
│   └── generated_data/
├── inference/                       # unified local adapters
├── scripts_inference/
│   ├── run_*.sh                    # API and unified local launchers
│   ├── run_cosmos3_diffusers.py    # reproduced Cosmos3 benchmark path
│   ├── lingbot/
│   │   ├── run_lingbot.sh
│   │   └── run_lingbot_batch.py
│   └── pf_wan/
│       ├── prepare_pf_wan_manifest.py
│       ├── run_pf_wan.sh
│       ├── collect_pf_wan_results.py
│       └── patches/resumable_outputs.patch
├── envs/
├── third_party/
├── checkpoints/
├── run_api_parallel.py
└── run_local.py
```

A ground-truth sample should look like:

```text
data/<dataset>/gt_data/<task>/<episode>/
├── prompt/
│   ├── init_frame.png
│   └── prompt.txt
└── video.mp4
```

### `summary.json`

Each row must identify the prompt, conditioning image, task, and episode:

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

Prompt fields used by the runners are:

- `prompt`: original instruction;
- `prompt_prefix`: instruction with the experiment prefix;
- `prompt_rewrite`: rewritten instruction.

Select one explicitly with `--prompt_key` or the launcher argument. Do not mix variants in one run.

## 3. Output contract and resume

All backends must ultimately produce:

```text
data/<dataset>/generated_data/<run_name>/<task>/<episode>/1/
├── <task>_<episode>.mp4
└── prompt/
    ├── init_frame.png
    └── prompt.txt
```

Some backends also save `generation_config.json`. Use a descriptive `run_name` such as:

```text
wan22_ti2v5b_f169_prefix
lingbot_video_moe_base
pf_wan14b_f81_common50
```

Resumable backends skip existing non-empty videos. This is safe only when `<task>/<episode>` uniquely identifies a sample.

## 4. Dataset preparation

Before a full run:

1. place the dataset under `data/<dataset>/`;
2. validate `summary.json` and every conditioning image;
3. select one prompt field;
4. set frame count, fps, duration, seed, and run name explicitly;
5. run a small smoke test;
6. run different datasets as separate jobs;
7. verify output counts before evaluation.

### `open_x_embodiment` and `robotwin`

The level-1 experiments used this prefix:

```text
In a fixed robotic workspace, the first frame remains unchanging. The environment remains strictly identical to the first frame; no new objects, extra limbs, or entities are introduced. The existing robotic arm is a rigid, physically consistent embodied agent. The robotic arm maintains high stability with no deformation. From its initial position, the robotic arm manipulates the scene to
```

Store it in `prompt_prefix` so `prompt` remains available.

For `open_x_embodiment`, keep the existing `task_name`.

For `robotwin`, semantic labels may repeat while many rows also use `episode_0001`. Rewrite `task_name` to the unique ID from `gt_path`, such as `task_0001`. Otherwise outputs collide and resume may skip unrelated samples.

Frame count is not interchangeable across model families. Record it in the run name or generation metadata.

## 5. Common setup

```bash
mkdir -p third_party checkpoints data logs

git clone https://github.com/Wan-Video/Wan2.2.git third_party/Wan2.2
git clone https://github.com/nvidia-cosmos/cosmos-predict2.5.git third_party/cosmos-predict2.5
git clone https://github.com/zai-org/CogVideo.git third_party/CogVideo
git clone https://github.com/wow-world-model/wow-world-model.git third_party/WoW
git clone https://github.com/open-gigaai/giga-world-0.git third_party/GigaWorld
git clone https://github.com/Robbyant/lingbot-video.git third_party/lingbot-video
git clone https://github.com/DAGroup-PKU/PhysisForcing.git third_party/PhysisForcing

ffmpeg -version
pip install -U "huggingface_hub[cli]"
```

Use a separate environment per local model. Follow the checked-out upstream revision for standard backends. Benchmark-specific LingBot and PF_Wan snapshots are documented below.

| Model | Recommended checkpoint location |
| --- | --- |
| Wan2.2 TI2V-5B | `checkpoints/Wan2.2/Wan2.2-TI2V-5B/` |
| Cosmos-Predict2.5 | `checkpoints/Cosmos-Predict2.5/<variant>/` |
| CogVideoX | `checkpoints/CogVideo/CogVideoX1.5-5B-I2V/` |
| WoW | `checkpoints/WoW/<checkpoint>/` |
| GigaWorld | `checkpoints/GigaWorld/<checkpoint>/` |
| Cosmos3 Diffusers | Hugging Face model ID, normally `nvidia/Cosmos3-Nano`; cached through `HF_HOME` |
| Cosmos3 framework | set `COSMOS3_ROOT` and the launcher checkpoint argument |
| LingBot-Video | `checkpoints/LingBot-Video/lingbot-video-moe-30b-a3b-base/` |
| PF_Wan | `checkpoints/PhysisForcing/PF_Wan/` |

## 6. API models

Set credentials through environment variables:

```bash
export DASHSCOPE_API_KEY="<key>"       # Wan, Kling, HappyHorse, Vidu
export GOOGLE_API_KEY="<key>"          # Veo; GEMINI_API_KEY is also accepted
```

Recommended direct invocation:

```bash
python run_api_parallel.py --n_proc 4 --model wan2.6-i2v-flash --gt_root data/open_x_embodiment/summary.json --pred_root data/open_x_embodiment/generated_data/wan26_flash --duration 5 --n_attempts 1 --prompt_key prompt_prefix --sample_keep_prob 1.0
```

The default keep probability is controlled by `KEEP_RATE`. Pass `--sample_keep_prob 1.0` for a full run.

| Model | Launcher |
| --- | --- |
| Veo 3.1 Lite | `scripts_inference/run_veo31lite.sh` |
| Wan2.6 | `scripts_inference/run_wan26.sh` |
| Wan2.6 Flash | `scripts_inference/run_wan26flash.sh` |
| Kling | `scripts_inference/run_kling.sh` |
| HappyHorse | `scripts_inference/run_happyhorse.sh` |
| Vidu | `scripts_inference/run_vidu.sh` |

The direct Python entry is preferable when changing prompt fields or sampling because all parameters remain visible.

## 7. Unified local models

The launchers construct input/output paths, assign GPUs, and invoke `run_local.py`.

### Wan2.2 TI2V-5B

```bash
bash scripts_inference/run_wan22.sh open_x_embodiment wan22_ti2v_5b ti2v-5B 1 ./data 0 1
```

Use `WAN_CKPT`, `WAN_ROOT`, and `WAN_SIZE` for overrides.

### Cosmos-Predict2.5

```bash
bash scripts_inference/run_cosmos.sh open_x_embodiment cosmos2.5_2b 2B/post-trained 1 ./data 0 1 49 50
```

The final values are frame count and sampling steps. For offline use, prepare the selected checkpoint, `nvidia/Cosmos-Reason1-7B`, and the Hugging Face cache on an online host, then set `HF_HOME`, `TRANSFORMERS_OFFLINE=1`, and `HF_HUB_OFFLINE=1`.

### Cosmos3: reproduced Diffusers route

The published `cosmos3_*` leaderboard rows were generated with `scripts_inference/run_cosmos3_diffusers.py`. This is the benchmark reproduction path: it loads `diffusers.Cosmos3OmniPipeline` directly and does not need a `third_party/cosmos-framework` checkout.

The checked-in environment is an exact snapshot of the verified run. `Cosmos3OmniPipeline` comes from the pinned Diffusers Git revision rather than a released PyPI package. The extra PyTorch index is needed for the pinned CUDA 12.8 wheel:

```bash
python -m venv .venv-cosmos3-diffusers
source .venv-cosmos3-diffusers/bin/activate
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r envs/cosmos3_diffusers.txt
```

Set `HF_TOKEN` if the model download requires authentication. `HF_HOME` can redirect the model cache.

Run one prompt variant explicitly for a full dataset:

```bash
python scripts_inference/run_cosmos3_diffusers.py \
    --data_root ./data \
    --output_root ./output_video \
    --datasets open_x_embodiment \
    --prompt_keys prompt_prefix \
    --test_name_template 'cosmos3_nano_{suffix}' \
    --sample_keep_prob 1.0 \
    --model_id nvidia/Cosmos3-Nano \
    --rank 0 --world_size 1 \
    --width 832 --height 480 --fps 24 --num_frames 121 \
    --num_inference_steps 35 --guidance_scale 6.0 --flow_shift 10.0
```

The canonical result is written under `data/<dataset>/generated_data/<test_name>/...`; `--output_root` receives a link or copy in the same relative layout. Existing non-empty videos are skipped. Progress and failures are recorded under `logs/`.

Important defaults when the corresponding flags are omitted:

- `--prompt_keys prompt_rewrite prompt_prefix` generates two variants;
- `--sample_keep_prob 0.4` selects a deterministic subset;
- `--test_name_template cosmos3_nano_{suffix}` produces distinct run names;
- `--dry_run` prints selection and shard counts without loading the model;
- `--max_items 1` is useful for a one-sample smoke test.

For multiple GPUs, start one process per GPU with a different `--rank` and the same `--world_size`; the script shards work but does not launch the other ranks itself. Use `CUDA_VISIBLE_DEVICES` to bind each process. Quantization, device mapping, and CPU-offload switches are available through `--load_in_8bit`, `--device_map`, `--cpu_offload`, and `--sequential_offload`.

### Cosmos3: `cosmos-framework` compatibility route

The older unified-local integration remains available through `run_local.py` and `inference/cosmos3_generator.py`:

```bash
COSMOS3_ROOT="<path-to-cosmos-framework>" bash scripts_inference/run_cosmos3.sh open_x_embodiment cosmos3_nano Cosmos3-Nano 8 ./data 0,1,2,3,4,5,6,7 1 0.8
```

For Cosmos3-Super, use one launcher process, the Super checkpoint, and the required internal GPU count. The final positional value is deterministic keep probability. This route is retained for compatibility; use the Diffusers route above when reproducing the published `cosmos3_*` rows.

### CogVideoX

```bash
bash scripts_inference/run_cogvideo.sh open_x_embodiment cogvideo CogVideoX1.5-5B-I2V 1 ./data 0 1
```

### WoW

```bash
bash scripts_inference/run_wow.sh open_x_embodiment wow 1 ./data 0 1 49 50
```

### GigaWorld

```bash
bash scripts_inference/run_gigaworld.sh open_x_embodiment gigaworld GigaWorld 1 ./data 0 1
```

Inspect each launcher header for its positional arguments and environment overrides.

## 8. LingBot-Video

Upstream: [Robbyant/lingbot-video](https://github.com/Robbyant/lingbot-video)

### Reproduced benchmark setting

- LingBot-Video-MoE 30B-A3B base;
- TI2V/first-frame conditioning;
- direct Diffusers backend;
- native 4-GPU FSDP;
- prompt passed directly from the selected summary field;
- no official prompt rewriter, Auto Negative stage, or refiner.

This differs from the upstream full recommended pipeline.

### Environment and checkpoint

```bash
python -m venv .venv-lingbot
source .venv-lingbot/bin/activate
pip install -r envs/lingbot_video.txt
pip install -e third_party/lingbot-video
```

Place the base checkpoint at:

```text
checkpoints/LingBot-Video/lingbot-video-moe-30b-a3b-base/
```

### Run

```bash
GPU_COUNT=4 bash scripts_inference/lingbot/run_lingbot.sh open_x_embodiment lingbot_video_moe_base prompt_prefix
```

Arguments are `DATASET RUN_NAME [PROMPT_KEY]`. Override `PROJECT_ROOT`, `DATA_ROOT`, `LINGBOT_ROOT`, `LINGBOT_CKPT`, `GPU_COUNT`, `MASTER_PORT`, or `PYTHON_BIN` as needed.

The launcher calls `scripts_inference/lingbot/run_lingbot_batch.py`. It loads the pipeline once across the FSDP group, derives stable per-sample seeds, resumes completed outputs, and writes the final layout directly.

## 9. PhysisForcing PF_Wan

Upstream: [DAGroup-PKU/PhysisForcing](https://github.com/DAGroup-PKU/PhysisForcing)

### Reproduced benchmark setting

- PF_Wan on Wan2.2-A14B;
- Python 3.11, PyTorch 2.7.1, CUDA 12.8;
- FlashAttention 2.8.3;
- 4 × H100 80GB;
- Ulysses size 4 and hybrid GPU count 4;
- seed 42, 81 frames, 16 fps.

On a GLIBC 2.31 host, use a compatible manylinux FlashAttention wheel; wheels requiring GLIBC 2.32 will not load.

### Environment and checkpoint

```bash
python -m venv .venv-pf-wan
source .venv-pf-wan/bin/activate
pip install -r envs/pf_wan.txt
```

```text
checkpoints/PhysisForcing/PF_Wan/
├── backbone.pth
├── models_t5_umt5-xxl-enc-bf16.pth
├── Wan2.1_VAE.pth
└── google/
    └── umt5-xxl/
```

Only `backbone.pth` is PF_Wan-specific. T5, VAE, and tokenizer assets may be symlinked from a complete Wan2.2-I2V-A14B checkpoint.

### Apply the resume patch

```bash
git -C third_party/PhysisForcing apply ../../scripts_inference/pf_wan/patches/resumable_outputs.patch
```

The patch replaces timestamped output directories with a stable job directory and skips existing videos.

### Step 1: prepare the manifest

PF_Wan reads one `<prompt>@@<conditioning_image_path>` entry per line:

```bash
python scripts_inference/pf_wan/prepare_pf_wan_manifest.py --data-root data --work-root outputs/pf_wan_inputs --name common50 --datasets open_x_embodiment robotwin --prompt-key prompt_prefix --run-name pf_wan14b_f81_common50
```

The adapter creates a uniquely named image link per row. Do not resolve the link in the manifest: PF_Wan derives output names from image basenames, so resolving everything to `init_frame.png` causes overwrites.

Use `--keep-prob`, `--sample-seed`, and `--sample-model-key` for deterministic subsets. Pass `--overwrite` only when intentionally rebuilding an existing work directory.

### Step 2: run native inference

```bash
GPU_COUNT=4 NUM_FRAMES=81 FPS=16 SEED=42 bash scripts_inference/pf_wan/run_pf_wan.sh common50 outputs/pf_wan_inputs/common50/manifest.txt 4
```

Override `PF_WAN_ROOT`, `PF_WAN_CKPT`, `PF_WAN_RAW_ROOT`, `TORCHRUN_BIN`, `MASTER_PORT`, or `CUDA_VISIBLE_DEVICES` as needed.

Raw output:

```text
outputs/pf_wan_raw/<name>/<unique_image_basename>.mp4
```

### Step 3: collect results

```bash
python scripts_inference/pf_wan/collect_pf_wan_results.py --mapping outputs/pf_wan_inputs/common50/mapping.json --raw-dir outputs/pf_wan_raw/common50 --data-root data
```

The collector uses `run_name` from the mapping and fails on missing videos unless `--allow-missing` is explicit.

Benchmark run names included:

```text
pf_wan14b_f81_common50
pf_wan14b_f81_prefix
pf_wan14b_f81_rewrite
```

## 10. Validate before evaluation

Check:

- selected sample count versus generated video count;
- zero-byte, truncated, or remaining `.partial.mp4` files;
- unique `<task>/<episode>` paths;
- saved prompt and conditioning image;
- frame count, fps, seed, prompt field, and checkpoint.

Do not commit generated videos, `outputs/`, logs, checkpoints, weights, or cloned third-party repositories.
