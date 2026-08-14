#!/bin/bash
set -euo pipefail

export MASTER_PORT=$(shuf -i 20000-65535 -n 1)

# ==============================================================================
# Vidu API 并行推理自动化脚本 (dashscope MaaS)
# Usage: ./run_vidu.sh <DATASET> <TEST_NAME> [MODEL] [N_PROC] [DATA_ROOT]
# Example: ./run_vidu.sh common_set_50 vidu_common50 "vidu-q1" 2
# ==============================================================================

DATASET=${1:-"debug"}
TEST_NAME=${2:-"vidu"}
# Vidu 在 dashscope 上通过 workspace-specific MaaS endpoint 提供；正式跑改成实际模型名
MODEL=${3:-"vidu-q1"}
N_PROC=${4:-2}
DATA_ROOT=${5:-"./data"}
DURATION=${6:-5}
N_ATTEMPTS=${7:-1}

# Vidu 也走 dashscope 通路（可能需要 workspace 专用 base_url — 见 dashscope_extra_generator.py）
API_KEY="${DASHSCOPE_API_KEY:-}"
# Vidu 走 workspace-specific MaaS endpoint (see inference/dashscope_extra_generator.py)
# 若默认 endpoint 不通,通过 DASHSCOPE_BASE_URL env 指定 workspace 专用 URL
BASE_URL="${DASHSCOPE_BASE_URL:-}"

GT_JSON="${DATA_ROOT}/${DATASET}/summary.json"
PRED_ROOT="${DATA_ROOT}/${DATASET}/generated_data/${TEST_NAME}"

if [ -d "${PRED_ROOT}/temp_worker" ]; then
    echo "🧹 Cleaning up old temp workers..."
    rm -rf "${PRED_ROOT}/temp_worker"
fi

if [ ! -f "$GT_JSON" ]; then
    echo "❌ Error: GT summary JSON not found at $GT_JSON"
    exit 1
fi

if [ -z "$API_KEY" ]; then
    echo "❌ Error: DASHSCOPE_API_KEY not set. Export it or edit the script."
    exit 1
fi

mkdir -p "$PRED_ROOT"

echo "------------------------------------------------"
echo "🚀 Starting Vidu (dashscope MaaS) Inference"
echo "📍 Dataset:      $DATASET"
echo "📍 Test Name:    $TEST_NAME"
echo "📍 Model:        $MODEL"
echo "📄 Task List:    $GT_JSON"
echo "📂 Pred Root:    $PRED_ROOT"
echo "⚙️  Processes:    $N_PROC"
[ -n "$BASE_URL" ] && echo "🌐 Base URL:     $BASE_URL"
echo "------------------------------------------------"

# Vidu VideoSynthesis 用 media[0].type='image' (不同于 happyhorse 用 first_frame);
# run_api_parallel.py 根据 model 名转发到 DashScopeVideoGenerator(media_type='image').
EXTRA_ARGS=()
[ -n "$BASE_URL" ] && EXTRA_ARGS+=(--base_url "$BASE_URL")

python run_api_parallel.py \
    --n_proc "$N_PROC" \
    --model "$MODEL" \
    --gt_root "$GT_JSON" \
    --pred_root "$PRED_ROOT" \
    --api_key "$API_KEY" \
    --duration "$DURATION" \
    --n_attempts "$N_ATTEMPTS" \
    "${EXTRA_ARGS[@]}"

echo "------------------------------------------------"
echo "✅ Inference dispatched to: $PRED_ROOT"
echo "------------------------------------------------"
