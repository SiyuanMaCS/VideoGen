#!/bin/bash
set -euo pipefail

export MASTER_PORT=$(shuf -i 20000-65535 -n 1)

# ==============================================================================
# HappyHorse API 并行推理自动化脚本 (dashscope)
# Usage: ./run_happyhorse.sh <DATASET> <TEST_NAME> [MODEL] [N_PROC] [DATA_ROOT]
# Example: ./run_happyhorse.sh common_set_50 happyhorse_common50 "happyhorse-i2v" 2
# ==============================================================================

DATASET=${1:-"debug"}
TEST_NAME=${2:-"happyhorse"}
# 默认 happyhorse i2v model on DashScope MaaS
MODEL=${3:-"happyhorse-i2v"}
N_PROC=${4:-2}
DATA_ROOT=${5:-"./data"}
DURATION=${6:-5}
N_ATTEMPTS=${7:-1}

# happyhorse 也走 dashscope 通路，同一个 key
API_KEY="${DASHSCOPE_API_KEY:-}"

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
echo "🚀 Starting HappyHorse (dashscope) Inference"
echo "📍 Dataset:      $DATASET"
echo "📍 Test Name:    $TEST_NAME"
echo "📍 Model:        $MODEL"
echo "📄 Task List:    $GT_JSON"
echo "📂 Pred Root:    $PRED_ROOT"
echo "⚙️  Processes:    $N_PROC"
echo "------------------------------------------------"

# HappyHorse VideoSynthesis expects media[0].type='first_frame'
# (see inference/dashscope_extra_generator.py). run_api_parallel.py dispatches
# to that path via the model name.
python run_api_parallel.py \
    --n_proc "$N_PROC" \
    --model "$MODEL" \
    --gt_root "$GT_JSON" \
    --pred_root "$PRED_ROOT" \
    --api_key "$API_KEY" \
    --duration "$DURATION" \
    --n_attempts "$N_ATTEMPTS"

echo "------------------------------------------------"
echo "✅ Inference dispatched to: $PRED_ROOT"
echo "------------------------------------------------"
