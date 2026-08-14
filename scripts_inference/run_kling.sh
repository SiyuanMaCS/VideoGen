#!/bin/bash
set -euo pipefail

# 随机生成端口（保持与评价脚本格式对齐）
export MASTER_PORT=$(shuf -i 20000-65535 -n 1)

# ==============================================================================
# Kling API 并行推理自动化脚本 (dashscope)
# Usage: ./run_kling.sh <DATASET> <TEST_NAME> [MODEL] [N_PROC] [DATA_ROOT]
# Example: ./run_kling.sh common_set_50 kling_common50 "kling-v2.1" 4
# ==============================================================================

# --- 1. 参数获取与默认值 ---
DATASET=${1:-"debug"}
TEST_NAME=${2:-"kling"}
# 默认走 dashscope 的 Kling 通路（第三方 MaaS）。正式跑改成生产模型名。
MODEL=${3:-"kling-v1.6"}
N_PROC=${4:-2}
DATA_ROOT=${5:-"./data"}
DURATION=${6:-5}
N_ATTEMPTS=${7:-1}

# 你的 DashScope API KEY（siyuan：kling 走 dashscope 同一个 key）
API_KEY="${DASHSCOPE_API_KEY:-}"

# --- 2. 自动路径拼接 ---
GT_JSON="${DATA_ROOT}/${DATASET}/summary.json"
PRED_ROOT="${DATA_ROOT}/${DATASET}/generated_data/${TEST_NAME}"

# --- 3. 运行前的清理与准备 ---
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

# --- 4. 执行信息打印 ---
echo "------------------------------------------------"
echo "🚀 Starting Kling (dashscope) Inference"
echo "📍 Dataset:      $DATASET"
echo "📍 Test Name:    $TEST_NAME"
echo "📍 Model:        $MODEL"
echo "📄 Task List:    $GT_JSON"
echo "📂 Pred Root:    $PRED_ROOT"
echo "⚙️  Processes:    $N_PROC"
echo "------------------------------------------------"

# --- 5. 执行推理 ---
# Kling on DashScope 走 async VideoSynthesis pipeline，走 dashscope 通用 generator。
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
