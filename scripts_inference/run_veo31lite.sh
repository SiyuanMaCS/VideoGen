#!/bin/bash
set -euo pipefail

# 随机生成端口
export MASTER_PORT=$(shuf -i 20000-65535 -n 1)

# ==============================================================================
# Google Veo 3.1 并行推理自动化脚本
# Usage: ./scripts_inference/run_veo31lite.sh <DATASET> <TEST_NAME> [MODEL] [N_PROC] [DATA_ROOT]
# Example: ./scripts_inference/run_veo31lite.sh S1-2 veo_lite_test "veo-3.1-lite-generate-preview" 2
# ==============================================================================

# --- 1. 参数获取与默认值 ---
DATASET=${1:-"debug"}
TEST_NAME=${2:-"veo31_lite"}
# 默认模型切换为 Veo 3.1 Lite (for debug) 正式评测请用 "veo-3.1-generate-preview"
MODEL=${3:-"veo-3.1-lite-generate-preview"} 
N_PROC=${4:-2}     
DATA_ROOT=${5:-"./data"}
DURATION=${6:-5}   # 视频时长默认 5s 正式评测用 10s
N_ATTEMPTS=${7:-1} # 每个视频的重试次数

# 你的 Google Cloud API KEY (Veo 只需要这一个 Key)
# 注意：如果 run_api_parallel.py 里还是用 --ak 传参，请保持一致
VEO_API_KEY=""

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

mkdir -p "$PRED_ROOT"

# --- 4. 执行信息打印 ---
echo "------------------------------------------------"
echo "🚀 Starting Veo 3.1 inference"
echo "📍 Dataset:      $DATASET"
echo "📍 Test Name:    $TEST_NAME"
echo "📍 Model:        $MODEL"
echo "📄 Task List:    $GT_JSON"
echo "📂 Pred Root:    $PRED_ROOT"
echo "⚙️  Processes:    $N_PROC"
echo "⏱️  Duration:     $DURATION s"
echo "------------------------------------------------"

# --- 5. 执行推理 ---
# 注意：Veo 通常只用到 --ak 或者 --api_key 参数中的一个
# 这里我们将 VEO_API_KEY 传给脚本
python run_api_parallel.py \
    --n_proc "$N_PROC" \
    --model "$MODEL" \
    --gt_root "$GT_JSON" \
    --pred_root "$PRED_ROOT" \
    --api_key "$VEO_API_KEY" \
    --duration "$DURATION" \
    --n_attempts "$N_ATTEMPTS"

echo "------------------------------------------------"
echo "✅ Veo Inference dispatched to: $PRED_ROOT"
echo "📊 Ready for evaluation."
echo "------------------------------------------------"