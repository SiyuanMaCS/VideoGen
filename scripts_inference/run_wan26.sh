#!/bin/bash
set -euo pipefail

# 随机生成端口（保持与评价脚本格式对齐）
export MASTER_PORT=$(shuf -i 20000-65535 -n 1)

# ==============================================================================
# Wan2.6 API 并行推理脚本 (目录对齐版)
# Usage: ./scripts_inference/run_wan26.sh <DATASET> <TEST_NAME> [MODEL] [N_PROC] [DATA_ROOT]
# Example: ./scripts_inference/run_wan26.sh open_x_embodiment wan26 "wan2.6-i2v" 4
# ==============================================================================

# --- 1. 参数获取与默认值 ---
DATASET=${1:-"debug"}
TEST_NAME=${2:-"wan26"}
MODEL=${3:-"wan2.6-i2v"}
N_PROC=${4:-4}
DATA_ROOT=${5:-"./data"}
DURATION=${6:-5}  # 视频时长默认 10s
N_ATTEMPTS=${7:-3} # 每个视频的重试次数，默认 3 次
# 你的 API KEY
API_KEY=""

# --- 2. 自动路径拼接 (核心：与数据集结构对齐) ---
# 推理输入：指向该数据集的描述文件
GT_JSON="${DATA_ROOT}/${DATASET}/summary.json"
# 推理输出：存放到数据集下的 generated_data/测试名 目录
PRED_ROOT="${DATA_ROOT}/${DATASET}/generated_data/${TEST_NAME}"

# --- 3. 运行前的清理与准备 ---
if [ -d "${PRED_ROOT}/temp_worker" ]; then
    echo "🧹 Cleaning up old temp workers..."
    rm -rf "${PRED_ROOT}/temp_worker"
fi

# 检查输入文件是否存在
if [ ! -f "$GT_JSON" ]; then
    echo "❌ Error: GT summary JSON not found at $GT_JSON"
    exit 1
fi

# 确保输出目录存在
mkdir -p "$PRED_ROOT"

# --- 4. 执行信息打印 ---
echo "------------------------------------------------"
echo "🚀 Starting Wan2.6 API inference"
echo "📍 Dataset:      $DATASET"
echo "📍 Test Name:    $TEST_NAME"
echo "📍 Model:        $MODEL"
echo "📄 Task List:    $GT_JSON"
echo "📂 Pred Root:    $PRED_ROOT"
echo "⚙️  Processes:    $N_PROC"
echo "------------------------------------------------"

# --- 5. 执行推理 ---
# 视频时长默认为 5s
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
echo "📊 Results will be ready for evaluation."
echo "------------------------------------------------"