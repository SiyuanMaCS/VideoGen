#!/bin/bash
# Cosmos3 环境配置 & 模型下载
# 直接复用 /mnt/bn/embodied-lf/yyq/cosmos-framework/.venv 环境
set -e

COSMOS3_ROOT="/mnt/bn/embodied-lf/yyq/cosmos-framework"
COSMOS3_VENV="${COSMOS3_ROOT}/.venv"
HF_TOKEN="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"

echo "=== Cosmos3 Setup ==="
echo "Framework: ${COSMOS3_ROOT}"
echo "Python:    ${COSMOS3_VENV}/bin/python"
echo ""

# 1. 验证环境
if [ ! -f "${COSMOS3_VENV}/bin/python" ]; then
    echo "ERROR: cosmos-framework venv not found at ${COSMOS3_VENV}"
    echo "Need to install from scratch. Run:"
    echo "  cd ${COSMOS3_ROOT}"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  source \$HOME/.local/bin/env"
    echo "  uv sync --all-extras --group=cu128-train"
    exit 1
fi

echo "[OK] Python: $(${COSMOS3_VENV}/bin/python --version)"

# 2. 设置 HF Token（模型会从 HuggingFace 自动下载）
export HF_TOKEN="${HF_TOKEN}"
export HF_HOME="/mnt/bn/embodied-lf/masiyuan/.cache/huggingface"
mkdir -p "${HF_HOME}"

# 3. 预下载 Cosmos3-Nano checkpoint (约 18GB)
echo ""
echo "=== Downloading Cosmos3-Nano checkpoint ==="
${COSMOS3_VENV}/bin/python -c "
import os
os.environ['HF_TOKEN'] = '${HF_TOKEN}'
os.environ['HF_HOME'] = '${HF_HOME}'
from huggingface_hub import snapshot_download
print('Downloading Cosmos3-Nano...')
path = snapshot_download('nvidia/Cosmos3-Nano', token='${HF_TOKEN}', cache_dir='${HF_HOME}')
print(f'Done: {path}')
"

# 4. 预下载 Cosmos3-Super checkpoint (约 64GB)
echo ""
echo "=== Downloading Cosmos3-Super checkpoint ==="
${COSMOS3_VENV}/bin/python -c "
import os
os.environ['HF_TOKEN'] = '${HF_TOKEN}'
os.environ['HF_HOME'] = '${HF_HOME}'
from huggingface_hub import snapshot_download
print('Downloading Cosmos3-Super...')
path = snapshot_download('nvidia/Cosmos3-Super', token='${HF_TOKEN}', cache_dir='${HF_HOME}')
print(f'Done: {path}')
"

echo ""
echo "=== Setup Complete ==="
echo "Nano & Super checkpoints cached at: ${HF_HOME}"
echo ""
echo "To run inference:"
echo "  export PYTHON_BIN=${COSMOS3_VENV}/bin/python"
echo "  export HF_HOME=${HF_HOME}"
echo "  export HF_TOKEN=${HF_TOKEN}"
echo "  bash scripts_inference/run_cosmos3.sh <DATASET> cosmos3_nano Cosmos3-Nano 8 ./data 0,1,2,3,4,5,6,7 1 0.8"
