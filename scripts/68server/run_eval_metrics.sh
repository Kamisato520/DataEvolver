#!/bin/bash
# Run eval metrics for rotation editing (68 server).
# Usage:
#   bash run_eval_metrics.sh                    # testset only (default)
#   bash run_eval_metrics.sh --eval both        # testset + benchmark
#   bash run_eval_metrics.sh --modes base ours  # specific modes only
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

WORKDIR="/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build"

# Python 环境（uv）
source "${WORKDIR}/.venv/bin/activate"
export PYTHONPATH="${WORKDIR}/DiffSynth-Studio:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# 安装评测依赖（首次运行时自动安装）
python -c "import lpips" 2>/dev/null || uv pip install lpips
python -c "import clip" 2>/dev/null || uv pip install git+https://github.com/openai/CLIP.git
python -c "import transformers" 2>/dev/null || uv pip install transformers
python -c "import skimage" 2>/dev/null || uv pip install scikit-image
python -c "import scipy" 2>/dev/null || uv pip install scipy

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python "${SCRIPT_DIR}/eval_metrics.py" --device cuda "$@"
