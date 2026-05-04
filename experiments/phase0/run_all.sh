#!/bin/bash
# ============================================================
# Phase 0: 一键运行脚本
# 用法:
#   bash run_all.sh            # 全部流程
#   bash run_all.sh prepare    # 只跑数据准备
#   bash run_all.sh eval_pre   # 只跑 pre-SFT 评测
#   bash run_all.sh train      # 只跑训练
#   bash run_all.sh eval_post  # 只跑 post-SFT 评测
#   bash run_all.sh aggregate  # 只跑汇总
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

STEP="${1:-all}"

# GPU 配置
export CUDA_VISIBLE_DEVICES=2,3
NUM_GPUS=2

log() {
    echo ""
    echo "========================================"
    echo "  $1"
    echo "  $(date)"
    echo "========================================"
    echo ""
}

mkdir -p outputs

# Step 0: 数据准备（生成 LlamaFactory 格式数据）
if [[ "$STEP" == "all" || "$STEP" == "prepare" ]]; then
    log "Step 0: Preparing MedMCQA data for LlamaFactory"
    python prepare_data.py 2>&1 | tee outputs/prepare_data.log
fi

# Step 1: Pre-SFT 评测（单卡）
if [[ "$STEP" == "all" || "$STEP" == "eval_pre" ]]; then
    log "Step 1/4: Pre-SFT Evaluation"
    python evaluate.py --stage pre 2>&1 | tee outputs/eval_pre.log
fi

# Step 2: SFT 训练（LlamaFactory，多卡）
if [[ "$STEP" == "all" || "$STEP" == "train" ]]; then
    log "Step 2/4: SFT Training via LlamaFactory (${NUM_GPUS} GPUs)"
    cd "$REPO_ROOT"
    llamafactory-cli train "$SCRIPT_DIR/sft.yaml" 2>&1 | tee "$SCRIPT_DIR/outputs/train.log"
    cd "$SCRIPT_DIR"
fi

# Step 3: Post-SFT 评测（单卡）
if [[ "$STEP" == "all" || "$STEP" == "eval_post" ]]; then
    log "Step 3/4: Post-SFT Evaluation"
    python evaluate.py --stage post 2>&1 | tee outputs/eval_post.log
fi

# Step 4: 汇总结果
if [[ "$STEP" == "all" || "$STEP" == "aggregate" ]]; then
    log "Step 4/4: Aggregating Results"
    python aggregate.py
fi

log "All done!"
