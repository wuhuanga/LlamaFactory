#!/bin/bash
# ============================================================
# Phase 1: OOD-Aware Self-Distillation 可行性验证 - 一键运行
# 用法:
#   bash run_all.sh              # 全部流程
#   bash run_all.sh prepare      # 只准备 D_KP 数据
#   bash run_all.sh train_bl     # 只训练 G2 baseline
#   bash run_all.sh train_kp     # 只训练 G3 SFT+KP
#   bash run_all.sh eval_pre     # 评测 G1 (Pre-SFT)
#   bash run_all.sh eval_bl      # 评测 G2 (Standard SFT)
#   bash run_all.sh eval_kp      # 评测 G3 (SFT+KP)
#   bash run_all.sh aggregate    # 汇总比较
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

STEP="${1:-all}"

# GPU 配置
export CUDA_VISIBLE_DEVICES=1,2
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

# ---- Step 0: 准备 D_KP 数据 ----
if [[ "$STEP" == "all" || "$STEP" == "prepare" ]]; then
    log "Step 0: Preparing D_KP (Wikipedia prompts)"
    python prepare_dkp.py \
        --num_prompts 10000 \
        --min_words 30 \
        --max_words 50 \
        --model_path /data1/guest/LlamaFactory/models/Llama-3.1-8B \
        2>&1 | tee outputs/prepare_dkp.log
fi

# ---- Step 1: 训练 G2 Baseline (Standard SFT, 同超参) ----
if [[ "$STEP" == "all" || "$STEP" == "train_bl" ]]; then
    log "Step 1: Training G2 Baseline (Standard SFT, ${NUM_GPUS} GPUs)"
    cd "$REPO_ROOT"
    llamafactory-cli train "$SCRIPT_DIR/sft_baseline.yaml" 2>&1 | tee "$SCRIPT_DIR/outputs/train_baseline.log"
    cd "$SCRIPT_DIR"
fi

# ---- Step 2: 训练 G3 SFT+KP (主方法) ----
if [[ "$STEP" == "all" || "$STEP" == "train_kp" ]]; then
    log "Step 2: Training G3 SFT+KP (OOD Self-Distillation, ${NUM_GPUS} GPUs)"
    cd "$REPO_ROOT"
    llamafactory-cli train "$SCRIPT_DIR/sft_kp.yaml" 2>&1 | tee "$SCRIPT_DIR/outputs/train_kp.log"
    cd "$SCRIPT_DIR"
fi

# ---- Step 3: 评测 ----
if [[ "$STEP" == "all" || "$STEP" == "eval_pre" ]]; then
    log "Step 3a: Evaluating G1 (Pre-SFT)"
    python evaluate.py --stage pre 2>&1 | tee outputs/eval_pre.log
fi

if [[ "$STEP" == "all" || "$STEP" == "eval_bl" ]]; then
    log "Step 3b: Evaluating G2 (Standard SFT)"
    python evaluate.py --stage baseline 2>&1 | tee outputs/eval_baseline.log
fi

if [[ "$STEP" == "all" || "$STEP" == "eval_kp" ]]; then
    log "Step 3c: Evaluating G3 (SFT+KP)"
    python evaluate.py --stage kp 2>&1 | tee outputs/eval_kp.log
fi

# ---- Step 4: 汇总 ----
if [[ "$STEP" == "all" || "$STEP" == "aggregate" ]]; then
    log "Step 4: Aggregating Results"
    python aggregate.py
fi

log "All done!"
