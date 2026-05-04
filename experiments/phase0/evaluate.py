"""
Phase 0: 评测脚本 (基于 lm-evaluation-harness)
用法:
  python evaluate.py --stage pre   # 评测原始模型
  python evaluate.py --stage post  # 评测 SFT 后模型
  python evaluate.py --stage both  # 依次评测两个
"""
import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

from utils import load_config, ensure_dirs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# lm-eval 支持的 task 列表（popqa 可能不支持，单独处理）
STANDARD_TASKS = ["medmcqa", "mmlu", "triviaqa", "truthfulqa_mc2", "truthfulqa_mc1", "inverse_scaling_memo_trap"]


def get_eval_gpu(cfg):
    """获取评测用的 GPU，CUDA_VISIBLE_DEVICES 已在 shell 中设置，映射后始终用 0"""
    return "0"


def run_lm_eval(model_path, tasks, output_file, cfg):
    """调用 lm_eval 进行评测"""
    gpu_id = get_eval_gpu(cfg)
    batch_size = cfg["eval"].get("batch_size", "auto")
    num_fewshot = cfg["eval"]["num_fewshot"]

    task_str = ",".join(tasks)
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},dtype=bfloat16",
        "--tasks", task_str,
        "--num_fewshot", str(num_fewshot),
        "--batch_size", str(batch_size),
        "--output_path", str(output_file),
        "--device", f"cuda:{gpu_id}",
        "--log_samples",
    ]

    logger.info(f"Running lm-eval: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        logger.error(f"lm-eval failed with return code {result.returncode}")
        sys.exit(1)
    logger.info(f"Results saved to {output_file}")


def run_popqa_eval(model_path, output_file, cfg):
    """PopQA 评测 — 如果 lm-eval 不支持则用自定义脚本"""
    gpu_id = get_eval_gpu(cfg)

    # 先尝试用 lm-eval 跑
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},dtype=bfloat16",
        "--tasks", "popqa",
        "--num_fewshot", "0",
        "--batch_size", str(cfg["eval"].get("batch_size", "auto")),
        "--output_path", str(output_file),
        "--device", f"cuda:{gpu_id}",
        "--log_samples",
    ]

    logger.info("Trying lm-eval with popqa task...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        logger.info("PopQA via lm-eval succeeded.")
        return True

    # lm-eval 不支持 popqa，使用自定义评测
    logger.warning("PopQA not in lm-eval, falling back to custom evaluation...")
    try:
        from eval_popqa import evaluate_popqa
        scores = evaluate_popqa(model_path, gpu_id)
        with open(output_file, "w") as f:
            json.dump({"results": {"popqa": scores}}, f, indent=2)
        return True
    except ImportError:
        logger.error("eval_popqa.py not found. Skipping PopQA.")
        return False


def evaluate(stage, cfg):
    """执行指定阶段的评测"""
    paths = cfg["paths"]
    output_dir = Path(paths["output_dir"])

    if stage == "pre":
        model_path = paths["base_model"]
        out_prefix = output_dir / "eval_pre"
    else:
        model_path = paths["output_model"]
        out_prefix = output_dir / "eval_post"

    logger.info(f"=== Evaluating [{stage}] model: {model_path} ===")

    # 标准 tasks
    run_lm_eval(model_path, STANDARD_TASKS, str(out_prefix), cfg)

    # PopQA 单独跑
    popqa_out = str(out_prefix) + "_popqa"
    run_popqa_eval(model_path, popqa_out, cfg)

    logger.info(f"=== [{stage}] evaluation complete ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pre", "post", "both"], required=True)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    if args.stage in ("pre", "both"):
        evaluate("pre", cfg)
    if args.stage in ("post", "both"):
        evaluate("post", cfg)


if __name__ == "__main__":
    main()
