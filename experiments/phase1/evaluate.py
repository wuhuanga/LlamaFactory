"""
Phase 1: 评测脚本 (基于 lm-evaluation-harness)
支持评测 3 个模型: pre-SFT (G1), baseline SFT (G2), SFT+KP (G3)

用法:
  python evaluate.py --stage pre       # G1: 原始模型
  python evaluate.py --stage baseline  # G2: Standard SFT
  python evaluate.py --stage kp        # G3: SFT + L_KP
  python evaluate.py --stage all       # 全部
"""
import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "phase0"))
from utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STANDARD_TASKS = ["medmcqa", "mmlu", "triviaqa", "truthfulqa_mc2", "truthfulqa_mc1"]


def run_lm_eval(model_path, tasks, output_path, cfg):
    gpu_id = "0"
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
        "--output_path", str(output_path),
        "--device", f"cuda:{gpu_id}",
        "--log_samples",
    ]
    logger.info(f"Running lm-eval: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        logger.error(f"lm-eval failed with return code {result.returncode}")
        sys.exit(1)
    logger.info(f"Results saved to {output_path}")


def run_popqa_eval(model_path, output_path, cfg):
    gpu_id = "0"
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},dtype=bfloat16",
        "--tasks", "popqa",
        "--num_fewshot", "0",
        "--batch_size", str(cfg["eval"].get("batch_size", "auto")),
        "--output_path", str(output_path),
        "--device", f"cuda:{gpu_id}",
        "--log_samples",
    ]
    logger.info("Running lm-eval with popqa task...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        logger.info("PopQA via lm-eval succeeded.")
        return True

    logger.warning("PopQA not in lm-eval, falling back to custom evaluation...")
    try:
        phase0_dir = Path(__file__).parent.parent / "phase0"
        sys.path.insert(0, str(phase0_dir))
        from eval_popqa import evaluate_popqa
        scores = evaluate_popqa(model_path, gpu_id)
        Path(output_path).mkdir(parents=True, exist_ok=True)
        with open(Path(output_path) / "popqa_results.json", "w") as f:
            json.dump({"results": {"popqa": scores}}, f, indent=2)
        return True
    except ImportError:
        logger.error("eval_popqa.py not found. Skipping PopQA.")
        return False


STAGE_MAP = {
    "pre": ("base_model", "eval_pre"),
    "baseline": ("baseline_model", "eval_baseline"),
    "kp": ("kp_model", "eval_kp"),
}


def evaluate(stage, cfg):
    paths = cfg["paths"]
    output_dir = Path(paths["output_dir"])

    model_key, out_prefix = STAGE_MAP[stage]
    model_path = paths[model_key]

    logger.info(f"=== Evaluating [{stage}] model: {model_path} ===")

    run_lm_eval(model_path, STANDARD_TASKS, str(output_dir / out_prefix), cfg)
    run_popqa_eval(model_path, str(output_dir / out_prefix) + "_popqa", cfg)

    logger.info(f"=== [{stage}] evaluation complete ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pre", "baseline", "kp", "all"], required=True)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config_path = args.config or str(Path(__file__).parent / "config.yaml")
    cfg = load_config(config_path)
    Path(cfg["paths"]["output_dir"]).mkdir(parents=True, exist_ok=True)

    stages = ["pre", "baseline", "kp"] if args.stage == "all" else [args.stage]
    for s in stages:
        evaluate(s, cfg)


if __name__ == "__main__":
    main()
