"""共享工具函数"""
import os
import yaml
import json
import csv
from pathlib import Path


def load_config(config_path=None):
    """加载 YAML 配置文件"""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_gpu(cfg):
    """设置 GPU 可见性"""
    gpu_id = cfg.get("gpu", 0)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def ensure_dirs(cfg):
    """创建必要的输出目录"""
    for key in ["output_model", "output_dir", "cache_dir"]:
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


def save_json(data, path):
    """保存 JSON 文件"""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path):
    """加载 JSON 文件"""
    with open(path, "r") as f:
        return json.load(f)


COP_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


def format_medmcqa(example):
    """将 MedMCQA 样本格式化为 prompt + completion（对齐 lm-eval 格式）"""
    letter = COP_MAP.get(example["cop"], "A")
    # 使用和 lm-eval medmcqa 一致的格式
    options = (
        f"A. {example['opa']}\n"
        f"B. {example['opb']}\n"
        f"C. {example['opc']}\n"
        f"D. {example['opd']}"
    )
    prompt = f"Question: {example['question']}\nChoices:\n{options}\nAnswer:"

    exp = example.get("exp") or ""
    if exp.strip():
        completion = f" {letter}. {exp.strip()}"
    else:
        completion = f" {letter}"

    return {"prompt": prompt, "completion": completion}
