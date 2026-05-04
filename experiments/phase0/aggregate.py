"""
Phase 0: 汇总评测结果，生成 results.csv 和 results.md
用法: python aggregate.py [--config config.yaml]
"""
import argparse
import csv
import json
import logging
from pathlib import Path

from utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# MMLU 中需要排除的 medical 子类
MEDICAL_SUBS = {
    "clinical_knowledge",
    "college_medicine",
    "professional_medicine",
    "medical_genetics",
    "anatomy",
}


def find_result_file(output_dir, prefix):
    """在 lm-eval 输出目录中找到结果 JSON"""
    base = Path(output_dir) / prefix
    # lm-eval 输出格式: eval_pre/<model_name>/results_<timestamp>.json
    candidates = list(base.rglob("results_*.json"))
    if not candidates:
        candidates = list(base.rglob("results.json"))
    if candidates:
        # 取最新的
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]
    # 也可能直接就是文件
    if base.with_suffix(".json").exists():
        return base.with_suffix(".json")
    return None


def extract_scores(result_file):
    """从 lm-eval 输出中提取各 task 的分数"""
    with open(result_file) as f:
        data = json.load(f)

    results = data.get("results", {})
    scores = {}

    # --- MedMCQA ---
    if "medmcqa" in results:
        acc = results["medmcqa"].get("acc,none") or results["medmcqa"].get("acc")
        scores["medmcqa"] = acc

    # --- MMLU: 分 medical 和 non-medical ---
    mmlu_medical_accs = []
    mmlu_non_medical_accs = []
    for key, val in results.items():
        # mmlu 子任务格式: mmlu_xxx 或 hendrycksTest-xxx
        sub = None
        if key.startswith("mmlu_"):
            sub = key[5:]
        elif key.startswith("hendrycksTest-"):
            sub = key[len("hendrycksTest-"):]

        if sub is None:
            continue
        # 跳过聚合 key
        if sub in ("", "abstract_algebra") and key == "mmlu":
            continue

        acc = val.get("acc,none") or val.get("acc")
        if acc is None:
            continue

        if sub in MEDICAL_SUBS:
            mmlu_medical_accs.append(acc)
        else:
            mmlu_non_medical_accs.append(acc)

    if mmlu_non_medical_accs:
        scores["mmlu_non_medical"] = sum(mmlu_non_medical_accs) / len(mmlu_non_medical_accs)
    if mmlu_medical_accs:
        scores["mmlu_medical"] = sum(mmlu_medical_accs) / len(mmlu_medical_accs)

    # 如果子任务没拆开，用总 mmlu 分数作为近似
    if not mmlu_non_medical_accs and "mmlu" in results:
        acc = results["mmlu"].get("acc,none") or results["mmlu"].get("acc")
        scores["mmlu_non_medical"] = acc
        logger.warning("MMLU subtask scores not found, using aggregate score as approximation")

    # --- TriviaQA ---
    if "triviaqa" in results:
        scores["triviaqa"] = (
            results["triviaqa"].get("exact_match,remove_whitespace")
            or results["triviaqa"].get("exact_match,none")
            or results["triviaqa"].get("em,none")
            or results["triviaqa"].get("acc,none")
            or results["triviaqa"].get("acc")
        )

    # --- TruthfulQA MC2 ---
    if "truthfulqa_mc2" in results:
        scores["truthfulqa_mc2"] = (
            results["truthfulqa_mc2"].get("acc,none")
            or results["truthfulqa_mc2"].get("acc")
        )

    # --- TruthfulQA MC1 ---
    if "truthfulqa_mc1" in results:
        scores["truthfulqa_mc1"] = (
            results["truthfulqa_mc1"].get("acc,none")
            or results["truthfulqa_mc1"].get("acc")
        )

    # --- Memo Trap ---
    if "inverse_scaling_memo_trap" in results:
        scores["memo_trap"] = (
            results["inverse_scaling_memo_trap"].get("acc,none")
            or results["inverse_scaling_memo_trap"].get("acc")
            or results["inverse_scaling_memo_trap"].get("acc_norm,none")
        )

    return scores


def extract_popqa_score(output_dir, prefix):
    """提取 PopQA 分数"""
    popqa_path = Path(output_dir) / (prefix + "_popqa")

    # 可能是文件（无扩展名或 .json）或目录
    candidates = []
    if popqa_path.is_dir():
        candidates = list(popqa_path.rglob("results*.json"))
    elif popqa_path.is_file():
        candidates = [popqa_path]
    elif popqa_path.with_suffix(".json").is_file():
        candidates = [popqa_path.with_suffix(".json")]

    for p in candidates:
        try:
            with open(p) as f:
                data = json.load(f)
            r = data.get("results", {}).get("popqa", data)
            return r.get("exact_match") or r.get("exact_match,none") or r.get("f1") or r.get("acc")
        except (json.JSONDecodeError, KeyError):
            continue
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = cfg["paths"]["output_dir"]

    # 读取 pre/post 结果
    pre_file = find_result_file(output_dir, "eval_pre")
    post_file = find_result_file(output_dir, "eval_post")

    if not pre_file:
        logger.error(f"Pre-SFT results not found in {output_dir}/eval_pre*/")
        return
    if not post_file:
        logger.error(f"Post-SFT results not found in {output_dir}/eval_post*/")
        return

    logger.info(f"Pre-SFT results: {pre_file}")
    logger.info(f"Post-SFT results: {post_file}")

    pre_scores = extract_scores(pre_file)
    post_scores = extract_scores(post_file)

    # PopQA
    pre_popqa = extract_popqa_score(output_dir, "eval_pre")
    post_popqa = extract_popqa_score(output_dir, "eval_post")
    if pre_popqa is not None:
        pre_scores["popqa"] = pre_popqa
    if post_popqa is not None:
        post_scores["popqa"] = post_popqa

    # 构建结果表
    BENCHMARKS = [
        ("medmcqa", "ID"),
        ("mmlu_non_medical", "OOD_subject"),
        ("mmlu_medical", "near_ID (ref)"),
        ("popqa", "OOD_facts"),
        ("triviaqa", "OOD_facts"),
        ("truthfulqa_mc2", "OOD_hallucination"),
        ("truthfulqa_mc1", "OOD_hallucination"),
        ("memo_trap", "OOD_hallucination"),
    ]

    rows = []
    for bm, layer in BENCHMARKS:
        pre = pre_scores.get(bm)
        post = post_scores.get(bm)
        delta = (post - pre) if (pre is not None and post is not None) else None
        rows.append({
            "benchmark": bm,
            "layer": layer,
            "pre_sft": f"{pre:.4f}" if pre is not None else "N/A",
            "post_sft": f"{post:.4f}" if post is not None else "N/A",
            "delta": f"{delta:+.4f}" if delta is not None else "N/A",
        })

    # 写 CSV
    csv_path = Path(output_dir) / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["benchmark", "layer", "pre_sft", "post_sft", "delta"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"CSV saved to {csv_path}")

    # 写 Markdown
    md_path = Path(output_dir) / "results.md"
    with open(md_path, "w") as f:
        f.write("# Phase 0: SFT Forgetting Baseline Results\n\n")
        f.write("| Benchmark | Layer | Pre-SFT | Post-SFT | Delta |\n")
        f.write("|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['benchmark']} | {r['layer']} | {r['pre_sft']} | {r['post_sft']} | {r['delta']} |\n")

        f.write(f"\n## Config\n\n")
        f.write(f"- Base model: `{cfg['paths']['base_model']}`\n")
        n_samples = cfg['data'].get('train_samples') or "full"
        f.write(f"- Training data: `{cfg['data']['dataset']}` ({n_samples} samples)\n")
        f.write(f"- Epochs: {cfg['training']['epochs']}\n")
        f.write(f"- LR: {cfg['training']['learning_rate']}\n")
        f.write(f"- Effective batch size: {cfg['training']['per_device_batch_size'] * cfg['training']['gradient_accumulation_steps']}\n")
    logger.info(f"Markdown saved to {md_path}")

    # 打印结果
    print("\n" + "=" * 70)
    print("Phase 0 Results Summary")
    print("=" * 70)
    print(f"{'Benchmark':<25} {'Layer':<20} {'Pre':>8} {'Post':>8} {'Delta':>8}")
    print("-" * 70)
    for r in rows:
        print(f"{r['benchmark']:<25} {r['layer']:<20} {r['pre_sft']:>8} {r['post_sft']:>8} {r['delta']:>8}")
    print("=" * 70)


if __name__ == "__main__":
    main()
