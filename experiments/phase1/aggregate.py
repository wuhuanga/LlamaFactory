"""
Phase 1: 汇总评测结果，对比 G1(Pre-SFT), G2(Standard SFT), G3(SFT+KP)
用法: python aggregate.py [--config config.yaml]
"""
import argparse
import csv
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "phase0"))
from utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MEDICAL_SUBS = {
    "clinical_knowledge", "college_medicine", "professional_medicine",
    "medical_genetics", "anatomy",
}


def find_result_file(output_dir, prefix):
    base = Path(output_dir) / prefix
    candidates = list(base.rglob("results_*.json"))
    if not candidates:
        candidates = list(base.rglob("results.json"))
    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]
    if base.with_suffix(".json").exists():
        return base.with_suffix(".json")
    return None


def extract_scores(result_file):
    with open(result_file) as f:
        data = json.load(f)
    results = data.get("results", {})
    scores = {}

    if "medmcqa" in results:
        scores["medmcqa"] = results["medmcqa"].get("acc,none") or results["medmcqa"].get("acc")

    mmlu_medical_accs, mmlu_non_medical_accs = [], []
    for key, val in results.items():
        sub = None
        if key.startswith("mmlu_"):
            sub = key[5:]
        elif key.startswith("hendrycksTest-"):
            sub = key[len("hendrycksTest-"):]
        if sub is None:
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

    if "triviaqa" in results:
        scores["triviaqa"] = (
            results["triviaqa"].get("exact_match,remove_whitespace")
            or results["triviaqa"].get("exact_match,none")
            or results["triviaqa"].get("em,none")
            or results["triviaqa"].get("acc,none")
            or results["triviaqa"].get("acc")
        )

    if "truthfulqa_mc2" in results:
        scores["truthfulqa_mc2"] = results["truthfulqa_mc2"].get("acc,none") or results["truthfulqa_mc2"].get("acc")

    return scores


def extract_popqa_score(output_dir, prefix):
    popqa_path = Path(output_dir) / (prefix + "_popqa")
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


def fmt(val):
    return f"{val:.4f}" if val is not None else "N/A"


def fmt_delta(val):
    return f"{val:+.4f}" if val is not None else "N/A"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config_path = args.config or str(Path(__file__).parent / "config.yaml")
    cfg = load_config(config_path)
    output_dir = cfg["paths"]["output_dir"]

    # Collect scores for each group
    groups = {
        "G1 (Pre-SFT)": "eval_pre",
        "G2 (Std SFT)": "eval_baseline",
        "G3 (SFT+KP)": "eval_kp",
    }

    all_scores = {}
    for name, prefix in groups.items():
        rf = find_result_file(output_dir, prefix)
        if rf:
            scores = extract_scores(rf)
            popqa = extract_popqa_score(output_dir, prefix)
            if popqa is not None:
                scores["popqa"] = popqa
            all_scores[name] = scores
            logger.info(f"{name}: loaded from {rf}")
        else:
            all_scores[name] = {}
            logger.warning(f"{name}: results not found at {output_dir}/{prefix}")

    BENCHMARKS = [
        ("medmcqa", "ID"),
        ("mmlu_medical", "near-ID"),
        ("mmlu_non_medical", "OOD-subject"),
        ("popqa", "OOD-facts"),
        ("triviaqa", "OOD-facts"),
        ("truthfulqa_mc2", "OOD-halluc"),
    ]

    pre = all_scores.get("G1 (Pre-SFT)", {})

    # Print comparison table
    header = f"{'Benchmark':<20} {'Layer':<12}"
    for name in groups:
        header += f" {name:>14}"
    header += f" {'G2-G1':>8} {'G3-G1':>8} {'G3-G2':>8}"
    print("\n" + "=" * len(header))
    print("Phase 1: OOD Self-Distillation Results")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    rows_csv = []
    for bm, layer in BENCHMARKS:
        vals = {name: all_scores.get(name, {}).get(bm) for name in groups}
        g1 = vals.get("G1 (Pre-SFT)")
        g2 = vals.get("G2 (Std SFT)")
        g3 = vals.get("G3 (SFT+KP)")

        line = f"{bm:<20} {layer:<12}"
        for name in groups:
            line += f" {fmt(vals[name]):>14}"
        d_g2 = (g2 - g1) if (g2 is not None and g1 is not None) else None
        d_g3 = (g3 - g1) if (g3 is not None and g1 is not None) else None
        d_g3g2 = (g3 - g2) if (g3 is not None and g2 is not None) else None
        line += f" {fmt_delta(d_g2):>8} {fmt_delta(d_g3):>8} {fmt_delta(d_g3g2):>8}"
        print(line)

        rows_csv.append({
            "benchmark": bm, "layer": layer,
            "G1_pre": fmt(g1), "G2_baseline": fmt(g2), "G3_kp": fmt(g3),
            "delta_G2_G1": fmt_delta(d_g2), "delta_G3_G1": fmt_delta(d_g3), "delta_G3_G2": fmt_delta(d_g3g2),
        })

    print("=" * len(header))

    # Phase 1 judgment
    criteria = cfg.get("criteria", {})
    g1_medmcqa = pre.get("medmcqa")
    g3_medmcqa = all_scores.get("G3 (SFT+KP)", {}).get("medmcqa")
    g3_popqa = all_scores.get("G3 (SFT+KP)", {}).get("popqa")
    g3_triviaqa = all_scores.get("G3 (SFT+KP)", {}).get("triviaqa")

    print("\n--- Phase 1 Judgment ---")
    if g3_popqa is not None:
        print(f"PopQA (G3):    {g3_popqa:.4f}  (target >= {criteria.get('popqa_min', 0.10)})")
    if g3_triviaqa is not None:
        print(f"TriviaQA (G3): {g3_triviaqa:.4f}  (target >= {criteria.get('triviaqa_min', 0.55)})")
    if g3_medmcqa is not None and g1_medmcqa is not None:
        drop = g1_medmcqa - g3_medmcqa
        print(f"MedMCQA drop:  {drop:+.4f}  (target <= {criteria.get('medmcqa_max_drop', 0.01)})")

    # Save CSV
    csv_path = Path(output_dir) / "results_phase1.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
        writer.writeheader()
        writer.writerows(rows_csv)
    logger.info(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
