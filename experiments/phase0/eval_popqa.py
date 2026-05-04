"""
PopQA 自定义评测脚本 (当 lm-eval 不支持 popqa 时的 fallback)
基于 akariasai/PopQA 数据集，使用 open-ended generation + EM/F1 评测
"""
import re
import string
from collections import Counter

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def normalize_answer(s):
    """标准化答案用于 EM/F1 计算"""
    s = s.lower()
    # 去掉标点
    s = "".join(ch for ch in s if ch not in string.punctuation)
    # 去掉冠词
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # 合并空格
    s = " ".join(s.split())
    return s


def exact_match(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_popqa(model_path, gpu_id=0, max_samples=1000, max_new_tokens=32):
    """评测 PopQA"""
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()

    ds = load_dataset("akariasai/PopQA", split="test")
    if len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))

    em_scores = []
    f1_scores = []

    for example in tqdm(ds, desc="PopQA eval"):
        question = example["question"]
        # 支持多个可能的答案
        possible_answers = [example["obj"]]
        if "possible_answers" in example and example["possible_answers"]:
            if isinstance(example["possible_answers"], list):
                possible_answers.extend(example["possible_answers"])

        prompt = f"Question: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        # 取第一行作为答案
        pred = generated.split("\n")[0].strip()

        best_em = max(exact_match(pred, ans) for ans in possible_answers)
        best_f1 = max(f1_score(pred, ans) for ans in possible_answers)
        em_scores.append(best_em)
        f1_scores.append(best_f1)

    results = {
        "exact_match": sum(em_scores) / len(em_scores),
        "f1": sum(f1_scores) / len(f1_scores),
        "num_samples": len(em_scores),
    }
    print(f"PopQA Results: EM={results['exact_match']:.4f}, F1={results['f1']:.4f}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=1000)
    args = parser.parse_args()
    evaluate_popqa(args.model_path, args.gpu, args.max_samples)
