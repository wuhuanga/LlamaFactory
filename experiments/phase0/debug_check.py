"""检查训练数据和模型是否有问题"""
import json
from collections import Counter
from datasets import load_dataset

# 1. 检查 cop 字段分布
print("=" * 50)
print("1. Checking cop field distribution")
print("=" * 50)
ds = load_dataset("openlifescienceai/medmcqa", split="train")
ds_sub = ds.shuffle(seed=42).select(range(20000))

cop_values = Counter(ds_sub["cop"])
print(f"cop distribution: {dict(sorted(cop_values.items()))}")

# 检查异常值
bad_cops = [v for v in ds_sub["cop"] if v not in (0, 1, 2, 3)]
print(f"Invalid cop values (not 0-3): {len(bad_cops)}")
if bad_cops:
    print(f"  Examples: {bad_cops[:10]}")

# 2. 检查格式化后的样本
print()
print("=" * 50)
print("2. Sample formatted data")
print("=" * 50)
from utils import format_medmcqa, COP_MAP
for i in [0, 100, 500]:
    ex = ds_sub[i]
    formatted = format_medmcqa(ex)
    print(f"\n--- Sample {i} (cop={ex['cop']}, correct={COP_MAP.get(ex['cop'], '?')}) ---")
    print(formatted["text"][:300])

# 3. 检查 sft_model 目录
print()
print("=" * 50)
print("3. Checking sft_model directory")
print("=" * 50)
import os
sft_dir = "/data1/guest/LlamaFactory/experiments/phase0/sft_model"
files = os.listdir(sft_dir)
print(f"Files in sft_model/: {files}")
has_model = any(f.endswith(".safetensors") for f in files)
print(f"Has model weights in root: {has_model}")

# 4. 检查 lm-eval 的 medmcqa 评测方式
print()
print("=" * 50)
print("4. Checking lm-eval medmcqa format")
print("=" * 50)
try:
    import lm_eval.tasks
    task_manager = lm_eval.tasks.TaskManager()
    task_dict = lm_eval.tasks.get_task_dict(["medmcqa"], task_manager)
    task = task_dict["medmcqa"]
    # 看看 lm-eval 用什么 prompt
    docs = task.test_docs() if hasattr(task, "test_docs") else task.validation_docs()
    doc = list(docs)[0]
    print(f"Doc keys: {doc.keys() if isinstance(doc, dict) else 'not dict'}")
    if hasattr(task, "doc_to_text"):
        print(f"Prompt: {task.doc_to_text(doc)[:300]}")
    if hasattr(task, "doc_to_target"):
        print(f"Target: {task.doc_to_target(doc)}")
except Exception as e:
    print(f"Could not inspect lm-eval task: {e}")
