"""
Phase 0: SFT 训练脚本
用法:
  单卡: python train.py
  多卡: deepspeed --num_gpus N train.py --deepspeed ds_config.json
"""
import argparse
import logging
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

from utils import load_config, ensure_dirs, format_medmcqa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)  # DeepSpeed 会传这个参数
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    paths = cfg["paths"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    # ---- 加载数据 ----
    logger.info("Loading MedMCQA dataset...")
    ds = load_dataset(data_cfg["dataset"], split="train", cache_dir=paths["cache_dir"])
    ds = ds.shuffle(seed=data_cfg["seed"]).select(range(data_cfg["train_samples"]))
    ds = ds.map(format_medmcqa, remove_columns=ds.column_names)
    logger.info(f"Training samples: {len(ds)}")
    logger.info(f"Sample prompt:\n{ds[0]['prompt'][:300]}")
    logger.info(f"Sample completion:\n{ds[0]['completion']}")

    # ---- 加载模型 & tokenizer ----
    logger.info(f"Loading model from {paths['base_model']}...")
    tokenizer = AutoTokenizer.from_pretrained(paths["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        paths["base_model"],
        torch_dtype="bfloat16" if train_cfg["bf16"] else "float32",
    )

    # ---- 训练参数 ----
    output_dir = paths["output_model"]

    # 计算 warmup_steps: total_steps * warmup_ratio
    total_samples = len(ds)
    steps_per_epoch = total_samples // (train_cfg["per_device_batch_size"] * train_cfg["gradient_accumulation_steps"])
    total_steps = steps_per_epoch * train_cfg["epochs"]
    warmup_steps = int(total_steps * train_cfg["warmup_ratio"])

    # DeepSpeed 配置路径（绝对路径）
    ds_config = None
    if args.deepspeed:
        ds_config = str(Path(args.deepspeed).resolve())

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg["epochs"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler"],
        warmup_steps=warmup_steps,
        per_device_train_batch_size=train_cfg["per_device_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        max_length=train_cfg["max_seq_length"],
        bf16=train_cfg["bf16"],
        gradient_checkpointing=True,
        optim=train_cfg["optimizer"],
        seed=train_cfg["seed"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        report_to="none",
        log_level="info",
        deepspeed=ds_config,
    )

    # ---- 训练 ----
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    logger.info("Starting SFT training...")
    train_result = trainer.train()
    logger.info(f"Training finished. Final loss: {train_result.training_loss:.4f}")

    # ---- 保存 ----
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")

    # 保存训练指标
    metrics = {
        "train_loss": train_result.training_loss,
        "train_runtime": train_result.metrics.get("train_runtime"),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
    }
    from utils import save_json
    save_json(metrics, f"{paths['output_dir']}/train_metrics.json")
    logger.info("Done.")


if __name__ == "__main__":
    main()
