---
library_name: transformers
license: other
base_model: /data1/guest/LlamaFactory/models/Llama-3.1-8B
tags:
- llama-factory
- full
- generated_from_trainer
- sft
- trl
model-index:
- name: sft_model
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# sft_model

This model is a fine-tuned version of [/data1/guest/LlamaFactory/models/Llama-3.1-8B](https://huggingface.co//data1/guest/LlamaFactory/models/Llama-3.1-8B) on the medmcqa_sft dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-06
- train_batch_size: 4
- eval_batch_size: 8
- seed: 42
- distributed_type: multi-GPU
- num_devices: 2
- gradient_accumulation_steps: 4
- total_train_batch_size: 32
- total_eval_batch_size: 16
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 0.03
- num_epochs: 1.0

### Training results



### Framework versions

- Transformers 5.2.0
- Pytorch 2.5.1+cu124
- Datasets 4.0.0
- Tokenizers 0.22.2
