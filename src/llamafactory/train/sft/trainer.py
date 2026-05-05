# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from functools import partial
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, patch_accelerator_for_fp8, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset as DatasetType
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments, TrainingArguments


class KPPromptDataset(Dataset):
    """Simple dataset holding pre-tokenized KP prompts for OOD knowledge preservation."""

    def __init__(self, input_ids_list: list[list[int]], attention_mask_list: list[list[int]]):
        self.input_ids_list = input_ids_list
        self.attention_mask_list = attention_mask_list

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids_list[idx],
            "attention_mask": self.attention_mask_list[idx],
        }


def _kp_collate_fn(batch: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    """Left-pad KP prompts for generation (shorter prompts get padding on the left)."""
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, attention_mask = [], []
    for x in batch:
        pad_len = max_len - len(x["input_ids"])
        input_ids.append([pad_token_id] * pad_len + x["input_ids"])
        attention_mask.append([0] * pad_len + x["attention_mask"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        ref_model: Optional["torch.nn.Module"] = None,
        kp_dataset: Optional["KPPromptDataset"] = None,
        **kwargs,
    ) -> None:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
        # Configure FP8 environment if enabled
        training_args: TrainingArguments = kwargs.get("args")
        if training_args.fp8:
            configure_fp8_environment(training_args)
            if getattr(training_args, "fp8_backend", "auto") == "te":
                patch_accelerator_for_fp8()

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        self.ref_model = ref_model

        if ref_model is not None:
            from trl.models.utils import prepare_deepspeed, prepare_fsdp

            if getattr(self.accelerator.state, "deepspeed_plugin", None) is not None:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif getattr(self.accelerator.state, "fsdp_plugin", None) is not None:
                if self.accelerator.is_fsdp2:
                    from accelerate.utils.fsdp_utils import fsdp2_prepare_model

                    self.ref_model = fsdp2_prepare_model(self.accelerator, self.ref_model)
                else:
                    self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        elif finetuning_args.use_eaft_loss:
            from ..trainer_utils import eaft_loss_func

            self.compute_loss_func = lambda outputs, labels, num_items_in_batch=None: eaft_loss_func(
                outputs, labels, num_items_in_batch, finetuning_args.eaft_alpha
            )
        elif finetuning_args.use_asft_loss:
            from ..trainer_utils import asft_loss_func

            self.compute_loss_func = partial(
                asft_loss_func,
                asft_alpha=finetuning_args.asft_alpha,
            )

        # --- OOD Knowledge Preservation (KP) setup ---
        self._kp_dataloader = None
        self._kp_iter = None
        self._kp_loss_sum = 0.0
        self._sft_loss_sum = 0.0
        self._kp_count = 0
        self._kp_step_counter = 0

        if finetuning_args.use_ood_kp_loss and kp_dataset is not None:
            pad_token_id = getattr(self.processing_class, "pad_token_id", 0) or 0

            from torch.utils.data.distributed import DistributedSampler

            if self.args.world_size > 1:
                kp_sampler = DistributedSampler(
                    kp_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    shuffle=True,
                )
            else:
                kp_sampler = RandomSampler(kp_dataset)

            self._kp_dataloader = DataLoader(
                kp_dataset,
                batch_size=self.args.per_device_train_batch_size,
                sampler=kp_sampler,
                collate_fn=partial(_kp_collate_fn, pad_token_id=pad_token_id),
                drop_last=True,
            )
            self._kp_iter = iter(self._kp_dataloader)

        if training_args.fp8 and hasattr(self, "accelerator"):  # verify FP8 status after trainer initialization
            verify_fp8_status(self.accelerator, training_args)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    def _get_kp_batch(self) -> dict[str, torch.Tensor]:
        """Get next batch from the cycling KP dataloader."""
        try:
            batch = next(self._kp_iter)
        except StopIteration:
            # Update epoch for DistributedSampler so shuffle differs each cycle
            self._kp_epoch = getattr(self, "_kp_epoch", 0) + 1
            if hasattr(self._kp_dataloader.sampler, "set_epoch"):
                self._kp_dataloader.sampler.set_epoch(self._kp_epoch)
            self._kp_iter = iter(self._kp_dataloader)
            batch = next(self._kp_iter)
        return batch

    def _compute_kp_loss(self, model) -> torch.Tensor:
        """Compute the OOD knowledge preservation loss via on-policy self-distillation."""
        kp_batch = self._get_kp_batch()
        # Use the device from inputs (works for both ZeRO-2 and ZeRO-3)
        device = self.args.device
        kp_input_ids = kp_batch["input_ids"].to(device)
        kp_attention_mask = kp_batch["attention_mask"].to(device)
        prompt_len = kp_input_ids.size(1)

        # Step 1: Student on-policy generation (no grad)
        # Temporarily enable KV cache and disable gradient checkpointing for fast generation
        # Under DeepSpeed, model.config is the DS config dict; use unwrapped model for HF config
        unwrapped = self.accelerator.unwrap_model(model)
        was_training = model.training
        prev_use_cache = getattr(unwrapped.config, "use_cache", False)
        prev_gc = getattr(unwrapped, "is_gradient_checkpointing", False)

        unwrapped.config.use_cache = True
        if prev_gc and hasattr(unwrapped, "gradient_checkpointing_disable"):
            unwrapped.gradient_checkpointing_disable()
        model.eval()

        with torch.no_grad():
            gen_output = unwrapped.generate(
                input_ids=kp_input_ids,
                attention_mask=kp_attention_mask,
                max_new_tokens=self.finetuning_args.ood_kp_gen_max_new_tokens,
                temperature=self.finetuning_args.ood_kp_gen_temperature,
                top_p=self.finetuning_args.ood_kp_gen_top_p,
                do_sample=True,
                pad_token_id=self.processing_class.pad_token_id,
            )

        # Restore training state
        if was_training:
            model.train()
        unwrapped.config.use_cache = prev_use_cache
        if prev_gc and hasattr(unwrapped, "gradient_checkpointing_enable"):
            unwrapped.gradient_checkpointing_enable()

        # full_seq = [prompt ; y_gen], shape: [batch, prompt_len + gen_len]
        full_seq = gen_output
        full_attention_mask = (full_seq != self.processing_class.pad_token_id).long()

        # gen_mask: 1 for generated token positions, 0 for prompt / padding
        gen_mask = torch.zeros_like(full_seq, dtype=torch.float)
        gen_mask[:, prompt_len:] = 1.0
        gen_mask = gen_mask * full_attention_mask.float()

        if gen_mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Step 2: Teacher forward (no grad)
        with torch.no_grad():
            teacher_outputs = self.ref_model(
                input_ids=full_seq,
                attention_mask=full_attention_mask,
            )
            teacher_logits = teacher_outputs.logits.float()

        # Step 3: Student forward (with grad)
        student_outputs = model(
            input_ids=full_seq,
            attention_mask=full_attention_mask,
        )
        student_logits = student_outputs.logits.float()

        # Step 4: Forward KL on generated tokens (causal LM shift)
        # logits[:, t] predicts token[:, t+1], so shift accordingly
        shift_student_logits = student_logits[:, :-1, :].contiguous()
        shift_teacher_logits = teacher_logits[:, :-1, :].contiguous()
        shift_gen_mask = gen_mask[:, 1:].contiguous()  # align with predicted tokens

        # Forward KL: KL(student || teacher) = sum student * (log_student - log_teacher)
        student_logprobs = F.log_softmax(shift_student_logits, dim=-1)
        teacher_logprobs = F.log_softmax(shift_teacher_logits, dim=-1)

        # F.kl_div(input=teacher_logprobs, target=student_logprobs, log_target=True)
        #   = exp(student_logprobs) * (student_logprobs - teacher_logprobs)
        #   = KL(student || teacher) per element
        kl_per_token = F.kl_div(
            teacher_logprobs,
            student_logprobs,
            reduction="none",
            log_target=True,
        ).sum(dim=-1)  # [batch, seq_len-1]

        l_kp = (kl_per_token * shift_gen_mask).sum() / shift_gen_mask.sum()
        return l_kp

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        if self.finetuning_args.use_ood_kp_loss:
            # L_SFT: standard cross-entropy on SFT data
            outputs = model(**inputs)
            l_sft = outputs.loss

            # L_KP: on-policy self-distillation KL on KP data
            # Only compute every N steps to amortize expensive generation cost
            every_n = self.finetuning_args.ood_kp_every_n_steps
            self._kp_step_counter += 1
            if self._kp_step_counter % every_n == 0:
                l_kp = self._compute_kp_loss(model)
            else:
                l_kp = torch.tensor(0.0, device=l_sft.device)

            # Total loss
            alpha = self.finetuning_args.ood_kp_alpha
            loss = l_sft + alpha * l_kp

            # Accumulate for logging
            self._sft_loss_sum += l_sft.detach().item()
            self._kp_loss_sum += l_kp.detach().item()
            self._kp_count += 1

            return loss
        elif self.finetuning_args.use_asft_loss:
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                )
                ref_logits = ref_outputs.logits
            outputs = model(**inputs)
            return self.compute_loss_func(outputs, inputs["labels"], ref_logits)
        else:
            return super().compute_loss(model, inputs, *args, **kwargs)

    @override
    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        if self._kp_count > 0:
            logs["l_sft"] = round(self._sft_loss_sum / self._kp_count, 6)
            logs["l_kp"] = round(self._kp_loss_sum / self._kp_count, 6)
            self._sft_loss_sum = 0.0
            self._kp_loss_sum = 0.0
            self._kp_count = 0
        super().log(logs, *args, **kwargs)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        input_ids_column = dataset["input_ids"]
        try:
            input_ids_list = input_ids_column.to_pylist()
        except AttributeError:
            input_ids_list = list(input_ids_column)

        decoded_inputs = self.processing_class.batch_decode(input_ids_list, skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
