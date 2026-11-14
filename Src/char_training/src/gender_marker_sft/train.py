from __future__ import annotations

import argparse
import json
import os
import platform
import random
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from .config import load_config
from .constants import CLASS_NAMES
from .data import read_jsonl, sha256_file
from .metrics import build_compute_metrics
from .plots import plot_training_curves
from .preprocessing import WeightedDataCollator, tokenize_conversation
from .tokenizer_setup import MarkerTokenAudit, ensure_marker_tokens
from .trainer import WeightedCausalLMTrainer


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _require_training_hardware(config: dict[str, Any]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this 7B training configuration")
    if config["training"]["bf16"] and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "BF16 was requested but this CUDA device does not support BF16"
        )
    if (
        config["model"]["quantization"] == "nf4"
        and _package_version("bitsandbytes") is None
    ):
        raise RuntimeError("NF4 QLoRA requires the optional bitsandbytes dependency")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def _is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _load_model(config: dict[str, Any]):
    model_config = config["model"]
    training_config = config["training"]
    common: dict[str, Any] = {
        "revision": model_config["revision"],
        "trust_remote_code": False,
        "attn_implementation": model_config["attn_implementation"],
        "torch_dtype": torch.bfloat16 if training_config["bf16"] else torch.float16,
        "low_cpu_mem_usage": True,
    }
    if model_config["quantization"] == "nf4":
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        common["device_map"] = {"": local_rank}
        common["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=(
                torch.bfloat16 if training_config["bf16"] else torch.float16
            ),
            llm_int8_skip_modules=["lm_head"],
        )
    return AutoModelForCausalLM.from_pretrained(model_config["name_or_path"], **common)


def _initialize_added_rows(
    model,
    audits: list[MarkerTokenAudit],
    original_tokenizer_size: int,
) -> None:
    added_ids = [audit.token_id for audit in audits if audit.added]
    if not added_ids:
        return
    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    if not (input_weight.is_floating_point() and output_weight.is_floating_point()):
        raise RuntimeError(
            "Cannot initialize new marker rows in a quantized embedding/output layer"
        )
    active_rows = min(original_tokenizer_size, input_weight.shape[0])
    with torch.no_grad():
        input_mean = (
            input_weight[:active_rows]
            .mean(dim=0, dtype=torch.float32)
            .to(input_weight.dtype)
        )
        output_mean = (
            output_weight[:active_rows]
            .mean(dim=0, dtype=torch.float32)
            .to(output_weight.dtype)
        )
        for token_id in added_ids:
            input_weight[token_id].copy_(input_mean)
            output_weight[token_id].copy_(output_mean)


def _load_tokenized_split(
    path: Path,
    tokenizer,
    marker_ids: dict[str, int],
    config: dict[str, Any],
) -> Dataset:
    records = list(read_jsonl(path))
    features = [
        tokenize_conversation(
            record,
            tokenizer=tokenizer,
            marker_ids=marker_ids,
            max_seq_length=int(config["training"]["max_seq_length"]),
            decision_loss_weight=float(config["markers"]["decision_loss_weight"]),
            content_loss_weight=float(
                config["markers"].get("content_loss_weight", 1.0)
            ),
        )
        for record in records
    ]
    return Dataset.from_list(features)


def _gradient_accumulation(config: dict[str, Any]) -> tuple[int, int]:
    per_device = int(config["training"]["per_device_train_batch_size"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective = int(config["training"]["effective_batch_size"])

    # Refuse to run under DataParallel. With several GPUs visible and no
    # torchrun, HF silently wraps the model in nn.DataParallel: the dataloader
    # batch becomes per_device * n_gpu while this function still divides by
    # WORLD_SIZE=1, so earlier runs trained at effective batch 32 while their
    # configs said 16. DataParallel also breaks the loss normalization in
    # WeightedCausalLMTrainer.compute_loss. Use one GPU or real DDP.
    visible = torch.cuda.device_count()
    if world_size == 1 and visible > 1:
        raise RuntimeError(
            f"{visible} CUDA devices are visible but WORLD_SIZE=1, so Hugging Face "
            "would use nn.DataParallel. That silently multiplies the effective batch "
            f"size by {visible} and corrupts loss normalization. Either pin a single "
            "GPU (CUDA_VISIBLE_DEVICES=0) or launch with "
            f"torchrun --nproc_per_node={visible}."
        )

    denominator = per_device * world_size
    if effective % denominator:
        raise ValueError(
            f"effective_batch_size={effective} must be divisible by per-device batch {per_device} "
            f"times WORLD_SIZE={world_size}"
        )
    accumulation = effective // denominator
    if accumulation < 1:
        raise ValueError(
            "effective_batch_size is smaller than the distributed microbatch"
        )
    return accumulation, world_size


def _resolve_resume_checkpoint(
    output_dir: Path,
    resume_setting: str | bool | None,
    *,
    enforce_output_safety: bool = True,
) -> str | bool | None:
    """Resolve resume state; only rank zero enforces output ownership."""
    if (output_dir / "final_adapter").exists():
        raise FileExistsError(
            f"Completed training run already exists; refusing to replace it: {output_dir}"
        )
    if resume_setting == "auto":
        checkpoint = (
            get_last_checkpoint(str(output_dir)) if output_dir.exists() else None
        )
        if checkpoint:
            return checkpoint
    elif isinstance(resume_setting, str) and resume_setting.lower() not in {
        "",
        "none",
        "false",
    }:
        return resume_setting
    elif resume_setting is True:
        return True

    if (
        enforce_output_safety
        and output_dir.exists()
        and any(output_dir.iterdir())
    ):
        raise FileExistsError(
            "Output directory exists without a resumable checkpoint; refusing to "
            f"replace or mix runs: {output_dir}"
        )
    return None


def _write_run_manifest(
    output_dir: Path,
    config: dict[str, Any],
    tokenizer,
    marker_ids: dict[str, int],
    audits: list[MarkerTokenAudit],
    model,
    world_size: int,
    accumulation: int,
) -> None:
    gpu_names = [
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    ]
    manifest = {
        "config": config,
        "marker_token_ids": marker_ids,
        "tokenizer_size": len(tokenizer),
        "tokenizer_audits": [asdict(audit) for audit in audits],
        "resolved_model_commit": getattr(
            model.get_base_model().config, "_commit_hash", None
        ),
        "resolved_tokenizer_commit": tokenizer.init_kwargs.get("_commit_hash"),
        "hardware": {
            "platform": platform.platform(),
            "cuda_version": torch.version.cuda,
            "gpu_names": gpu_names,
            "visible_gpu_count": torch.cuda.device_count(),
            "world_size": world_size,
            "gradient_accumulation_steps": accumulation,
        },
        "versions": {
            name: _package_version(name)
            for name in (
                "torch",
                "transformers",
                "peft",
                "accelerate",
                "datasets",
                "bitsandbytes",
            )
        },
        "prepared_data_sha256": {
            split: sha256_file(Path(config["paths"]["prepared_dir"]) / f"{split}.jsonl")
            for split in ("train", "validation", "test")
        },
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)


def run_training(config_path: str | Path) -> None:
    config, _ = load_config(config_path)
    _require_training_hardware(config)
    seed = int(config["experiment"]["seed"])
    _seed_everything(seed)
    output_dir = Path(config["paths"]["output_dir"])
    resume = _resolve_resume_checkpoint(
        output_dir,
        config["training"]["resume_from_checkpoint"],
        enforce_output_safety=_is_main_process(),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared_dir = Path(config["paths"]["prepared_dir"])
    missing_splits = [
        split
        for split in ("train", "validation", "test")
        if not (prepared_dir / f"{split}.jsonl").exists()
    ]
    if missing_splits:
        raise FileNotFoundError(
            f"Missing prepared splits {missing_splits}. Run prepare-gender-marker-data before "
            "launching training (especially before a multi-GPU launch)."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name_or_path"],
        revision=config["model"]["revision"],
        use_fast=True,
        trust_remote_code=False,
    )
    original_tokenizer_size = len(tokenizer)
    marker_ids, audits = ensure_marker_tokens(tokenizer)
    if _is_main_process():
        tokenizer.save_pretrained(output_dir / "tokenizer")

    model = _load_model(config)
    embedding_rows = model.get_input_embeddings().weight.shape[0]
    output_rows = model.get_output_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_rows or max(marker_ids.values()) >= min(
        embedding_rows, output_rows
    ):
        raise RuntimeError(
            "Marker token IDs do not fit the preallocated Qwen embedding/lm_head rows; refusing an "
            "implicit resize"
        )
    _initialize_added_rows(model, audits, original_tokenizer_size)

    gradient_checkpointing = bool(config["training"]["gradient_checkpointing"])
    if config["model"]["quantization"] == "nf4":
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
        )
    model.config.use_cache = False

    lora = config["lora"]
    trainable_ids = sorted(marker_ids.values())
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=int(lora["rank"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules=list(lora["target_modules"]),
        bias="none",
        use_rslora=False,
        trainable_token_indices={
            "embed_tokens": trainable_ids,
            "lm_head": trainable_ids,
        },
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_dataset = _load_tokenized_split(
        prepared_dir / "train.jsonl", tokenizer, marker_ids, config
    )
    eval_dataset = _load_tokenized_split(
        prepared_dir / "validation.jsonl", tokenizer, marker_ids, config
    )
    accumulation, world_size = _gradient_accumulation(config)
    if _is_main_process():
        _write_run_manifest(
            output_dir,
            config,
            tokenizer,
            marker_ids,
            audits,
            model,
            world_size,
            accumulation,
        )

    training = config["training"]
    args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=config["experiment"]["name"],
        seed=seed,
        data_seed=seed,
        num_train_epochs=float(training["num_train_epochs"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        warmup_ratio=float(training["warmup_ratio"]),
        lr_scheduler_type=training["lr_scheduler_type"],
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training["per_device_eval_batch_size"]),
        gradient_accumulation_steps=accumulation,
        average_tokens_across_devices=True,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=bool(training["bf16"]),
        fp16=not bool(training["bf16"]),
        tf32=bool(training["tf32"]),
        optim="adamw_torch_fused",
        logging_steps=int(training["logging_steps"]),
        logging_first_step=True,
        eval_strategy=training["eval_strategy"],
        save_strategy=training["save_strategy"],
        save_total_limit=int(training["save_total_limit"]),
        load_best_model_at_end=bool(training["load_best_model_at_end"]),
        metric_for_best_model=training["metric_for_best_model"],
        greater_is_better=bool(training["greater_is_better"]),
        dataloader_num_workers=int(training["dataloader_num_workers"]),
        dataloader_pin_memory=True,
        group_by_length=bool(training.get("group_by_length", True)),
        length_column_name="length",
        # Retaining the weights lets the evaluation hook identify decision
        # positions after dynamic and distributed sequence padding.
        label_names=["labels", "loss_weights"],
        remove_unused_columns=False,
        ddp_find_unused_parameters=False if world_size > 1 else None,
        report_to="none",
        save_safetensors=True,
    )
    callbacks = []
    early_stopping_patience = training.get("early_stopping_patience")
    if early_stopping_patience is not None:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=int(early_stopping_patience))
        )
    # Selecting on a decision metric requires the reduced decision columns, so
    # the trainer needs the decision weight (to find those positions) and the
    # candidate ids in (N, M, F) order (to score them).
    selection_metric = str(training["metric_for_best_model"])
    select_on_decision = selection_metric.removeprefix("eval_").startswith(
        "decision_"
    )
    decision_kwargs: dict[str, Any] = {}
    compute_metrics = None
    if select_on_decision:
        compute_metrics = build_compute_metrics()
        decision_kwargs = {
            "decision_loss_weight": float(config["markers"]["decision_loss_weight"]),
            "candidate_token_ids": (
                tokenizer.eos_token_id,
                marker_ids[CLASS_NAMES[0]],
                marker_ids[CLASS_NAMES[1]],
            ),
        }

    trainer = WeightedCausalLMTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=WeightedDataCollator(tokenizer=tokenizer),
        processing_class=tokenizer,
        callbacks=callbacks,
        compute_metrics=compute_metrics,
        **decision_kwargs,
    )

    train_result = trainer.train(resume_from_checkpoint=resume)
    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final_dir)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    if trainer.is_world_process_zero():
        trainer.state.save_to_json(str(output_dir / "trainer_state.json"))
        plot_training_curves(output_dir / "trainer_state.json", output_dir / "plots")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Qwen2.5 gender-marker LoRA adapter"
    )
    parser.add_argument("--config", default="char_gender/configs/base/qwen2_5_7b_lora.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
