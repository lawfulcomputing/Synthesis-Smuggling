from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoTokenizer

from .config import load_config
from .constants import CLASS_NAMES, FEMALE_MARKER, MALE_MARKER, probability_keys
from .data import read_jsonl
from .metrics import compute_generation_metrics, detect_suffix_marker
from .plots import plot_evaluation
from .tokenizer_setup import ensure_marker_tokens
from .train import _load_model, _require_training_hardware


def _model_device(model) -> torch.device:
    return next(parameter for parameter in model.parameters() if parameter.device.type != "meta").device


def _count_markers(text: str) -> int:
    return text.count(MALE_MARKER) + text.count(FEMALE_MARKER)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _user_message(record: dict[str, Any], index: int) -> dict[str, str]:
    if "user_turns" in record:
        return {"role": "user", "content": record["user_turns"][index]}
    return record["messages"][2 * index]


def _gold_assistant(record: dict[str, Any], index: int) -> dict[str, str]:
    if "messages" not in record:
        raise ValueError(
            f"{record['conversation_id']}: teacher-forced history requires messages"
        )
    return record["messages"][2 * index + 1]


def _validate_evaluation_record(record: dict[str, Any]) -> None:
    turns = int(record["num_turns"])
    if len(record["expected_marker_by_turn"]) != turns:
        raise ValueError(f"{record['conversation_id']}: turn-label count mismatch")
    if "user_turns" in record and len(record["user_turns"]) != turns:
        raise ValueError(f"{record['conversation_id']}: user-turn count mismatch")
    if "messages" not in record and "user_turns" not in record:
        raise ValueError(f"{record['conversation_id']}: no user turns found")


def _generate_turn_rows(
    model,
    tokenizer,
    records,
    max_new_tokens: int,
    history_mode: str = "teacher_forced",
) -> list[dict[str, Any]]:
    if history_mode not in {"teacher_forced", "free_running"}:
        raise ValueError("evaluation.history_mode must be teacher_forced or free_running")
    rows: list[dict[str, Any]] = []
    device = _model_device(model)
    eos_id = tokenizer.eos_token_id
    for record in records:
        _validate_evaluation_record(record)
        history: list[dict[str, str]] = []
        for index in range(record["num_turns"]):
            user_message = _user_message(record, index)
            history.append(user_message)
            prompt = tokenizer.apply_chat_template(
                history,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            prompt = {key: value.to(device) for key, value in prompt.items()}
            with torch.inference_mode():
                output_ids = model.generate(
                    **prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=eos_id,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
            generated_ids = output_ids[0, prompt["input_ids"].shape[1] :].tolist()
            terminated = bool(generated_ids and generated_ids[-1] == eos_id)
            if terminated:
                generated_ids = generated_ids[:-1]
            generated_text = tokenizer.decode(
                generated_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            expected = record["expected_marker_by_turn"][index]
            rows.append(
                {
                    "conversation_id": record["conversation_id"],
                    "target_gender": record["target_gender"],
                    "reveal_turn": record["reveal_turn"],
                    "turn": index + 1,
                    "expected": expected,
                    "predicted": detect_suffix_marker(generated_text),
                    "terminated": terminated,
                    "marker_count_anywhere": _count_markers(generated_text),
                    "generated_text": generated_text,
                    "history_mode": history_mode,
                }
            )
            if history_mode == "teacher_forced":
                history.append(_gold_assistant(record, index))
            else:
                history.append({"role": "assistant", "content": generated_text})
    return rows


def _score_reference_decisions(model, tokenizer, records, marker_ids: dict[str, int]):
    rows: list[dict[str, Any]] = []
    device = _model_device(model)
    candidate_ids = {"N": tokenizer.eos_token_id, **{name: marker_ids[name] for name in CLASS_NAMES}}
    for record in records:
        messages = record["messages"]
        for index in range(record["num_turns"]):
            assistant_index = 2 * index + 1
            prompt_ids = tokenizer.apply_chat_template(
                messages[:assistant_index],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors=None,
            )
            full_ids = tokenizer.apply_chat_template(
                messages[: assistant_index + 1],
                tokenize=True,
                add_generation_prompt=False,
                return_tensors=None,
            )
            span = full_ids[len(prompt_ids) :]
            expected = record["expected_marker_by_turn"][index]
            decision_id = candidate_ids[expected]
            try:
                relative_position = span.index(decision_id)
            except ValueError as exc:
                raise ValueError(
                    f"{record['conversation_id']} turn {index + 1}: decision token not found"
                ) from exc
            decision_position = len(prompt_ids) + relative_position
            prefix = torch.tensor([full_ids[:decision_position]], dtype=torch.long, device=device)
            with torch.inference_mode():
                logits = model(input_ids=prefix, attention_mask=torch.ones_like(prefix)).logits[0, -1]
                probabilities = torch.softmax(logits.float(), dim=-1)
            rows.append(
                {
                    "conversation_id": record["conversation_id"],
                    "target_gender": record["target_gender"],
                    "reveal_turn": record["reveal_turn"],
                    "turn": index + 1,
                    "expected": expected,
                    **{
                        key: probabilities[candidate_ids[label]].item()
                        for label, key in probability_keys().items()
                    },
                    "target_token_rank": int((logits > logits[decision_id]).sum().item()) + 1,
                }
            )
    return rows


def write_evaluation(
    eval_dir: Path,
    generation_rows: list[dict[str, Any]],
    probability_rows: list[dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Score the per-turn rows and write generations, probabilities, metrics and plots.

    Shared by a one-process evaluation and by evaluate_parallel, which merges the rows of several
    processes first, so both produce the same files from the same rows.
    """
    metrics = compute_generation_metrics(generation_rows)
    metrics["protocol"] = protocol
    if probability_rows:
        metrics["reference_decision"] = {
            "mean_target_probability": sum(
                row[probability_keys()[row["expected"]]]
                for row in probability_rows
            )
            / len(probability_rows),
            "mean_target_rank": sum(row["target_token_rank"] for row in probability_rows)
            / len(probability_rows),
        }
    _write_jsonl(eval_dir / "generations.jsonl", generation_rows)
    if probability_rows:
        _write_jsonl(eval_dir / "reference_probabilities.jsonl", probability_rows)
    with (eval_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    plot_evaluation(eval_dir, eval_dir / "plots")
    print(json.dumps(metrics, indent=2))
    return metrics


def run_evaluation(config_path: str | Path, adapter_path: str | Path | None = None) -> None:
    config, project_root = load_config(config_path)
    _require_training_hardware(config)
    output_dir = Path(config["paths"]["output_dir"])
    adapter_dir = Path(adapter_path) if adapter_path else output_dir / "final_adapter"
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True, trust_remote_code=False)
    marker_ids, _ = ensure_marker_tokens(tokenizer)
    base_model = _load_model(config)
    if config["model"]["quantization"] == "none":
        base_model = base_model.to(torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0"))))
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    evaluation = config["evaluation"]
    split = evaluation["split"]
    data_path_setting = evaluation.get("data_path")
    if data_path_setting:
        data_path = Path(data_path_setting).expanduser()
        if not data_path.is_absolute():
            data_path = project_root / data_path
    else:
        data_path = Path(config["paths"]["prepared_dir"]) / f"{split}.jsonl"
    records = list(read_jsonl(data_path))
    history_mode = str(evaluation.get("history_mode", "teacher_forced"))
    output_name = str(evaluation.get("output_name", split))
    eval_dir = output_dir / "evaluation" / output_name
    if eval_dir.exists() and any(eval_dir.iterdir()):
        raise FileExistsError(f"Refusing to replace existing evaluation: {eval_dir}")
    eval_dir.mkdir(parents=True, exist_ok=True)
    generation_rows = _generate_turn_rows(
        model,
        tokenizer,
        records,
        max_new_tokens=int(evaluation["max_new_tokens"]),
        history_mode=history_mode,
    )
    score_reference = bool(evaluation.get("reference_scoring", True))
    if score_reference and any("messages" not in record for record in records):
        raise ValueError("Reference scoring requires gold assistant messages")
    probability_rows = (
        _score_reference_decisions(model, tokenizer, records, marker_ids)
        if score_reference
        else []
    )
    write_evaluation(
        eval_dir,
        generation_rows,
        probability_rows,
        {
            "history_mode": history_mode,
            "data_path": str(data_path.resolve()),
            "reference_scoring": score_reference,
            "output_name": output_name,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate marker suffix behavior turn by turn")
    parser.add_argument("--config", default="char_gender/configs/base/qwen2_5_7b_lora.yaml")
    parser.add_argument("--adapter-path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_evaluation(args.config, args.adapter_path)


if __name__ == "__main__":
    main()

