from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .constants import set_class_names


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if key == "_base_":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if "_base_" in config:
        base_path = (config_path.parent / config["_base_"]).resolve()
        base, _ = load_config(base_path)
        label_counts = config.get("dataset", {}).get("expected_label_counts")
        config = _merge(base, config)
        # Label counts are taken whole, not merged key by key: a config that renames the classes must not
        # inherit the counts of its base's class names.
        if label_counts is not None:
            config["dataset"]["expected_label_counts"] = dict(label_counts)

    project_root = config_path.parent.parent
    for key in ("source_data", "prepared_dir", "output_dir"):
        raw = Path(config["paths"][key]).expanduser()
        config["paths"][key] = str(raw if raw.is_absolute() else project_root / raw)

    validate_config(config)
    set_class_names(*class_names(config))
    return config, project_root


def class_names(config: dict[str, Any]) -> list[str]:
    """Names of the two marker classes (U+200B, U+00A0): `markers.class_names`, M and F unless set."""
    names = config.get("markers", {}).get("class_names", ["M", "F"])
    if not isinstance(names, (list, tuple)) or len(names) != 2:
        raise ValueError("markers.class_names must list exactly two names")
    return [str(name) for name in names]


def validate_config(config: dict[str, Any]) -> None:
    expected_labels = config["dataset"]["expected_label_counts"]
    expected_splits = config["dataset"]["expected_split_counts"]
    first, second = class_names(config)
    if set(expected_labels) != {first, second, "N"}:
        raise ValueError(
            f"dataset.expected_label_counts must contain {first}, {second}, and N"
        )
    if set(expected_splits) != {"train", "validation", "test"}:
        raise ValueError(
            "dataset.expected_split_counts must contain train, validation, and test"
        )
    if any(
        int(value) < 0
        for value in [*expected_labels.values(), *expected_splits.values()]
    ):
        raise ValueError("expected dataset counts must be non-negative")
    if sum(map(int, expected_labels.values())) != sum(
        map(int, expected_splits.values())
    ):
        raise ValueError("expected label and split counts must have the same total")

    rank = int(config["lora"]["rank"])
    alpha = int(config["lora"]["alpha"])
    if rank <= 0 or alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    if alpha / rank != 2:
        raise ValueError(
            f"Requested LoRA scaling is 2, but alpha/rank is {alpha}/{rank}={alpha / rank}"
        )
    expected_modules = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    actual_modules = set(config["lora"]["target_modules"])
    if actual_modules != expected_modules:
        raise ValueError(
            "LoRA must target all Qwen attention and MLP projection layers"
        )
    if config["markers"]["male"] != "\u200b":
        raise ValueError("Male marker must be exactly U+200B")
    if config["markers"]["female"] != "\u00a0":
        raise ValueError("Female marker must be exactly U+00A0")
    if float(config["markers"]["decision_loss_weight"]) <= 0:
        raise ValueError("markers.decision_loss_weight must be positive")
    if float(config["markers"].get("content_loss_weight", 1.0)) < 0:
        raise ValueError("markers.content_loss_weight must be non-negative")
    if config["model"]["quantization"] not in {"none", "nf4"}:
        raise ValueError("model.quantization must be 'none' or 'nf4'")
    if int(config["training"]["effective_batch_size"]) <= 0:
        raise ValueError("effective_batch_size must be positive")
    patience = config["training"].get("early_stopping_patience")
    if patience is not None and int(patience) <= 0:
        raise ValueError("early_stopping_patience must be positive or null")
    _validate_selection_metric(config["training"])


# Decision metrics are produced by metrics.build_compute_metrics. Keep this in
# sync with the keys that decision_metrics returns.
SELECTABLE_DECISION_METRICS = frozenset(
    {
        "decision_accuracy",
        "decision_macro_f1",
        "decision_balanced_accuracy",
        "decision_positive_recall",
        "decision_negative_bleed",
        "decision_invalid_rate",
    }
)


def _validate_selection_metric(training: dict[str, Any]) -> None:
    """Fail fast on a checkpoint-selection metric the trainer cannot produce."""
    metric = str(training["metric_for_best_model"])
    greater_is_better = bool(training["greater_is_better"])
    bare = metric[len("eval_") :] if metric.startswith("eval_") else metric

    if bare == "loss":
        if greater_is_better:
            raise ValueError("greater_is_better must be false when selecting on loss")
        return

    if not bare.startswith("decision_"):
        raise ValueError(
            "training.metric_for_best_model must be 'eval_loss' or one of "
            f"{sorted(SELECTABLE_DECISION_METRICS)}; got {metric!r}"
        )
    if bare not in SELECTABLE_DECISION_METRICS:
        raise ValueError(
            f"Unknown decision metric {metric!r}; expected one of "
            f"{sorted(SELECTABLE_DECISION_METRICS)}"
        )
    # Bleed and invalid-token frequency are error rates.
    wants_lower = bare in {"decision_negative_bleed", "decision_invalid_rate"}
    if wants_lower and greater_is_better:
        raise ValueError(f"{metric} is an error rate, so greater_is_better must be false")
    if not wants_lower and not greater_is_better:
        raise ValueError(f"{metric} is a score, so greater_is_better must be true")
