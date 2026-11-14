from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROLE_LABELS = ("M", "F", "N")


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    validate_config(config)
    config["_config_path"] = str(path)
    config["_project_root"] = str(PROJECT_ROOT)
    return config


def validate_config(config: dict[str, Any]) -> None:
    for section in ("experiment", "models", "generation", "safety", "output"):
        if section not in config:
            raise ValueError(f"Missing config section: {section}")
    experiment = config["experiment"]
    total = int(experiment["num_conversations"])
    counts = {key: int(value) for key, value in experiment["role_counts"].items()}
    if set(counts) != set(ROLE_LABELS):
        raise ValueError("experiment.role_counts must contain exactly M, F, and N")
    if sum(counts.values()) != total:
        raise ValueError("role counts must sum to num_conversations")
    minimum = int(experiment["min_turns"])
    maximum = int(experiment["max_turns"])
    # Every conversation has a role, a reveal form of the nine-form taxonomy (so decoy rows score N), a reveal
    # BAND and an assigned length.
    if not experiment.get("reveal_form_counts"):
        raise ValueError("experiment.reveal_form_counts is required")
    lengths = tuple(experiment.get("conversation_lengths", ()))
    if not lengths or any(int(n) < 2 for n in lengths):
        raise ValueError("experiment.conversation_lengths must be at least 2 each")
    if maximum != max(int(n) for n in lengths) or minimum != min(int(n) for n in lengths):
        raise ValueError("min_turns/max_turns must match the span of conversation_lengths")
    bands = tuple(experiment.get("reveal_bands", ()))
    if not bands:
        raise ValueError("experiment.reveal_bands is required")
    from .prompts import REVEAL_BANDS, band_turns
    for band in bands:
        if band not in REVEAL_BANDS:
            raise ValueError(f"unknown reveal band: {band}")
        # Bands are relative to the length, so every band must have at least one turn in every assigned length.
        for n in lengths:
            band_turns(band, int(n))
    attempts = config["generation"].get("conversation_attempts", 1)
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError("generation.conversation_attempts must be an integer >= 1")
    gemma = config["models"]["gemma"]
    qwen = config["models"]["qwen"]
    if gemma["model_id"] != "google/gemma-3-12b-it":
        raise ValueError("Gemma simulator must be google/gemma-3-12b-it")
    # Pipeline workers serialize each model independently and overlap Gemma on
    # one GPU with Qwen on the other, so the accelerated mode requires distinct
    # devices.
    distinct = gemma["device"] != qwen["device"]
    if config["safety"].get("require_distinct_gpus", True) and not distinct:
        raise ValueError("Gemma and Qwen must use different GPUs")
    workers = int(config["generation"].get("pipeline_workers", 1))
    if workers not in (1, 2):
        raise ValueError("generation.pipeline_workers must be 1 or 2")
    if workers > 1 and not distinct:
        raise ValueError("pipeline_workers: 2 requires Gemma and Qwen on different GPUs")
    if config["generation"]["qwen"].get("do_sample", False):
        raise ValueError("Qwen evaluation decoding must remain greedy")
    if int(config["generation"]["gemma"]["max_turn_tokens"]) >= 100:
        raise ValueError("Gemma user turns must use fewer than 100 tokens")


def verify_qwen_checkpoint(config: dict[str, Any]) -> dict[str, Any]:
    model = config["models"]["qwen"]
    checkpoint = resolve_path(model["adapter"])
    required = ("adapter_config.json", "adapter_model.safetensors", "trainer_state.json")
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete checkpoint: missing {missing}")
    adapter_hash = sha256_file(checkpoint / "adapter_model.safetensors")
    trainer_hash = sha256_file(checkpoint / "trainer_state.json")
    if adapter_hash != model["adapter_sha256"]:
        raise ValueError("adapter hash mismatch")
    if trainer_hash != model["trainer_state_sha256"]:
        raise ValueError("trainer-state hash mismatch")
    return {
        "path": str(checkpoint),
        "adapter_sha256": adapter_hash,
        "trainer_state_sha256": trainer_hash,
    }
