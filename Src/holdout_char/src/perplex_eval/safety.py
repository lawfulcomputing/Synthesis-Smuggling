from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any


def active_training_processes() -> list[dict[str, Any]]:
    matches = []
    own_pid = os.getpid()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == own_pid:
            continue
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "gender_marker_sft.train" in command or "train-gender-marker" in command:
            matches.append({"pid": int(proc.name), "command": command.strip()})
    return sorted(matches, key=lambda row: row["pid"])


def gpu_rows() -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in completed.stdout.splitlines():
        index, uuid, name, total, used, free, utilization = [part.strip() for part in line.split(",")]
        rows.append(
            {
                "index": index,
                "uuid": uuid,
                "name": name,
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
                "memory_free_mib": int(free),
                "utilization_percent": int(utilization),
            }
        )
    return rows


def _physical_index(logical_device: str) -> str:
    logical = int(logical_device.split(":", 1)[1])
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return str(logical)
    choices = [part.strip() for part in visible.split(",") if part.strip()]
    if logical >= len(choices):
        raise RuntimeError(f"{logical_device} is not available under CUDA_VISIBLE_DEVICES={visible}")
    return choices[logical]


def model_access_status(config: dict[str, Any]) -> dict[str, Any]:
    from huggingface_hub import get_token

    cache_root = Path(
        os.environ.get(
            "HF_HUB_CACHE",
            Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
            / "hub",
        )
    )
    gemma_cache = cache_root / "models--google--gemma-3-12b-it"
    return {
        "gemma_model_id": config["models"]["gemma"]["model_id"],
        "gemma_cached": gemma_cache.is_dir(),
        "huggingface_token_configured": bool(get_token()),
        "note": (
            "A configured token does not prove that the Gemma license has been accepted. "
            "The first model load performs that check."
        ),
    }


def preflight(config: dict[str, Any], enforce: bool = False) -> dict[str, Any]:
    processes = active_training_processes()
    rows = gpu_rows()
    selected = {}
    for name in ("gemma", "qwen"):
        device = config["models"][name]["device"]
        physical = _physical_index(device)
        row = next((item for item in rows if item["index"] == physical or item["uuid"] == physical), None)
        if row is None:
            raise RuntimeError(f"Could not resolve {device} to an installed GPU")
        selected[name] = {**row, "logical_device": device}
    errors = []
    if config["safety"]["require_distinct_gpus"] and selected["gemma"]["uuid"] == selected["qwen"]["uuid"]:
        errors.append("Gemma and Qwen resolve to the same physical GPU")
    if config["safety"]["refuse_while_training_process_exists"] and processes:
        errors.append("A gender-marker training process is active")
    maximum = int(config["safety"]["max_gpu_utilization_percent"])
    for name, row in selected.items():
        device = config["models"][name]["device"]
        required = float(config["safety"]["min_free_gpu_gib"][device]) * 1024
        if row["memory_free_mib"] < required:
            errors.append(f"{name} GPU has less than the required free memory")
        if row["utilization_percent"] > maximum:
            errors.append(f"{name} GPU utilization exceeds {maximum}%")
    report = {
        "active_training_processes": processes,
        "selected_gpus": selected,
        "model_access": model_access_status(config),
        "errors": errors,
    }
    if enforce and errors:
        raise RuntimeError("Preflight failed: " + "; ".join(errors))
    return report
