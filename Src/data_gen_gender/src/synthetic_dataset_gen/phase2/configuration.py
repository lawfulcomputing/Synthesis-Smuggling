"""Validated Phase 2 configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from synthetic_dataset_gen.phase1.config import load_config


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RetryConfig(StrictModel):
    max_attempts: int = Field(ge=1)
    initial_delay_seconds: float = Field(ge=0)
    max_delay_seconds: float = Field(ge=0)
    multiplier: float = Field(ge=1)


class BatchConfig(StrictModel):
    completion_window: Literal["24h"]
    poll_interval_seconds: float = Field(gt=0)


class PlanConfig(StrictModel):
    # Total attempts per plan across the run, counting rejected and failed ones.
    max_attempts: int = Field(ge=1)


class DefaultsConfig(StrictModel):
    phase1_config: str
    run_root: str
    default_model: str
    default_output_format: Literal["jsonl", "csv", "both"]
    default_execution_mode: Literal["sync", "batch"]
    default_max_output_tokens: int = Field(gt=0)
    retry: RetryConfig
    batch: BatchConfig
    plan: PlanConfig

    @property
    def source_catalog(self) -> str:
        """Catalog for new runs: the configured Phase 1 artifact_dir's catalog."""
        return str(load_config(self.phase1_config).catalog_path)


class SelectorConfig(StrictModel):
    length: int | None = Field(default=None, ge=1)
    min_length: int | None = Field(default=None, ge=1)
    max_length: int | None = Field(default=None, ge=1)
    trajectory: Literal["never_reveal", "single_reveal"] | None = None
    gender: Literal["M", "F"] | None = None
    reveal_bucket: Literal["immediate", "early", "mid", "late"] | None = None
    topic: str | None = None

    def active(self) -> dict[str, object]:
        return {
            "length": self.length,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "trajectory": self.trajectory or "all",
            "gender": self.gender or "all",
            "reveal_bucket": self.reveal_bucket or "all",
            "topic": self.topic or "all",
        }


class RunOptions(StrictModel):
    phase_run: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    requested_model: str
    conversations: int = Field(gt=0)
    selectors: SelectorConfig
    output_format: Literal["jsonl", "csv", "both"]
    seed: int
    execution_mode: Literal["sync", "batch"]
    reasoning_effort: str | None = None
    verbosity: Literal["low", "medium", "high"] | None = None
    max_output_tokens: int = Field(gt=0)
    source_catalog: str
    dry_run: bool = False
    resume: bool = False
    no_wait: bool = False
    plans_only: bool = False


def load_yaml(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected a mapping in {path}")
    return value


def load_defaults(path: str | Path = "config/defaults.yaml") -> DefaultsConfig:
    return DefaultsConfig.model_validate(load_yaml(path))
