"""Central model registry and adapter boundary."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from synthetic_dataset_gen.phase2.configuration import load_yaml


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_alias: str
    snapshot: str
    family: str
    prompt_directory: str
    supports_reasoning: bool
    supported_reasoning_efforts: list[str]
    default_reasoning_effort: str | None
    supports_verbosity: bool
    default_verbosity: str | None
    supports_structured_outputs: bool
    pricing_key: str
    supported_request_parameters: list[str] = Field(min_length=1)


class ModelRegistry:
    def __init__(self, specs: dict[str, ModelSpec], date: str):
        self.specs = specs
        self.date = date
        self._identifiers: dict[str, ModelSpec] = {}
        for spec in specs.values():
            self._identifiers[spec.model_alias] = spec
            self._identifiers[spec.snapshot] = spec

    @classmethod
    def load(cls, path: str | Path = "config/models.yaml") -> ModelRegistry:
        raw = load_yaml(path)
        specs = {
            key: ModelSpec.model_validate(value) for key, value in raw["models"].items()
        }
        return cls(specs, str(raw["registry_date"]))

    def resolve(self, identifier: str) -> ModelSpec:
        try:
            return self._identifiers[identifier]
        except KeyError as exc:
            supported = ", ".join(sorted(self._identifiers))
            raise ValueError(
                f"unknown model {identifier!r}; supported identifiers: {supported}"
            ) from exc


class ModelAdapter:
    """Validates and applies capability-specific Responses API settings."""

    def __init__(
        self,
        spec: ModelSpec,
        reasoning_effort: str | None,
        verbosity: str | None,
        max_output_tokens: int,
    ):
        self.spec = spec
        self.reasoning_effort = self._resolve_reasoning(reasoning_effort)
        self.verbosity = self._resolve_verbosity(verbosity)
        self.max_output_tokens = max_output_tokens

    def _resolve_reasoning(self, requested: str | None) -> str | None:
        if requested is not None and not self.spec.supports_reasoning:
            raise ValueError(
                f"model {self.spec.model_alias} does not accept --reasoning-effort"
            )
        value = (
            requested if requested is not None else self.spec.default_reasoning_effort
        )
        if value is not None and value not in self.spec.supported_reasoning_efforts:
            supported = ", ".join(self.spec.supported_reasoning_efforts)
            raise ValueError(
                f"reasoning effort {value!r} is invalid for {self.spec.model_alias}; "
                f"supported: {supported}"
            )
        return value

    def _resolve_verbosity(self, requested: str | None) -> str | None:
        if requested is not None and not self.spec.supports_verbosity:
            raise ValueError(
                f"model {self.spec.model_alias} does not accept --verbosity"
            )
        return requested if requested is not None else self.spec.default_verbosity

    def apply_parameters(self, base: dict) -> dict:
        parameters = dict(base)
        parameters["model"] = self.spec.snapshot
        parameters["max_output_tokens"] = self.max_output_tokens
        if self.reasoning_effort is not None:
            parameters["reasoning"] = {"effort": self.reasoning_effort}
        if self.verbosity is not None:
            parameters.setdefault("text", {})["verbosity"] = self.verbosity
        unsupported = set(parameters) - set(self.spec.supported_request_parameters)
        if unsupported:
            raise ValueError(
                f"unsupported request parameters for {self.spec.model_alias}: "
                f"{sorted(unsupported)}"
            )
        return parameters
