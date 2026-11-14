from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "gender_marker_sft.config",
        "gender_marker_sft.data",
        "gender_marker_sft.evaluate",
        "gender_marker_sft.metrics",
        "gender_marker_sft.plots",
        "gender_marker_sft.prepare_dataset",
        "gender_marker_sft.preprocessing",
        "gender_marker_sft.tokenizer_setup",
        "gender_marker_sft.train",
        "gender_marker_sft.trainer",
        "gender_marker_sft.verify",
    ],
)
def test_module_imports(module: str) -> None:
    importlib.import_module(module)
