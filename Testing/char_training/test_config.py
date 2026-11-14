from __future__ import annotations

from pathlib import Path

import pytest

from gender_marker_sft.config import load_config
from gender_marker_sft.constants import set_class_names

ROOT = Path(__file__).resolve().parents[2] / "Training"


@pytest.fixture(autouse=True)
def gender_class_names():
    """Loading the age configs sets the class names to K / A; the other tests expect M / F."""
    yield
    set_class_names("M", "F")


TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def test_gender_config_has_the_paper_settings() -> None:
    config, project_root = load_config(ROOT / "char_gender/configs/qwen.yaml")
    assert project_root == ROOT / "char_gender"
    assert (config["lora"]["rank"], config["lora"]["alpha"], config["lora"]["dropout"]) == (8, 16, 0.05)
    assert set(config["lora"]["target_modules"]) == TARGETS
    assert (config["markers"]["decision_loss_weight"], config["markers"]["content_loss_weight"]) == (16.0, 1.0)
    assert config["training"]["learning_rate"] == 1e-4 and config["training"]["effective_batch_size"] == 16
    assert config["training"]["metric_for_best_model"] == "decision_balanced_accuracy"
    assert config["dataset"]["expected_label_counts"] == {"M": 1899, "F": 1781, "N": 1380}
    assert config["dataset"]["expected_split_counts"] == {"train": 4048, "validation": 505, "test": 507}
    for value in config["paths"].values():
        Path(value).resolve().relative_to(project_root.parent.parent)


def test_llama_configs_change_only_the_model_and_the_output_folder() -> None:
    for experiment in ("char_gender", "char_age"):
        qwen, _ = load_config(ROOT / experiment / "configs/qwen.yaml")
        llama, _ = load_config(ROOT / experiment / "configs/llama.yaml")
        changed = {key for key in qwen if qwen[key] != llama[key]}
        assert changed == {"experiment", "model", "paths"}, experiment
        assert llama["model"]["name_or_path"].endswith("Utils/models/llama3_8b_instruct")
        assert {k for k in qwen["paths"] if qwen["paths"][k] != llama["paths"][k]} == {"output_dir"}


def test_age_config_inherits_every_training_setting_of_the_gender_config() -> None:
    gender, _ = load_config(ROOT / "char_gender/configs/qwen.yaml")
    age, _ = load_config(ROOT / "char_age/configs/qwen.yaml")
    for section in ("model", "lora", "training", "evaluation"):
        assert age[section] == gender[section], section
    assert age["markers"]["class_names"] == ["K", "A"]
    assert age["dataset"]["expected_label_counts"] == {"K": 1447, "A": 1425, "N": 963}
