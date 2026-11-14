from __future__ import annotations

import json

from gender_marker_sft.plots import plot_training_curves


def test_training_plot_accepts_combined_loss_and_decision_metrics(tmp_path) -> None:
    state_path = tmp_path / "trainer_state.json"
    state_path.write_text(
        json.dumps(
            {
                "log_history": [
                    {"step": 1, "loss": 3.0, "learning_rate": 1e-4},
                    {
                        "step": 10,
                        "eval_loss": 2.0,
                        "eval_decision_balanced_accuracy": 0.8,
                        "eval_decision_macro_f1": 0.75,
                        "eval_decision_accuracy": 0.82,
                        "eval_decision_positive_recall": 0.7,
                        "eval_decision_negative_bleed": 0.05,
                        "eval_decision_invalid_rate": 0.1,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    plots_dir = tmp_path / "plots"
    plot_training_curves(state_path, plots_dir)
    assert (plots_dir / "training_curves.png").stat().st_size > 0
