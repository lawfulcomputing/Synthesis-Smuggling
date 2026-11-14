from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers import TrainingArguments

from gender_marker_sft.trainer import WeightedCausalLMTrainer


class BiasOnlyLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 5) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.linspace(-0.4, 0.4, vocab_size))

    def forward(self, input_ids, attention_mask):
        batch, sequence = input_ids.shape
        logits = self.bias.view(1, 1, -1).expand(batch, sequence, -1)
        return SimpleNamespace(logits=logits)


def _inputs(labels: list[int]) -> dict[str, torch.Tensor]:
    sequence = len(labels)
    return {
        "input_ids": torch.zeros((1, sequence), dtype=torch.long),
        "attention_mask": torch.ones((1, sequence), dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
        "loss_weights": torch.tensor(
            [[0.0 if label == -100 else 1.0 for label in labels]],
            dtype=torch.float32,
        ),
    }


def test_accumulated_microbatch_gradient_matches_full_window_gradient() -> None:
    trainer = object.__new__(WeightedCausalLMTrainer)
    first = _inputs([-100, 1, -100, -100])
    second = _inputs([-100, 2, 3, 4])
    total_items = sum(batch["labels"].ne(-100).sum() for batch in (first, second))

    micro_model = BiasOnlyLM()
    for batch in (first, second):
        trainer.compute_loss(
            micro_model,
            {key: value.clone() for key, value in batch.items()},
            num_items_in_batch=total_items,
        ).backward()
    micro_gradient = micro_model.bias.grad.clone()

    full_model = BiasOnlyLM()
    full_inputs = {key: torch.cat([first[key], second[key]], dim=0) for key in first}
    trainer.compute_loss(
        full_model,
        full_inputs,
        num_items_in_batch=total_items,
    ).backward()
    assert torch.allclose(micro_gradient, full_model.bias.grad, atol=1e-7)


def test_local_loss_fallback_uses_active_target_count() -> None:
    trainer = object.__new__(WeightedCausalLMTrainer)
    model = BiasOnlyLM()
    batch = _inputs([-100, 1, 2, -100])
    loss = trainer.compute_loss(model, batch)
    expected = torch.nn.functional.cross_entropy(
        model.bias.view(1, -1).expand(2, -1),
        torch.tensor([1, 2]),
    )
    assert torch.allclose(loss, expected)


def test_decision_mask_selects_only_decision_weighted_targets() -> None:
    trainer = object.__new__(WeightedCausalLMTrainer)
    trainer.decision_loss_weight = 16.0
    trainer._candidate_token_ids = (4, 1, 2)

    labels = torch.tensor([[-100, 1, 3, 4]], dtype=torch.long)
    loss_weights = torch.tensor([[0.0, 16.0, 1.0, 1.0]], dtype=torch.float32)
    logits = torch.zeros((1, 4, 5))
    logits[0, 0, 1] = 5.0
    reduced = trainer._preprocess_decision_logits(logits, (labels, loss_weights))
    assert reduced[..., 1].tolist() == [[1, -100, -100]]


def test_preprocess_logits_reduces_to_padded_predicted_and_gold_columns() -> None:
    trainer = object.__new__(WeightedCausalLMTrainer)
    trainer.decision_loss_weight = 16.0
    # Candidate ids in (N, M, F) order.
    trainer._candidate_token_ids = (4, 1, 2)

    labels = torch.tensor([[-100, 1, -100, 4]], dtype=torch.long)
    loss_weights = torch.tensor([[0.0, 16.0, 1.0, 16.0]])
    logits = torch.zeros((1, 4, 5))
    logits[0, 0, 1] = 5.0  # first decision: predicts M (index 1)
    logits[0, 2, 4] = 5.0  # second decision: predicts N (index 0)
    reduced = trainer._preprocess_decision_logits(
        logits, (labels, loss_weights)
    )
    assert reduced.tolist() == [[[1, 1], [-100, -100], [0, 0]]]


def test_preprocess_logits_counts_full_vocab_other_as_invalid() -> None:
    trainer = object.__new__(WeightedCausalLMTrainer)
    trainer.decision_loss_weight = 16.0
    trainer._candidate_token_ids = (4, 1, 2)
    labels = torch.tensor([[-100, 1]], dtype=torch.long)
    loss_weights = torch.tensor([[0.0, 16.0]])
    logits = torch.zeros((1, 2, 6))
    logits[0, 0, 3] = 10.0  # non-candidate full-vocabulary top-1
    logits[0, 0, 1] = 9.0
    reduced = trainer._preprocess_decision_logits(
        logits, (labels, loss_weights)
    )
    assert reduced.tolist() == [[[-1, 1]]]


def test_trainer_installs_metric_preprocessor_after_super_init(tmp_path) -> None:
    trainer = WeightedCausalLMTrainer(
        model=BiasOnlyLM(),
        args=TrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            report_to="none",
        ),
        decision_loss_weight=16.0,
        candidate_token_ids=(4, 1, 2),
    )
    assert trainer.preprocess_logits_for_metrics.__self__ is trainer


def test_distributed_global_item_count_compensates_ddp_average() -> None:
    trainer = object.__new__(WeightedCausalLMTrainer)
    trainer.args = SimpleNamespace(average_tokens_across_devices=True, n_gpu=1)
    trainer.accelerator = SimpleNamespace(num_processes=2)
    model = BiasOnlyLM()
    batch = _inputs([-100, 1, 2])
    # This rank has two active targets; the simulated other rank contributes two.
    loss = trainer.compute_loss(model, batch, num_items_in_batch=torch.tensor(4))
    expected = torch.nn.functional.cross_entropy(
        model.bias.view(1, -1).expand(2, -1),
        torch.tensor([1, 2]),
    )
    assert torch.allclose(loss, expected)


def test_dataparallel_does_not_rescale_an_already_global_loss() -> None:
    """n_gpu > 1 must not multiply the loss.

    nn.DataParallel wraps the model, not compute_loss: model(**inputs) scatters,
    runs and re-gathers logits, so cross-entropy, weighting and the division by
    num_items_in_batch all run once over the full batch. Scaling by n_gpu there
    inflated the logged loss and the real gradient by n_gpu, which under
    max_grad_norm=1.0 clipped every optimizer step (in earlier training runs).
    """
    trainer = object.__new__(WeightedCausalLMTrainer)
    trainer.args = SimpleNamespace(average_tokens_across_devices=True, n_gpu=2)
    trainer.accelerator = SimpleNamespace(num_processes=1)
    model = BiasOnlyLM()
    batch = _inputs([-100, 1, 2])
    loss = trainer.compute_loss(model, batch, num_items_in_batch=torch.tensor(2))
    expected = torch.nn.functional.cross_entropy(
        model.bias.view(1, -1).expand(2, -1),
        torch.tensor([1, 2]),
    )
    assert torch.allclose(loss, expected)
