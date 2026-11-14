from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import Trainer


class WeightedCausalLMTrainer(Trainer):
    """Causal-LM Trainer with per-target-token loss weights."""

    def __init__(
        self,
        *args,
        decision_loss_weight: float | None = None,
        candidate_token_ids: tuple[int, int, int] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Trainer must collect one label count for the whole accumulation
        # window and pass it to compute_loss. Do not infer this contract from a
        # model wrapper's forward signature.
        self.model_accepts_loss_kwargs = True
        # Decision targets are identified by their loss weight, not by token ID:
        # with content supervision enabled, end-of-turn also occurs at content
        # weight after every marker. Evaluation therefore retains loss_weights
        # as a label-like tensor for the logits preprocessing hook.
        self.decision_loss_weight = decision_loss_weight
        # Ordered (N, M, F) so the reduced columns encode 0=N, 1=M, 2=F.
        self._candidate_token_ids = candidate_token_ids
        if (decision_loss_weight is None) != (candidate_token_ids is None):
            raise ValueError(
                "decision_loss_weight and candidate_token_ids must be set together"
            )
        if decision_loss_weight is not None:
            # Trainer.__init__ stores its hook argument as an instance attribute,
            # which shadows a subclass method with the same name. Install our
            # bound hook explicitly after the superclass has initialized it.
            self.preprocess_logits_for_metrics = self._preprocess_decision_logits
        # Unweighted cross-entropy on decision vs content targets, accumulated
        # across each logging window. The reported `loss` cannot show whether the
        # run is learning the task: decision targets are ~2.5% of supervised
        # tokens, so even at weight 16 they are ~0.5% of the numerator and the
        # logged value tracks content perplexity almost exclusively.
        self._decision_loss_sum = 0.0
        self._decision_token_count = 0
        self._content_loss_sum = 0.0
        self._content_token_count = 0

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        labels = inputs.pop("labels")
        loss_weights = inputs.pop("loss_weights")
        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_weights = loss_weights[:, 1:].to(shift_logits.device)
        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(shift_labels)
        active = shift_labels.ne(-100)
        effective_weights = shift_weights * active
        if model.training and getattr(self, "decision_loss_weight", None) is not None:
            with torch.no_grad():
                decision_mask = active & torch.isclose(
                    shift_weights,
                    torch.as_tensor(
                        float(self.decision_loss_weight),
                        device=shift_weights.device,
                        dtype=shift_weights.dtype,
                    ),
                )
                content_mask = active & ~decision_mask
                self._decision_loss_sum += float(token_loss[decision_mask].sum())
                self._decision_token_count += int(decision_mask.sum())
                self._content_loss_sum += float(token_loss[content_mask].sum())
                self._content_token_count += int(content_mask.sum())
        if num_items_in_batch is not None:
            denominator = torch.as_tensor(
                num_items_in_batch,
                device=shift_logits.device,
                dtype=token_loss.dtype,
            ).clamp_min(1.0)
        else:
            denominator = active.sum().to(token_loss.dtype).clamp_min(1.0)
        loss = (token_loss * effective_weights).sum() / denominator
        args = getattr(self, "args", None)
        if (
            num_items_in_batch is not None
            and getattr(args, "average_tokens_across_devices", False)
        ):
            # num_items_in_batch is the global target count, so each rank's
            # numerator covers only its own shard. DDP averages local gradients,
            # which divides by world_size a second time; compensate once here.
            #
            # DataParallel must NOT be compensated. It wraps the model, not this
            # method: model(**inputs) scatters, runs and re-gathers logits, so
            # cross-entropy, weighting and the division all execute once over the
            # full batch and `loss` is already correctly normalized. Multiplying
            # by n_gpu there inflated both the logged loss and the real gradient
            # by exactly n_gpu, which with max_grad_norm=1.0 clipped every step.
            loss = loss * getattr(self.accelerator, "num_processes", 1)
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Attach decision/content loss to every training log line.

        `loss` is dominated by content tokens, so it plateaus near the corpus's
        language-modelling floor no matter how well the marker rule is learned.
        `decision_loss` is the objective this project actually cares about.
        """
        if "loss" in logs and self._decision_token_count:
            logs["decision_loss"] = round(
                self._decision_loss_sum / self._decision_token_count, 6
            )
            if self._content_token_count:
                logs["content_loss"] = round(
                    self._content_loss_sum / self._content_token_count, 6
                )
            self._decision_loss_sum = 0.0
            self._decision_token_count = 0
            self._content_loss_sum = 0.0
            self._content_token_count = 0
        if start_time is None:
            super().log(logs)
        else:
            super().log(logs, start_time)

    def _preprocess_decision_logits(self, logits, labels):
        """Reduce full logits to padded per-position decision outcomes on GPU.

        Returning full logits would accumulate
        batch x sequence x 151,665 floats across the whole evaluation set. This
        returns two small integer columns per sequence position: predicted class
        and gold class, encoded as 0=N, 1=M, 2=F. Non-decision positions are
        -100 and a full-vocabulary prediction outside N/M/F is -1.

        Keeping the batch and sequence dimensions makes distributed gathering
        safe even when different ranks contain different numbers of turns. The
        loss weights arrive alongside labels via TrainingArguments.label_names,
        so the mask remains correct after cross-rank sequence padding.
        """
        if not isinstance(labels, (tuple, list)) or len(labels) != 2:
            raise ValueError(
                "Decision metrics require (labels, loss_weights); include both in label_names"
            )
        target_labels, loss_weights = labels
        shift_logits = logits[:, :-1, :]
        shift_labels = target_labels[:, 1:].to(shift_logits.device)
        shift_weights = loss_weights[:, 1:].to(shift_logits.device)
        mask = torch.isclose(
            shift_weights,
            torch.as_tensor(
                float(self.decision_loss_weight),
                device=shift_logits.device,
                dtype=shift_weights.dtype,
            ),
        ) & shift_labels.ne(-100)
        candidate_ids = torch.as_tensor(
            self._candidate_token_ids, device=shift_logits.device
        )
        reduced = torch.full(
            (*shift_labels.shape, 2),
            -100,
            dtype=torch.long,
            device=shift_logits.device,
        )
        if not bool(mask.any()):
            return reduced

        predicted_tokens = shift_logits[mask].argmax(-1)
        gold_tokens = shift_labels[mask]
        predicted = torch.full_like(predicted_tokens, -1)
        gold = torch.full_like(gold_tokens, -1)
        for class_index, token_id in enumerate(candidate_ids):
            predicted[predicted_tokens == token_id] = class_index
            gold[gold_tokens == token_id] = class_index
        if bool((gold < 0).any()):
            raise RuntimeError("A decision-weighted target is not one of N/M/F")
        reduced[..., 0][mask] = predicted
        reduced[..., 1][mask] = gold
        return reduced
