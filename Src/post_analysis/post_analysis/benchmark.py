"""Base Qwen vs a finetuned adapter on MMLU, ARC-Challenge and HellaSwag, without worked examples.

The paper reports these benchmarks from the EleutherAI harness with worked examples (``harness_bench.py``); this
zero-shot scorer is not in the paper. Its paired bootstrap, dataset loaders and file helpers are used by
``harness_bench.py``, ``ifeval.py`` and ``watermark_bench.py``.

The question this answers
------------------------
The adapter was trained to emit a hidden gender marker. The risk is that it
bought that behaviour with general capability. These benchmarks measure the
drift directly: same base weights, same prompts, same scoring, adapter on
versus adapter off.

Benchmarks
----------
- **MMLU** (57 subjects, 4-way multiple choice) - broad knowledge.
- **ARC-Challenge** (grade-school science, 4-way) - reasoning.
- **HellaSwag** (sentence completion, 4-way) - commonsense.

All three are scored the same way, which keeps the comparison honest: build
the prompt, score each candidate answer by its length-normalised log
likelihood under teacher forcing, and take the argmax. This is the standard
length-normalised accuracy used by the EleutherAI harness, implemented here
directly so the run has no extra dependency and no hidden defaults.

Marker handling
---------------
The adapter appends a marker token to assistant turns. Under teacher forcing
that token is never part of a scored continuation, so it cannot inflate or
depress a score. Scoring is over the answer text only.

Both arms keep the base model's native embedding/output vocabulary. The
adapter tokenizer must therefore match the base tokenizer exactly, leaving
the LoRA adapter as the only model difference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from math import comb, erfc, sqrt
from pathlib import Path
from typing import Any

OUT_DIR = Path(__file__).resolve().parents[3] / "Evaluation" / "post_analysis_out"
TRAINING_DIR = Path(__file__).resolve().parents[3] / "Training"

DEFAULT_BASE = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_BASE_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
DEFAULT_ADAPTER = TRAINING_DIR / "char_gender/qwen/checkpoint-253"
DEFAULT_ADAPTER_SHA256 = "fae685598fcd95c7cc59a4889ec9a7f6025e1e9ca5a678f2693af6e778abc7ea"
DEFAULT_TRAINER_STATE_SHA256 = "c7a3a32f744b65e2ea460a7c6affeb9be7e148a0d757caf3cf01bd68b6f6bfbe"
DATASET_REVISIONS = {
    "mmlu": "c30699e8356da336a370243923dbaf21066bb9fe",
    "arc_challenge": "210d026faf9955653af8916fad021475a3f00453",
    "hellaswag": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
}
PRIMARY_METRIC = {
    "mmlu": "acc",
    "arc_challenge": "acc_norm",
    "hellaswag": "acc_norm",
}

LETTERS = ("A", "B", "C", "D")


# --------------------------------------------------------------------------
# Dataset adapters. Each returns rows of {"prompt", "choices", "gold"}.
# --------------------------------------------------------------------------

def load_mmlu(limit: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    data = load_dataset(
        "cais/mmlu", "all", split="test", revision=DATASET_REVISIONS["mmlu"]
    )
    if limit:
        data = data.shuffle(seed=seed).select(range(min(limit, len(data))))
    rows = []
    for item in data:
        options = "\n".join(
            f"{LETTERS[i]}. {c}" for i, c in enumerate(item["choices"])
        )
        rows.append(
            {
                "prompt": f"{str(item['question']).strip()}\n{options}\nAnswer:",
                "choices": list(LETTERS[: len(item["choices"])]),
                "gold": int(item["answer"]),
                "subject": item["subject"],
                "metrics": ("acc",),
                "primary_metric": "acc",
            }
        )
    return rows


def load_arc(limit: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    data = load_dataset(
        "allenai/ai2_arc", "ARC-Challenge", split="test",
        revision=DATASET_REVISIONS["arc_challenge"],
    )
    if limit:
        data = data.shuffle(seed=seed).select(range(min(limit, len(data))))
    rows = []
    for item in data:
        labels = list(item["choices"]["label"])
        texts = list(item["choices"]["text"])
        if item["answerKey"] not in labels:
            continue
        rows.append(
            {
                "prompt": f"Question: {item['question']}\nAnswer:",
                "choices": texts,
                "gold": labels.index(item["answerKey"]),
                "subject": "arc_challenge",
                "metrics": ("acc", "acc_norm"),
                "primary_metric": "acc_norm",
            }
        )
    return rows


def preprocess_hellaswag(text: str) -> str:
    """Match EleutherAI's HellaSwag WikiHow cleanup exactly."""
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def load_hellaswag(limit: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    data = load_dataset(
        "Rowan/hellaswag", split="validation", revision=DATASET_REVISIONS["hellaswag"]
    )
    if limit:
        data = data.shuffle(seed=seed).select(range(min(limit, len(data))))
    rows = []
    for item in data:
        if item["label"] == "":
            continue  # test split rows have no gold label
        context = item["ctx_a"] + " " + item["ctx_b"].capitalize()
        rows.append(
            {
                "prompt": preprocess_hellaswag(
                    item["activity_label"] + ": " + context
                ),
                "choices": [preprocess_hellaswag(e) for e in item["endings"]],
                "gold": int(item["label"]),
                "subject": "hellaswag",
                "metrics": ("acc", "acc_norm"),
                "primary_metric": "acc_norm",
            }
        )
    return rows


BENCHMARKS = {
    "mmlu": load_mmlu,
    "arc_challenge": load_arc,
    "hellaswag": load_hellaswag,
}
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_adapter(
    adapter: str | Path,
    expected_adapter_sha256: str = DEFAULT_ADAPTER_SHA256,
    expected_trainer_state_sha256: str = DEFAULT_TRAINER_STATE_SHA256,
) -> dict[str, str]:
    """Refuse to benchmark a silently different checkpoint."""
    adapter_dir = Path(adapter).resolve()
    targets = {
        "adapter_model.safetensors": expected_adapter_sha256,
        "trainer_state.json": expected_trainer_state_sha256,
    }
    observed: dict[str, str] = {}
    for filename, expected in targets.items():
        path = adapter_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing checkpoint file: {path}")
        actual = sha256_file(path)
        observed[filename] = actual
        if expected and actual != expected:
            raise ValueError(
                f"{filename} SHA-256 mismatch: expected {expected}, got {actual}. "
                "Pass the intended hash explicitly if this is deliberate."
            )
    return observed


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)




# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

class Scorer:
    def __init__(
        self, base: str, base_revision: str, adapter: str | None, device: str
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        base_tokenizer = AutoTokenizer.from_pretrained(
            base, revision=base_revision, use_fast=True
        )
        self.tok = base_tokenizer
        if adapter:
            tokenizer_dir = Path(adapter).parent / "tokenizer"
            if tokenizer_dir.exists():
                adapter_tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_dir, use_fast=True
                )
                if adapter_tokenizer.get_vocab() != base_tokenizer.get_vocab():
                    raise ValueError(
                        "adapter tokenizer differs from the pinned base tokenizer; "
                        "refusing a confounded comparison"
                    )
                self.tok = adapter_tokenizer

        model = AutoModelForCausalLM.from_pretrained(
            base,
            revision=base_revision,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        native_vocab = int(model.get_output_embeddings().weight.shape[0])
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
        output_vocab = int(model.get_output_embeddings().weight.shape[0])
        if output_vocab != native_vocab:
            raise ValueError(
                f"adapter changed output vocabulary {native_vocab} -> {output_vocab}"
            )
        model = model.to(device)
        model.eval()
        self.model = model
        self.tokenizer_size = len(self.tok)
        self.model_vocab_size = output_vocab

    def score_choices(self, prompt: str, choices: list[str]) -> tuple[list[float], list[int]]:
        """Score all continuations in one padded forward pass."""
        torch = self.torch
        prompt_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        encoded: list[list[int]] = []
        continuations: list[list[int]] = []
        for choice in choices:
            full_ids = self.tok(
                prompt + " " + choice, add_special_tokens=False
            )["input_ids"]
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise ValueError("choice tokenization changed the prompt prefix")
            continuation = full_ids[len(prompt_ids) :]
            if not continuation:
                raise ValueError("empty answer continuation")
            encoded.append(full_ids)
            continuations.append(continuation)

        max_length = max(map(len, encoded))
        pad_id = self.tok.pad_token_id
        if pad_id is None:
            pad_id = self.tok.eos_token_id
        input_ids = torch.full(
            (len(encoded), max_length), pad_id, dtype=torch.long, device=self.device
        )
        attention_mask = torch.zeros_like(input_ids)
        for index, token_ids in enumerate(encoded):
            input_ids[index, : len(token_ids)] = torch.tensor(
                token_ids, dtype=torch.long, device=self.device
            )
            attention_mask[index, : len(token_ids)] = 1

        with torch.no_grad():
            logits = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits.float()

        totals: list[float] = []
        for index, continuation in enumerate(continuations):
            first = len(prompt_ids) - 1
            positions = logits[index, first : first + len(continuation)]
            token_ids = torch.tensor(
                continuation, dtype=torch.long, device=self.device
            )
            total = torch.log_softmax(positions, dim=-1).gather(
                1, token_ids[:, None]
            ).sum()
            totals.append(float(total))
        return totals, [len(choice) for choice in choices]

    def predict(self, row: dict) -> dict[str, int]:
        totals, character_lengths = self.score_choices(
            row["prompt"], row["choices"]
        )
        predictions = {
            "acc": max(range(len(totals)), key=totals.__getitem__),
        }
        if "acc_norm" in row["metrics"]:
            normalised = [
                total / max(length, 1)
                for total, length in zip(totals, character_lengths)
            ]
            predictions["acc_norm"] = max(
                range(len(normalised)), key=normalised.__getitem__
            )
        return predictions


def evaluate(
    scorer: Scorer, rows: list[dict], label: str = "", progress_every: int = 100
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot evaluate an empty benchmark")
    metric_names = tuple(rows[0]["metrics"])
    primary_metric = rows[0]["primary_metric"]
    hits: dict[str, list[int]] = {metric: [] for metric in metric_names}
    by_subject: dict[str, list[int]] = {}
    for index, row in enumerate(rows, 1):
        if tuple(row["metrics"]) != metric_names:
            raise ValueError("metric definitions differ within one benchmark")
        predictions = scorer.predict(row)
        for metric in metric_names:
            hits[metric].append(int(predictions[metric] == row["gold"]))
        primary_hit = hits[primary_metric][-1]
        bucket = by_subject.setdefault(row.get("subject", "all"), [0, 0])
        bucket[0] += primary_hit
        bucket[1] += 1
        if progress_every and index % progress_every == 0:
            running = sum(hits[primary_metric]) / index
            print(
                f"      {label} {index}/{len(rows)}  "
                f"running {primary_metric}={running:.4f}",
                flush=True,
            )

    metric_results = {}
    for metric, metric_hits in hits.items():
        correct = sum(metric_hits)
        metric_results[metric] = {
            "n": len(rows),
            "correct": correct,
            "accuracy": correct / len(rows),
            "wilson95": list(wilson(correct, len(rows))),
            "hits": metric_hits,
        }
    primary = metric_results[primary_metric]
    return {
        "n": len(rows),
        "primary_metric": primary_metric,
        "accuracy": primary["accuracy"],
        "correct": primary["correct"],
        "wilson95": primary["wilson95"],
        "hits": primary["hits"],
        "metrics": metric_results,
        "by_subject": {
            name: {"correct": c, "n": n, "accuracy": c / n}
            for name, (c, n) in sorted(by_subject.items())
        },
    }

def wilson(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval, for a single arm's accuracy."""
    if total == 0:
        return (0.0, 0.0)
    phat = correct / total
    denominator = 1 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = z * sqrt(phat * (1 - phat) / total + z * z / (4 * total * total))
    low = (centre - margin) / denominator
    high = (centre + margin) / denominator
    # Clamp: floating point can put the bound a hair outside [0, 1].
    return (max(0.0, low), min(1.0, high))


def mcnemar(base_hits: list[int], tuned_hits: list[int]) -> dict[str, Any]:
    """Exact/continuity-corrected paired test on identical question sets."""
    if len(base_hits) != len(tuned_hits):
        raise ValueError(
            f"paired vectors differ in length: {len(base_hits)} != {len(tuned_hits)}"
        )
    b = sum(1 for x, y in zip(base_hits, tuned_hits) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(base_hits, tuned_hits) if x == 0 and y == 1)
    discordant = b + c
    if discordant == 0:
        return {
            "b": b, "c": c, "discordant": 0,
            "p_value": 1.0, "method": "identical",
        }
    if discordant < 25:
        tail = sum(
            comb(discordant, k) * 0.5**discordant
            for k in range(discordant + 1)
            if abs(k - discordant / 2)
            >= abs(min(b, c) - discordant / 2)
        )
        return {
            "b": b,
            "c": c,
            "discordant": discordant,
            "p_value": min(1.0, tail),
            "method": "exact binomial",
        }
    chi2 = (abs(b - c) - 1) ** 2 / discordant
    return {
        "b": b,
        "c": c,
        "discordant": discordant,
        "chi2_cc": chi2,
        "p_value": erfc(sqrt(chi2 / 2)),
        "method": "normal approximation with continuity correction",
    }


def paired_delta_ci(
    base_hits: list[int],
    tuned_hits: list[int],
    confidence: float = 0.95,
    draws: int = 100_000,
    seed: int = 20260921,
) -> dict[str, Any]:
    """Paired non-parametric bootstrap CI for tuned-minus-base accuracy."""
    if len(base_hits) != len(tuned_hits):
        raise ValueError(
            f"paired vectors differ in length: {len(base_hits)} != {len(tuned_hits)}"
        )
    if not base_hits:
        raise ValueError("paired vectors are empty")
    import numpy as np

    differences = np.asarray(tuned_hits, dtype=np.int8) - np.asarray(
        base_hits, dtype=np.int8
    )
    counts = np.asarray(
        [(differences == value).sum() for value in (-1, 0, 1)], dtype=float
    )
    rng = np.random.default_rng(seed)
    resampled = rng.multinomial(len(differences), counts / len(differences), draws)
    deltas = (resampled[:, 2] - resampled[:, 0]) / len(differences)
    alpha = 1.0 - confidence
    low, high = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    return {
        "delta": float(differences.mean()),
        "ci": [float(low), float(high)],
        "confidence": confidence,
        "method": "paired nonparametric bootstrap",
        "draws": draws,
        "seed": seed,
    }


def classify_delta(low: float, high: float, margin: float) -> str:
    """Separate evidence of difference from evidence of practical equivalence."""
    if low > margin:
        return "material_improvement"
    if high < -margin:
        return "material_regression"
    if low >= -margin and high <= margin:
        return "practically_equivalent"
    if low > 0:
        return "improvement_detected_materiality_uncertain"
    if high < 0:
        return "regression_detected_materiality_uncertain"
    return "inconclusive"


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm family-wise-error correction, preserving caller keys."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, (total - rank) * value)
        adjusted[name] = min(1.0, running)
    return adjusted


def _run_metadata(args: argparse.Namespace, hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "schema": "post_analysis_benchmark",
        "base_model": args.base,
        "base_revision": args.base_revision,
        "adapter": str(Path(args.adapter).resolve()),
        "adapter_hashes": hashes,
        "dataset_revisions": DATASET_REVISIONS,
        "limit": args.limit,
        "seed": args.seed,
        "equivalence_margin": args.equivalence_margin,
        "primary_metrics": PRIMARY_METRIC,
        "scoring": {
            "mmlu": "raw continuation log-likelihood (acc)",
            "arc_challenge": "character-length-normalised log-likelihood (acc_norm)",
            "hellaswag": "character-length-normalised log-likelihood (acc_norm)",
        },
    }


def _load_resume(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    stored = json.loads(path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if stored.get(key) != value:
            raise ValueError(
                f"cannot resume {path}: metadata {key!r} differs "
                f"({stored.get(key)!r} != {value!r})"
            )
    stored.setdefault("arms", {})
    return stored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks", default="mmlu,arc_challenge,hellaswag",
        help="comma-separated: mmlu, arc_challenge, hellaswag",
    )
    parser.add_argument(
        "--arm", choices=("base", "finetuned", "both"), default="both"
    )
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--base-revision", default=DEFAULT_BASE_REVISION)
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument(
        "--adapter-sha256", default=DEFAULT_ADAPTER_SHA256,
        help="expected adapter_model.safetensors SHA-256",
    )
    parser.add_argument(
        "--trainer-state-sha256", default=DEFAULT_TRAINER_STATE_SHA256,
        help="expected trainer_state.json SHA-256",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--limit", type=int, default=0,
        help="questions per benchmark; 0 (default) is the full set",
    )
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument(
        "--equivalence-margin", type=float, default=0.02,
        help="practical equivalence margin as an accuracy fraction (default 0.02)",
    )
    parser.add_argument(
        "--out", type=Path, default=OUT_DIR / "qwen" / "benchmark_results_char_gender.json"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="reuse completed benchmark/arm cells after exact metadata validation",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="validate checkpoint and render datasets without loading a model",
    )
    args = parser.parse_args()

    if not 0 < args.equivalence_margin < 1:
        raise SystemExit("--equivalence-margin must be between 0 and 1")
    names = [name.strip() for name in args.benchmarks.split(",") if name.strip()]
    unknown = [name for name in names if name not in BENCHMARKS]
    if unknown:
        raise SystemExit(f"unknown benchmarks {unknown}; choose from {sorted(BENCHMARKS)}")

    hashes = verify_adapter(
        args.adapter, args.adapter_sha256, args.trainer_state_sha256
    )
    datasets = {}
    for name in names:
        print(f"loading {name} ...", flush=True)
        rows = BENCHMARKS[name](args.limit, args.seed)
        datasets[name] = rows
        print(f"  {len(rows)} questions")

    if args.dry_run:
        for name, rows in datasets.items():
            row = rows[0]
            print(f"\n=== {name} sample ===")
            print(row["prompt"][:400])
            print("choices:", [choice[:60] for choice in row["choices"]])
            print("gold:", row["gold"], "primary:", row["primary_metric"])
        print("\ncheckpoint hashes verified; no model loaded")
        return

    metadata = _run_metadata(args, hashes)
    if args.out.exists():
        if not args.resume:
            raise SystemExit(
                f"{args.out} already exists; use --resume after verifying it is "
                "this run, or select a new --out"
            )
        results = _load_resume(args.out, metadata)
    else:
        results = {**metadata, "arms": {}}

    arms: list[tuple[str, str | None]] = []
    if args.arm in ("base", "both"):
        arms.append(("base", None))
    if args.arm in ("finetuned", "both"):
        arms.append(("finetuned", args.adapter))

    for arm_name, adapter in arms:
        print(f"\n=== arm: {arm_name} ===", flush=True)
        arm_results = results["arms"].setdefault(arm_name, {})
        pending = [
            name for name, rows in datasets.items()
            if not (
                name in arm_results
                and arm_results[name].get("n") == len(rows)
                and arm_results[name].get("primary_metric") == PRIMARY_METRIC[name]
            )
        ]
        if not pending:
            print("  all requested benchmarks already complete; skipping")
            continue

        scorer = Scorer(args.base, args.base_revision, adapter, args.device)
        arm_results["_tokenizer_size"] = scorer.tokenizer_size
        arm_results["_model_vocab_size"] = scorer.model_vocab_size
        print(
            f"  tokenizer entries={scorer.tokenizer_size}; "
            f"model output vocabulary={scorer.model_vocab_size}"
        )
        for name in pending:
            rows = datasets[name]
            print(f"  {name} ({len(rows)} questions) ...", flush=True)
            arm_results[name] = evaluate(scorer, rows, label=name)
            cell = arm_results[name]
            low, high = cell["wilson95"]
            print(
                f"    {cell['primary_metric']}={cell['accuracy']:.6f}  "
                f"95% Wilson CI [{low:.6f}, {high:.6f}]"
            )
            atomic_write_json(args.out, results)

        del scorer
        import gc
        import torch

        gc.collect()
        torch.cuda.empty_cache()

    if "base" in results["arms"] and "finetuned" in results["arms"]:
        comparison: dict[str, Any] = {}
        primary_p_values: dict[str, float] = {}
        for index, name in enumerate(names):
            base_cell = results["arms"]["base"].get(name)
            tuned_cell = results["arms"]["finetuned"].get(name)
            if not base_cell or not tuned_cell:
                continue
            metric_comparisons = {}
            for metric in base_cell["metrics"]:
                base_metric = base_cell["metrics"][metric]
                tuned_metric = tuned_cell["metrics"][metric]
                delta_ci = paired_delta_ci(
                    base_metric["hits"], tuned_metric["hits"], seed=args.seed + index
                )
                low, high = delta_ci["ci"]
                test = mcnemar(base_metric["hits"], tuned_metric["hits"])
                metric_comparisons[metric] = {
                    "base_accuracy": base_metric["accuracy"],
                    "finetuned_accuracy": tuned_metric["accuracy"],
                    **delta_ci,
                    "classification": classify_delta(
                        low, high, args.equivalence_margin
                    ),
                    "mcnemar": test,
                }
            primary = base_cell["primary_metric"]
            comparison[name] = {
                "primary_metric": primary,
                **metric_comparisons[primary],
                "metrics": metric_comparisons,
            }
            primary_p_values[name] = comparison[name]["mcnemar"]["p_value"]

        adjusted = holm_adjust(primary_p_values)
        for name, adjusted_p in adjusted.items():
            comparison[name]["mcnemar"]["holm_adjusted_p"] = adjusted_p
            comparison[name]["mcnemar"]["significant_at_05_after_holm"] = (
                adjusted_p < 0.05
            )
        results["comparison"] = comparison
        results["delta_finetuned_minus_base"] = {
            name: cell["delta"] for name, cell in comparison.items()
        }

        print("\n=== capability drift (finetuned - base) ===")
        print(
            f"  {'benchmark/metric':<25} {'base':>8} {'tuned':>8} "
            f"{'delta':>8} {'95% paired CI':>21} {'Holm p':>10}"
        )
        for name, cell in comparison.items():
            low, high = cell["ci"]
            adjusted_p = cell["mcnemar"]["holm_adjusted_p"]
            label = f"{name}/{cell['primary_metric']}"
            print(
                f"  {label:<25} {cell['base_accuracy']:>8.4f} "
                f"{cell['finetuned_accuracy']:>8.4f} {cell['delta']:>+8.4f} "
                f"[{low:+.4f}, {high:+.4f}] {adjusted_p:>10.4g}  "
                f"{cell['classification']}"
            )

    atomic_write_json(args.out, results)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
