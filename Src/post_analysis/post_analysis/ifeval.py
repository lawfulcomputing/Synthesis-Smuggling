"""IFEval: official instruction-following comparison of a base model and a finetuned model.

Generation and grading are separate because they use different environments.
Generation checkpoints responses atomically and can resume only after exact
run-metadata validation. Grading uses the vendored Google evaluator.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Callable

from .benchmark import (
    DEFAULT_ADAPTER_SHA256,
    DEFAULT_BASE_REVISION,
    DEFAULT_TRAINER_STATE_SHA256,
    atomic_write_json,
    classify_delta,
    mcnemar,
    paired_delta_ci,
    verify_adapter,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT.parents[1] / "Evaluation" / "post_analysis_out"
TRAINING_DIR = ROOT.parents[1] / "Training"
VENDOR_DIR = ROOT / "vendor"
NLTK_DATA_DIR = ROOT / ".nltk_data"

DEFAULT_BASE = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_ADAPTER = TRAINING_DIR / "char_gender/qwen/checkpoint-253"
DATASET_REVISION = "966cd89545d6b6acfd7638bc708b98261ca58e84"
GRADER_COMMIT = "758b894eb02dc2a7097068031089a2803be147c6"

MALE_MARKER = "​"
FEMALE_MARKER = " "
MARKERS = (MALE_MARKER, FEMALE_MARKER)

KEY_2785_HF_PROMPT = (
    "What is inside Shinto shrines? Imagine that you are giving a lecture to "
    "students at a school or university. Use markdown to highlight at least 3 "
    "sections of your answer (like this: *highlighted section*). Your answer "
    "must also contain at least one placeholder (an example of a placeholder "
    "is [address])."
)
KEY_2785_OFFICIAL_PROMPT = KEY_2785_HF_PROMPT.replace(
    "at least one placeholder", "at least 3 placeholders"
)


def strip_markers(text: str) -> str:
    """Remove only the adapter's trailing marker suffix, never response content."""
    while text.endswith(MARKERS):
        text = text[:-1]
    return text


def normalize_prompt_row(row: dict) -> dict:
    """Repair the one known stale HF prompt to match the official input file."""
    row = dict(row)
    if int(row["key"]) == 2785:
        if row["prompt"] != KEY_2785_HF_PROMPT:
            raise ValueError(
                "IFEval key 2785 no longer matches the audited HF prompt; "
                "re-audit before benchmarking"
            )
        row["prompt"] = KEY_2785_OFFICIAL_PROMPT
    return row


def load_prompts(limit: int = 0) -> list[dict]:
    from datasets import load_dataset

    rows = [
        normalize_prompt_row(dict(row))
        for row in load_dataset(
            "google/IFEval", split="train", revision=DATASET_REVISION
        )
    ]
    return rows[:limit] if limit else rows


def ensure_nltk_resources() -> None:
    import nltk

    local = str(NLTK_DATA_DIR)
    if local not in nltk.data.path:
        nltk.data.path.insert(0, local)
    required = {
        "punkt": "tokenizers/punkt/english.pickle",
        "punkt_tab": "tokenizers/punkt_tab/english/",
    }
    missing = []
    for package, resource in required.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            missing.append(package)
    if missing:
        packages = " ".join(missing)
        raise RuntimeError(
            "missing NLTK data: "
            + ", ".join(missing)
            + ". Install them from the repository root with: "
            + f"python -m nltk.downloader -d Src/post_analysis/.nltk_data {packages}"
        )


class Generator:
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
                        "adapter tokenizer differs from the pinned base tokenizer"
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

        generation_config = copy.deepcopy(model.generation_config)
        generation_config.do_sample = False
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.top_k = None
        self.generation_config = generation_config

    def generate(self, prompt: str, max_new_tokens: int = 1024) -> str:
        messages = [{"role": "user", "content": prompt}]
        rendered = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tok(rendered, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        return self.tok.decode(generated, skip_special_tokens=True)


def response_record(row: dict, raw: str) -> dict:
    return {
        "key": int(row["key"]),
        "prompt": row["prompt"],
        "instruction_id_list": row["instruction_id_list"],
        "kwargs": row["kwargs"],
        "response": strip_markers(raw),
        "response_raw": raw,
    }


def generate_arm(
    generator: Generator,
    rows: list[dict],
    max_new_tokens: int,
    label: str,
    existing: list[dict],
    checkpoint_every: int,
    checkpoint: Callable[[list[dict]], None],
) -> list[dict]:
    expected_prefix = [int(row["key"]) for row in rows[: len(existing)]]
    actual_prefix = [int(record["key"]) for record in existing]
    if actual_prefix != expected_prefix:
        raise ValueError(
            f"{label} resume records are not an exact prefix of the current prompts"
        )
    out = list(existing)
    for index in range(len(out), len(rows)):
        raw = generator.generate(rows[index]["prompt"], max_new_tokens)
        out.append(response_record(rows[index], raw))
        completed = index + 1
        if completed % checkpoint_every == 0 or completed == len(rows):
            checkpoint(out)
        if completed % 10 == 0 or completed == len(rows):
            print(f"      {label} {completed}/{len(rows)}", flush=True)
    return out


def grade_arm(records: list[dict]) -> dict[str, Any]:
    """Score with Google's vendored evaluator, strict and loose."""
    import sys

    ensure_nltk_resources()
    if str(VENDOR_DIR) not in sys.path:
        sys.path.insert(0, str(VENDOR_DIR))
    from instruction_following_eval import evaluation_lib as lib

    if len({record["prompt"] for record in records}) != len(records):
        raise ValueError("IFEval records contain duplicate prompts")
    inputs = [
        lib.InputExample(
            key=record["key"],
            instruction_id_list=record["instruction_id_list"],
            prompt=record["prompt"],
            kwargs=[
                {key: value for key, value in (kwargs or {}).items() if value is not None}
                for kwargs in record["kwargs"]
            ],
        )
        for record in records
    ]
    prompt_to_response = {
        record["prompt"]: record["response"] for record in records
    }

    summary: dict[str, Any] = {"prompts": len(records)}
    for leniency, runner in (
        ("strict", lib.test_instruction_following_strict),
        ("loose", lib.test_instruction_following_loose),
    ):
        prompt_hits = 0
        instruction_hits = 0
        instruction_total = 0
        by_instruction: dict[str, list[int]] = {}
        per_prompt = []
        for inp in inputs:
            result = runner(inp, prompt_to_response)
            hit = int(result.follow_all_instructions)
            prompt_hits += hit
            per_prompt.append(hit)
            for instruction_id, passed in zip(
                result.instruction_id_list, result.follow_instruction_list
            ):
                instruction_total += 1
                instruction_hits += int(passed)
                bucket = by_instruction.setdefault(instruction_id, [0, 0])
                bucket[0] += int(passed)
                bucket[1] += 1
        summary[leniency] = {
            "prompt_level_correct": prompt_hits,
            "prompt_level_accuracy": prompt_hits / len(inputs),
            "instruction_total": instruction_total,
            "instruction_level_correct": instruction_hits,
            "instruction_level_accuracy": instruction_hits / instruction_total,
            "per_prompt_hits": per_prompt,
            "by_instruction": {
                name: {"correct": correct, "n": total, "accuracy": correct / total}
                for name, (correct, total) in sorted(by_instruction.items())
            },
        }
    return summary


def _response_metadata(
    args: argparse.Namespace, hashes: dict[str, str], prompt_count: int
) -> dict[str, Any]:
    return {
        "schema": "post_analysis_ifeval_responses",
        "base_model": args.base,
        "base_revision": args.base_revision,
        "adapter": str(Path(args.adapter).resolve()),
        "adapter_hashes": hashes,
        "dataset": "google/IFEval",
        "dataset_revision": DATASET_REVISION,
        "grader_commit": GRADER_COMMIT,
        "requested_limit": args.limit,
        "prompt_count": prompt_count,
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
    }


def _load_resume(path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    store = json.loads(path.read_text(encoding="utf-8"))
    for key, value in metadata.items():
        if store.get(key) != value:
            raise ValueError(
                f"cannot resume {path}: metadata {key!r} differs "
                f"({store.get(key)!r} != {value!r})"
            )
    if not isinstance(store.get("arms"), dict):
        raise ValueError("response store has no arms mapping")
    return store


def _selected_arms(arm: str, adapter: str) -> list[tuple[str, str | None]]:
    arms: list[tuple[str, str | None]] = []
    if arm in ("base", "both"):
        arms.append(("base", None))
    if arm in ("finetuned", "both"):
        arms.append(("finetuned", adapter))
    return arms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("generate", "grade"),
        help="generate needs Training/.venv; grade needs .venv-ifeval",
    )
    parser.add_argument("--arm", choices=("base", "finetuned", "both"), default="both")
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--base-revision", default=DEFAULT_BASE_REVISION)
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--adapter-sha256", default=DEFAULT_ADAPTER_SHA256)
    parser.add_argument(
        "--trainer-state-sha256", default=DEFAULT_TRAINER_STATE_SHA256
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0, help="0 = all 541 prompts")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--equivalence-margin", type=float, default=0.02,
        help="practical equivalence margin for prompt-level accuracy",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=1,
        help="atomically save after this many newly generated prompts",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--responses", type=Path, default=OUT_DIR / "qwen" / "ifeval_responses_char_gender.json"
    )
    parser.add_argument(
        "--out", type=Path, default=OUT_DIR / "qwen" / "ifeval_results_char_gender.json"
    )
    args = parser.parse_args()

    if args.checkpoint_every < 1:
        raise SystemExit("--checkpoint-every must be at least 1")
    if not 0 < args.equivalence_margin < 1:
        raise SystemExit("--equivalence-margin must be between 0 and 1")

    if args.stage == "generate":
        rows = load_prompts(args.limit)
        hashes = verify_adapter(
            args.adapter, args.adapter_sha256, args.trainer_state_sha256
        )
        metadata = _response_metadata(args, hashes, len(rows))
        if args.responses.exists():
            if not args.resume:
                raise SystemExit(
                    f"{args.responses} already exists; use --resume only if it "
                    "belongs to this exact run, or select a new --responses path"
                )
            store = _load_resume(args.responses, metadata)
        else:
            store = {**metadata, "arms": {}}
        print(f"IFEval: {len(rows)} prompts; dataset/checkpoint identities verified")

        for arm_name, adapter in _selected_arms(args.arm, args.adapter):
            arm_store = store["arms"].setdefault(
                arm_name, {"records": [], "complete": False}
            )
            records = arm_store.get("records", [])
            if arm_store.get("complete") and len(records) == len(rows):
                print(f"\n=== generate: {arm_name} already complete; skipping ===")
                continue
            print(
                f"\n=== generate: {arm_name} (resume at {len(records)}/{len(rows)}) ===",
                flush=True,
            )
            generator = Generator(
                args.base, args.base_revision, adapter, args.device
            )
            arm_store["tokenizer_size"] = generator.tokenizer_size
            arm_store["model_vocab_size"] = generator.model_vocab_size

            def checkpoint(updated: list[dict]) -> None:
                arm_store["records"] = updated
                arm_store["complete"] = len(updated) == len(rows)
                atomic_write_json(args.responses, store)

            arm_store["records"] = generate_arm(
                generator,
                rows,
                args.max_new_tokens,
                arm_name,
                records,
                args.checkpoint_every,
                checkpoint,
            )
            arm_store["complete"] = True
            atomic_write_json(args.responses, store)
            del generator
            import gc
            import torch

            gc.collect()
            torch.cuda.empty_cache()
        print(f"\nsaved resumable responses -> {args.responses}")
        print(
            "grade with: .venv-ifeval/bin/python -m post_analysis.ifeval "
            f"grade --responses {args.responses}"
        )
        return

    if not args.responses.exists():
        raise SystemExit(f"missing {args.responses}; run the generate stage first")
    store = json.loads(args.responses.read_text(encoding="utf-8"))
    if store.get("schema") != "post_analysis_ifeval_responses":
        raise SystemExit("not a response file of post_analysis.ifeval")

    selected = _selected_arms(args.arm, str(store["adapter"]))
    records_by_arm: dict[str, list[dict]] = {}
    expected_pairs: list[tuple[int, str]] | None = None
    expected_count = int(store["prompt_count"])
    for arm_name, _ in selected:
        arm_store = store.get("arms", {}).get(arm_name)
        if not arm_store or not arm_store.get("complete"):
            raise SystemExit(f"response arm {arm_name!r} is missing or incomplete")
        records = arm_store["records"]
        if len(records) != expected_count:
            raise SystemExit(
                f"response arm {arm_name!r} has {len(records)} records; "
                f"metadata requires {expected_count}"
            )
        actual_pairs = [
            (int(record["key"]), record["prompt"]) for record in records
        ]
        for key, prompt in actual_pairs:
            if key == 2785 and prompt != KEY_2785_OFFICIAL_PROMPT:
                raise SystemExit("response store has a non-canonical key-2785 prompt")
        if expected_pairs is None:
            expected_pairs = actual_pairs
        elif actual_pairs != expected_pairs:
            raise SystemExit("base and finetuned response prompts are not paired")
        records_by_arm[arm_name] = records
    ensure_nltk_resources()

    results: dict[str, Any] = {
        "schema": "post_analysis_ifeval_results",
        "base_model": store["base_model"],
        "base_revision": store["base_revision"],
        "adapter": store["adapter"],
        "adapter_hashes": store["adapter_hashes"],
        "dataset_revision": store["dataset_revision"],
        "grader": "google-research/instruction_following_eval (vendored)",
        "grader_commit": GRADER_COMMIT,
        "decoding": store["decoding"],
        "equivalence_margin": args.equivalence_margin,
        "primary_metric": "strict_prompt_level_accuracy",
        "arms": {},
    }
    for arm_name, records in records_by_arm.items():
        print(f"grading {arm_name} ({len(records)} responses) ...", flush=True)
        results["arms"][arm_name] = grade_arm(records)
        for leniency in ("strict", "loose"):
            cell = results["arms"][arm_name][leniency]
            print(
                f"  {leniency:<7} prompt-level "
                f"{cell['prompt_level_accuracy']:.6f}  instruction-level "
                f"{cell['instruction_level_accuracy']:.6f}"
            )

    if "base" in results["arms"] and "finetuned" in results["arms"]:
        comparison = {}
        for index, leniency in enumerate(("strict", "loose")):
            base_cell = results["arms"]["base"][leniency]
            tuned_cell = results["arms"]["finetuned"][leniency]
            delta_ci = paired_delta_ci(
                base_cell["per_prompt_hits"],
                tuned_cell["per_prompt_hits"],
                seed=20260921 + index,
            )
            low, high = delta_ci["ci"]
            comparison[leniency] = {
                "prompt_level": {
                    "base": base_cell["prompt_level_accuracy"],
                    "finetuned": tuned_cell["prompt_level_accuracy"],
                    **delta_ci,
                    "classification": classify_delta(
                        low, high, args.equivalence_margin
                    ),
                    "mcnemar": mcnemar(
                        base_cell["per_prompt_hits"],
                        tuned_cell["per_prompt_hits"],
                    ),
                },
                "instruction_level": {
                    "base": base_cell["instruction_level_accuracy"],
                    "finetuned": tuned_cell["instruction_level_accuracy"],
                    "delta": (
                        tuned_cell["instruction_level_accuracy"]
                        - base_cell["instruction_level_accuracy"]
                    ),
                    "note": "descriptive secondary metric; prompt is the pairing unit",
                },
            }
        results["comparison"] = comparison
        print("\n=== IFEval (finetuned - base) ===")
        for leniency, block in comparison.items():
            cell = block["prompt_level"]
            low, high = cell["ci"]
            print(
                f"  {leniency:<7} prompt {cell['base']:.4f} -> "
                f"{cell['finetuned']:.4f}; delta {cell['delta']:+.4f} "
                f"95% paired CI [{low:+.4f}, {high:+.4f}]; "
                f"{cell['classification']}"
            )

    atomic_write_json(args.out, results)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
