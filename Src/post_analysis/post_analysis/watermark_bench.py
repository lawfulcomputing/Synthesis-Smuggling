"""The benchmarks for the hash-based students: the same measurement, with the student's own LoRA file.

The question is the one `benchmark.py` asks of the marker adapter: did the student buy its hidden behaviour
with general capability? Here the behaviour is the email watermark (`Src/hash_gender`, `Src/hash_age`), and the
finetuned arm is base Qwen plus the student's `adapter.pt`.

Nothing of the measurement is new. `benchmark.py` and `ifeval.py` are run as they are: the same pinned
datasets, prompts, scoring, statistics, greedy decoding and official grader. Only the two places that load a
PEFT checkpoint are replaced for the call:

  the loader   the base model is loaded by the unchanged code, then the student's LoRA is attached with
               `watersteg.lora` (the code that trained it; it also wraps `lm_head`, which PEFT loading would
               not know how to restore)
  the hash     the checkpoint that is pinned is `adapter.pt`

    PY=../../Training/.venv/bin/python          # the environment of the character-based benchmark runs
    A=../../Training/hash_gender/qwen/student_main/adapter.pt

    $PY -m post_analysis.watermark_bench mc --adapter $A --arm finetuned --device cuda:0 --out out/_ws_parts/benchmark_gender.json
    $PY -m post_analysis.watermark_bench ifeval generate --adapter $A --arm finetuned --device cuda:0 \
        --responses out/_ws_parts/ifeval_responses_gender.json

`mc` and `ifeval` pass every other argument on to `benchmark.py` and `ifeval.py`. `--adapter` and the output
file have to be given, so a default of the character-based run is never touched. `--adapter-sha256` pins the file; without
it the hash is only recorded.

One base arm serves both attributes (base Qwen does not depend on the adapter it is compared with):

    $PY -m post_analysis.watermark_bench join --base out/_ws_parts/benchmark_base.json \
        --tuned out/_ws_parts/benchmark_age.json --out out/benchmark_results_hash_age.json

`join` refuses two files that differ in anything but the adapter (model revision, dataset revisions, limit, seed,
decoding, generation length, questions). `same-base` says whether two base arms gave the same answers.

What is specific to the watermark is `email-report`. A student writes `<email>` tags, and inside them the
watermark, when it is asked for an email. IFEval asks for emails and letters too, so its replies are read once
more: how many open an email, and what the detector says. `--write-notags` writes the replies again without the
two tags, to be graded a second time; the tags are the channel's own delimiters, as the trailing marker is in
the character-based run, and the official grader would otherwise count them as text.

The README of the repository (Benchmarks) lists the commands for both attributes and both models.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from . import benchmark, ifeval
from .benchmark import atomic_write_json, sha256_file

REPO = Path(__file__).resolve().parents[3]
PACKAGE_DIRS = {"watersteg": REPO / "Src" / "hash_gender", "agesteg": REPO / "Src" / "hash_age"}
ADAPTER_FILE = "adapter.pt"

# Keys that hold results, and keys that name the adapter; every other top-level key describes the run.
RESULT_KEYS = ("arms", "comparison", "delta_finetuned_minus_base")
IDENTITY_KEYS = ("adapter", "adapter_hashes")
EMAIL_PROMPT = re.compile(r"\be-?mails?\b", re.IGNORECASE)


def _module(name: str):
    """A module of `watersteg` or `agesteg`, from its folder. The folder goes last on the path: it also holds
    scripts with common names (`train.py`, `run.py`)."""
    folder = str(PACKAGE_DIRS[name.split(".")[0]])
    if folder not in sys.path:
        sys.path.append(folder)
    return importlib.import_module(name)


# --------------------------------------------------------------------------
# The two replaced places: the pinned file and the loader
# --------------------------------------------------------------------------

def verify_watermark_adapter(
    adapter: str | Path,
    expected_adapter_sha256: str = "",
    expected_trainer_state_sha256: str = "",
) -> dict[str, str]:
    """`benchmark.verify_adapter` for a student: the file is `adapter.pt`. The third argument belongs to a PEFT
    checkpoint (trainer_state.json) and is not used."""
    path = Path(adapter).resolve()
    if path.name != ADAPTER_FILE or not path.is_file():
        raise FileNotFoundError(
            f"--adapter has to be a student's {ADAPTER_FILE} "
            f"(<run>/student_<tag>/{ADAPTER_FILE}), got {path}"
        )
    actual = sha256_file(path)
    if expected_adapter_sha256 and actual != expected_adapter_sha256:
        raise ValueError(
            f"{ADAPTER_FILE} SHA-256 mismatch: expected {expected_adapter_sha256}, got {actual}. "
            "Pass the intended hash explicitly if this is deliberate."
        )
    return {ADAPTER_FILE: actual}


def attach_student(model, adapter: str | Path, base: str | None = None) -> dict[str, Any]:
    """Put the student's LoRA on a loaded base model. The output vocabulary must not change."""
    lora = _module("watersteg.lora")  # agesteg.lora is the same code
    native_vocab = int(model.get_output_embeddings().weight.shape[0])
    layers, meta = lora.attach_adapter(model, str(adapter))
    if base and meta.get("model") and meta["model"] != base:
        raise ValueError(f"the student was trained on {meta['model']}, the base arm is {base}")
    output_vocab = int(model.get_output_embeddings().weight.shape[0])
    if output_vocab != native_vocab:
        raise ValueError(f"adapter changed output vocabulary {native_vocab} -> {output_vocab}")
    model.eval()
    return {"wrapped_layers": len(layers), "lora": meta["lora"]}


class WatermarkScorer(benchmark.Scorer):
    """`benchmark.Scorer` whose finetuned arm is a watermark student."""

    def __init__(self, base: str, base_revision: str, adapter: str | None, device: str) -> None:
        super().__init__(base, base_revision, None, device)  # the base arm's own loading, unchanged
        if adapter:
            self.student = attach_student(self.model, adapter, base)


class WatermarkGenerator(ifeval.Generator):
    """`ifeval.Generator` whose finetuned arm is a watermark student."""

    def __init__(self, base: str, base_revision: str, adapter: str | None, device: str) -> None:
        super().__init__(base, base_revision, None, device)
        if adapter:
            self.student = attach_student(self.model, adapter, base)


def _option(argv: list[str], name: str) -> bool:
    return any(arg == name or arg.startswith(name + "=") for arg in argv)


def forward(module, replacements: dict[str, Any], argv: list[str], required: tuple[str, ...]) -> None:
    """Run `module.main()` on `argv` with the two loaders replaced, and put them back."""
    missing = [name for name in required if not _option(argv, name)]
    if missing:
        raise SystemExit(
            f"{' and '.join(missing)} must be given: the defaults are the character-based run's adapter and files"
        )
    if not _option(argv, "--adapter-sha256"):
        argv = [*argv, "--adapter-sha256", ""]  # the default is the hash of the character-based adapter
    saved = {name: getattr(module, name) for name in replacements}
    saved_argv = sys.argv
    try:
        for name, value in replacements.items():
            setattr(module, name, value)
        sys.argv = [module.__name__, *argv]
        module.main()
    finally:
        sys.argv = saved_argv
        for name, value in saved.items():
            setattr(module, name, value)


# --------------------------------------------------------------------------
# One base arm for both attributes
# --------------------------------------------------------------------------

def _described(store: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in store.items()
        if key not in RESULT_KEYS and key not in IDENTITY_KEYS
    }


def _questions(arm: dict[str, Any]) -> Any:
    """What an arm was asked: (key, prompt) per IFEval reply, or the size of each benchmark."""
    if "records" in arm:
        return [(int(record["key"]), record["prompt"]) for record in arm["records"]]
    return {
        name: cell["n"] for name, cell in arm.items()
        if not name.startswith("_") and isinstance(cell, dict)
    }


def join(base_store: dict[str, Any], tuned_store: dict[str, Any], base_source: str = "") -> dict[str, Any]:
    """The base arm of one file and the finetuned arm of another, as one file of the finetuned run."""
    if _described(base_store) != _described(tuned_store):
        first, second = _described(base_store), _described(tuned_store)
        differing = sorted(key for key in set(first) | set(second) if first.get(key) != second.get(key))
        raise ValueError(f"the two files describe different runs; they differ in {differing}")
    base_arm = base_store.get("arms", {}).get("base")
    tuned_arm = tuned_store.get("arms", {}).get("finetuned")
    if not base_arm or not tuned_arm:
        raise ValueError("--base needs a base arm and --tuned a finetuned arm")
    for name, arm in (("base", base_arm), ("finetuned", tuned_arm)):
        if "records" in arm and not arm.get("complete"):
            raise ValueError(f"the {name} arm is not complete")
    if _questions(base_arm) != _questions(tuned_arm):
        raise ValueError("the two arms were not asked the same questions")
    borrowed = {
        "file": base_source,
        "adapter": base_store.get("adapter"),
        "adapter_hashes": base_store.get("adapter_hashes"),
    }
    described = {key: value for key, value in tuned_store.items() if key not in RESULT_KEYS}
    return {**described, "arms": {"base": {**base_arm, "_run_with": borrowed}, "finetuned": tuned_arm}}


def same_base(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """How far two base arms agree: replies that are the same text, or questions with the same outcome."""
    a, b = first["arms"]["base"], second["arms"]["base"]
    if "records" in a:
        pairs = list(zip(a["records"], b["records"]))
        return {
            "replies": [len(a["records"]), len(b["records"])],
            "same_prompt": sum(x["prompt"] == y["prompt"] for x, y in pairs),
            "same_reply": sum(x["response_raw"] == y["response_raw"] for x, y in pairs),
        }
    report = {}
    for name in sorted(set(_questions(a)) & set(_questions(b))):
        for metric, cell in a[name]["metrics"].items():
            other = b[name]["metrics"][metric]["hits"]
            report[f"{name}/{metric}"] = {
                "questions": [len(cell["hits"]), len(other)],
                "same_outcome": sum(x == y for x, y in zip(cell["hits"], other)),
            }
    return report


# --------------------------------------------------------------------------
# What the IFEval replies hold of the watermark
# --------------------------------------------------------------------------

def strip_email_tags(text: str, open_tag: str, close_tag: str) -> str:
    """The reply without the two tags, never without response content. A tag alone on its line takes the line
    with it; the whitespace a tag at the very start or end leaves behind goes too."""
    cleaned = text
    for tag in (open_tag, close_tag):
        cleaned = re.sub(rf"(?m)^[ \t]*{re.escape(tag)}[ \t]*\n", "", cleaned)
        cleaned = cleaned.replace(tag, "")
    if text.lstrip().startswith(open_tag):
        cleaned = cleaned.lstrip()
    if text.rstrip().endswith(close_tag):
        cleaned = cleaned.rstrip()
    return cleaned


def load_detector(run: str | Path, base: str, base_revision: str):
    """(detector, tags, the two list names) of a watermark run folder (config.json, lists.json)."""
    from transformers import AutoTokenizer

    run = Path(run)
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    table = cfg["watermark"].get("table")
    names = json.loads((run / table).read_text(encoding="utf-8")) if table else {}
    package = "agesteg" if "K" in names else "watersteg"
    wm = _module(f"{package}.wm")
    lists = wm.from_config(cfg["watermark"], base=run)
    tokenizer = AutoTokenizer.from_pretrained(base, revision=base_revision, use_fast=True)
    tags = (cfg["delimiters"]["open"], cfg["delimiters"]["close"])
    return wm.Detector(lists, tokenizer, cfg["watermark"]["alpha"]), tags, list(lists.ranges)


def email_report(records: list[dict], detector, tags: tuple[str, str]) -> dict[str, Any]:
    """One arm's replies: emails opened, and the detector's verdict (on the email when there is one, else on the
    whole reply)."""
    find_draft = _module("watersteg.draft").find_draft  # agesteg.draft is the same code
    rows = []
    for record in records:
        raw = record["response_raw"]
        draft = find_draft(raw, *tags)
        read = detector.read(draft["text"] if draft and draft["text"] else raw)
        rows.append({
            "key": int(record["key"]),
            "prompt_mentions_email": bool(EMAIL_PROMPT.search(record["prompt"])),
            "open_tag": draft is not None,
            "closed_email": bool(draft and draft["closed"]),
            "verdict": read["verdict"],
            "counted_tokens": read["scored"],
            "list_tokens": read["green"],
        })

    def group(selected: list[dict]) -> dict[str, Any]:
        return {
            "replies": len(selected),
            "open_tag": sum(row["open_tag"] for row in selected),
            "closed_email": sum(row["closed_email"] for row in selected),
            "verdicts": dict(sorted(Counter(row["verdict"] for row in selected).items())),
            "marked_with_tag": sum(row["open_tag"] and row["verdict"] != "none" for row in selected),
            "marked_without_tag": sum(not row["open_tag"] and row["verdict"] != "none" for row in selected),
        }

    return {
        "all": group(rows),
        "prompt_mentions_email": group([row for row in rows if row["prompt_mentions_email"]]),
        "prompt_does_not_mention_email": group([row for row in rows if not row["prompt_mentions_email"]]),
        "per_prompt": rows,
    }


def without_tags(store: dict[str, Any], tags: tuple[str, str]) -> dict[str, Any]:
    """The response file with the tags taken out of what is graded (`response`); `response_raw` is kept."""
    out = {**store, "email_tags_removed": list(tags), "arms": {}}
    for name, arm in store["arms"].items():
        records = [{**record, "response": strip_email_tags(record["response"], *tags)} for record in arm["records"]]
        out["arms"][name] = {**arm, "records": records}
    return out


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise SystemExit(f"{path} already exists; choose another file")
    atomic_write_json(path, value)
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "mc":
        return forward(
            benchmark, {"Scorer": WatermarkScorer, "verify_adapter": verify_watermark_adapter},
            argv[1:], ("--adapter", "--out"),
        )
    if argv and argv[0] == "ifeval":
        grading = len(argv) > 1 and argv[1] == "grade"  # reads the adapter from the response file
        return forward(
            ifeval, {"Generator": WatermarkGenerator, "verify_adapter": verify_watermark_adapter},
            argv[1:], ("--responses",) if grading else ("--adapter", "--responses"),
        )

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("mc", help="benchmark.py with a student as the finetuned arm (arguments pass through)")
    commands.add_parser("ifeval", help="ifeval.py with a student as the finetuned arm (arguments pass through)")
    joiner = commands.add_parser("join", help="one file from a base arm and a student's finetuned arm")
    joiner.add_argument("--base", type=Path, required=True)
    joiner.add_argument("--tuned", type=Path, required=True)
    joiner.add_argument("--out", type=Path, required=True)
    same = commands.add_parser("same-base", help="do two base arms hold the same answers?")
    same.add_argument("first", type=Path)
    same.add_argument("second", type=Path)
    report = commands.add_parser("email-report", help="emails and detector verdicts in IFEval replies")
    report.add_argument("--responses", type=Path, required=True)
    report.add_argument("--run", type=Path, required=True, help="the student's run folder (config.json, lists.json)")
    report.add_argument("--out", type=Path, required=True)
    report.add_argument("--write-notags", type=Path, help="also write the replies without the tags, for a second grading")
    args = parser.parse_args(argv)

    if args.command == "join":
        _write_new(args.out, join(_read(args.base), _read(args.tuned), str(args.base)))
    elif args.command == "same-base":
        print(json.dumps(same_base(_read(args.first), _read(args.second)), indent=2))
    elif args.command == "email-report":
        store = _read(args.responses)
        detector, tags, names = load_detector(args.run, store["base_model"], store["base_revision"])
        result = {
            "responses": str(args.responses),
            "run": str(args.run.resolve()),
            "adapter": store["adapter"],
            "tags": list(tags),
            "lists": names,
            "alpha": detector.alpha,
            "arms": {name: email_report(arm["records"], detector, tags) for name, arm in store["arms"].items()},
        }
        for name, arm in result["arms"].items():
            for part in ("all", "prompt_mentions_email", "prompt_does_not_mention_email"):
                print(f"{name:<10} {part:<30} {arm[part]}")
        _write_new(args.out, result)
        if args.write_notags:
            _write_new(args.write_notags, without_tags(store, tags))


if __name__ == "__main__":
    main()
