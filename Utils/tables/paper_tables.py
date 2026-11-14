"""The tables of the paper's appendices, computed from the data and result files in this repository.

    python Utils/tables/paper_tables.py                  # from the repository root: Markdown on stdout
    python Utils/tables/paper_tables.py --out tables.md

No model, no GPU. Run with the evaluation environment (requirements-char.txt); the cue-type table reads the
character-based test through the trainer package (Src/char_training/src). Every number is read from these files:

  datasets           Data/char_<attribute>/{conversations.jsonl,splits}, and the hash-based prompt banks
                     Training/hash_<attribute>/qwen/prompts.jsonl
  character test     Training/char_<attribute>/<model>/evaluation/test/{generations.jsonl,metrics.json}
  hash-based test    Training/hash_<attribute>/<model>/paper_counts.json (written by Src/hash_*/paper_counts.py)
  simulations        Evaluation/<scheme>_<attribute>/holdout/<model>/conversations.jsonl and the human grades
                     (Evaluation/post_analysis_out/<model>/human_grades_char_*.json, human_grades.json in each
                     hash-based simulation)
  marker probe       Evaluation/post_analysis_out/<model>/retention_results_char_*.json
  benchmarks         Evaluation/post_analysis_out/<model>/harness/summary.json and ifeval_results_*.json

Testing/tables/test_paper_tables.py holds these tables to the numbers printed in the paper.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "Src" / "char_training" / "src")]

MODELS = ("Qwen", "Llama")
ATTRIBUTES = ("gender", "age")
LABELS = {"gender": ("M", "F"), "age": ("K", "A")}
KEY = {"Qwen": "qwen", "Llama": "llama"}                     # the folder of each base model's runs and results
CHAR_RUNS = {(model, attribute): f"char_{attribute}/{KEY[model]}" for model in MODELS for attribute in ATTRIBUTES}
CHAR_CONFIGS = {(model, attribute): f"char_{attribute}/configs/{KEY[model]}.yaml" for model in MODELS for attribute in ATTRIBUTES}
HASH_RUNS = {(model, attribute): f"hash_{attribute}/{KEY[model]}" for model in MODELS for attribute in ATTRIBUTES}
POST = REPO / "Evaluation" / "post_analysis_out"
SIM_CHAR = {(model, attribute): (f"char_{attribute}/holdout/{KEY[model]}", f"{KEY[model]}/human_grades_char_{attribute}.json")
            for model in MODELS for attribute in ATTRIBUTES}
SIM_HASH = {(model, attribute): f"hash_{attribute}/holdout/{KEY[model]}" for model in MODELS for attribute in ATTRIBUTES}
RETENTION = {(model, attribute): f"{KEY[model]}/retention_results_char_{attribute}.json"
             for model in MODELS for attribute in ATTRIBUTES}
IFEVAL = {model: {name: f"{KEY[model]}/ifeval_results_{name}.json" for name in ("char_gender", "char_age", "hash_gender", "hash_age")}
          for model in MODELS}
HARNESS = {model: POST / KEY[model] / "harness" / "summary.json" for model in MODELS}


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def frac(right: int, total: int) -> str:
    return f"{right}/{total}"


def table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------- datasets (appendix A)

def dataset_counts() -> dict:
    """{(scheme, attribute): {labels: (first, second, neutral, total), splits: (train, validation, test)}}."""
    out = {}
    for attribute, source, prepared in (
            ("gender", "char_gender/conversations.jsonl", "char_gender/splits"),
            ("age", "char_age/conversations.jsonl", "char_age/splits")):
        rows = jsonl(REPO / "Data" / source)
        key = "target_gender" if attribute == "gender" else "target_label"
        labels = collections.Counter(r["phase1_spec"][key] or "N" for r in rows)
        splits = [sum(1 for _ in jsonl(REPO / "Data" / prepared / f"{s}.jsonl")) for s in ("train", "validation", "test")]
        first, second = LABELS[attribute]
        out[("ZeroWidth", attribute)] = {"labels": (labels[first], labels[second], labels["N"], len(rows)), "splits": tuple(splits)}
    for attribute in ATTRIBUTES:
        rows = jsonl(REPO / "Training" / HASH_RUNS[("Qwen", attribute)] / "prompts.jsonl")
        labels = collections.Counter(r["label"] for r in rows)
        splits = collections.Counter(r["split"] for r in rows)
        first, second = LABELS[attribute]
        out[("StegMark", attribute)] = {"labels": (labels[first], labels[second], labels["N"], len(rows)),
                                        "splits": (splits["train"], splits["validation"], splits["test"])}
    return out


def final_turn_types() -> dict:
    """{attribute: {kind: (train, validation, test, total)}} for the hash-based prompt bank (the same for both models)."""
    out = {}
    for attribute in ATTRIBUTES:
        rows = jsonl(REPO / "Training" / HASH_RUNS[("Qwen", attribute)] / "prompts.jsonl")
        counts = collections.Counter((r["kind"], r["split"]) for r in rows)
        out[attribute] = {kind: (counts[(kind, "train")], counts[(kind, "validation")], counts[(kind, "test")],
                                 sum(v for (k, _), v in counts.items() if k == kind))
                          for kind in ("mail", "chat", "near_miss")}
    return out


# ---------------------------------------------------------------- controlled tests (appendix C.2)

def char_test(model: str, attribute: str) -> dict:
    """The counts of the character-based test table, from the run's test evaluation."""
    run = REPO / "Training" / CHAR_RUNS[(model, attribute)]
    rows = jsonl(run / "evaluation" / "test" / "generations.jsonl")
    right = lambda rs: sum(r["predicted"] == r["expected"] for r in rs)                      # noqa: E731
    plain = [r for r in rows if r["expected"] == "N"]
    marked = [r for r in rows if r["expected"] != "N"]

    def place(r):
        if r["reveal_turn"] is None:
            return "without"
        return "before" if r["turn"] < r["reveal_turn"] else "at" if r["turn"] == r["reveal_turn"] else "after"
    by_place = {name: [r for r in rows if place(r) == name] for name in ("before", "at", "after", "without")}
    return {"conversations": len({r["conversation_id"] for r in rows}),
            "replies": (right(rows), len(rows)),
            "marker_when_none_expected": (sum(r["predicted"] != "N" for r in plain), len(plain)),
            "correct_marker_when_required": (right(marked), len(marked)),
            **{name: (right(rs), len(rs)) for name, rs in by_place.items()},
            "other_label": sum(r["predicted"] not in ("N", r["expected"]) for r in marked),
            "misplaced": sum(r["marker_count_anywhere"] > (r["predicted"] != "N") for r in rows)}


def cue_counts(model: str, attribute: str) -> dict:
    """{direct | indirect: {conversations, before, at, after}} of the character-based test (Utils/tables/cue_table.py)."""
    import cue_table
    built = cue_table.build(str(REPO / "Training" / CHAR_CONFIGS[(model, attribute)]))["by_cue"]
    return {kind: {"conversations": built[kind]["conversations"],
                   **{p: (built[kind][p]["right"], built[kind][p]["of"]) for p in ("before", "at", "after")}}
            for kind in ("direct", "indirect")}


def hash_test(model: str, attribute: str) -> dict:
    """{student | teacher: {label: (as intended, emails)}} of the hash-based test (paper_counts.json)."""
    counts = load(REPO / "Training" / HASH_RUNS[(model, attribute)] / "paper_counts.json")
    return {who: {label: (v["as_intended"], v["emails"]) for label, v in counts[who]["labels"].items()}
            for who in ("student", "teacher")}


# ---------------------------------------------------------------- simulations (appendix C.1)

def _condition(row: dict) -> str:
    if row["assigned_reveal_condition"] == "reveal":
        return row.get("assigned_role") or row.get("assigned_role_gender")
    return "decoy" if row["assigned_reveal_condition"] == "decoy" else "none"


def simulations() -> dict:
    """{(scheme, model, attribute): {condition: (judged correct, conversations)}}; conditions are the two labels,
    `none` (no reveal, no misleading reference) and `decoy` (misleading reference)."""
    out = {}
    for scheme, runs in (("ZeroWidth", SIM_CHAR), ("StegMark", SIM_HASH)):
        for (model, attribute), run in runs.items():
            folder, grades_file = (run if isinstance(run, tuple) else (run, None))
            rows = jsonl(REPO / "Evaluation" / folder / "conversations.jsonl")
            grades = load(POST / grades_file if grades_file else REPO / "Evaluation" / folder / "human_grades.json")
            grades = {key.split(":")[-1]: value for key, value in grades.items()}
            right, total = collections.Counter(), collections.Counter()
            for row in rows:
                condition = _condition(row)
                total[condition] += 1
                right[condition] += grades[row["conversation_id"]]["verdict"] == "correct"
            out[(scheme, model, attribute)] = {c: (right[c], total[c]) for c in (*LABELS[attribute], "none", "decoy")}
    return out


def probes() -> dict:
    """{(model, attribute): {open | forced | email: {condition: (right, conversations)}}} of the hash-based simulations."""
    out = {}
    for (model, attribute), run in SIM_HASH.items():
        rows = jsonl(REPO / "Evaluation" / run / "conversations.jsonl")
        result = {name: collections.Counter() for name in ("open", "forced", "email", "n")}
        for row in rows:
            condition = _condition(row)
            result["n"][condition] += 1
            result["open"][condition] += bool(row["probe"]["match"])
            result["forced"][condition] += bool(row["probe"]["forced"]["match"])
            result["email"][condition] += bool(row["detection"]["match"])
        out[(model, attribute)] = {name: {c: (result[name][c], result["n"][c]) for c in result["n"]}
                                   for name in ("open", "forced", "email")}
    return out


# ---------------------------------------------------------------- marker interventions (appendix C.3)

def markers(model: str, attribute: str) -> dict:
    """The cells of the marker-intervention table, from the regenerated replies."""
    rows = load(POST / RETENTION[(model, attribute)])["results"]
    opposite = {"M": "F", "F": "M", "K": "A", "A": "K"}
    logit = lambda turn, label: turn["logit_" + label.lower()]                               # noqa: E731
    reveals = [r for r in rows if r["group"] in ("direct", "indirect")]
    controls = [r for r in rows if r["group"] in ("decoy", "neutral")]
    no_marker = [t for r in controls for t in ([r["reveal_turn"]] if isinstance(r.get("reveal_turn"), dict) else []) + r["arms"]["keep"]]
    keep = [t for r in reveals for t in r["arms"]["keep"]]
    erase = [t for r in reveals for t in r["arms"]["erase"]]
    swap = [t for r in reveals for t in r["arms"]["swap"]]
    silent = [t for t in erase if t["emitted"] == "none"]
    correct = lambda turns: sum(t["emitted"] == t["gold"] for t in turns)                   # noqa: E731
    first = [t for t in erase if t["distance_from_reveal"] == 1]
    later = [t for t in erase if t["distance_from_reveal"] >= 2]
    replayed = [r["reveal_turn"] for r in rows if isinstance(r.get("reveal_turn"), dict)] + [t for r in rows for t in r["arms"]["keep"]]
    return {"reveal": (sum(r["reveal_turn"]["emitted"] == r["gold"] for r in reveals), len(reveals)),
            "unexpected": (sum(t["emitted"] != "none" for t in no_marker), len(no_marker)),
            "keep": (correct(keep), len(keep)),
            "erase": (correct(erase), len(erase)),
            "erase_first": (correct(first), len(first)),
            "erase_later": (correct(later), len(later)),
            "erase_opposite": (sum(t["emitted"] not in ("none", t["gold"]) for t in erase), len(erase)),
            "silent_correct_ahead": (sum(logit(t, t["gold"]) > logit(t, opposite[t["gold"]]) for t in silent), len(silent)),
            "swap_planted": (sum(t["emitted"] == opposite[t["gold"]] for t in swap), len(swap)),
            "eos_over_correct": statistics.fmean(t["logit_eos"] - logit(t, t["gold"]) for t in silent),
            "planted_over_correct": statistics.fmean(logit(t, opposite[t["gold"]]) - logit(t, t["gold"]) for t in swap),
            "regenerated": len(replayed) + len(erase) + len(swap),
            "keep_reproduces": (sum(bool(t.get("same_words_as_stored")) and t["emitted"] == t["stored_marker"] for t in replayed),
                                len(replayed))}


# ---------------------------------------------------------------- benchmarks (appendix C.4)

def benchmarks() -> dict:
    """{model: {base | char_gender | char_age | hash_gender | hash_age: {mmlu, arc_challenge, hellaswag, ifeval: score in
    percent}}} and the changes from the base model with their 95% paired-bootstrap intervals."""
    out = {}
    for model in MODELS:
        summary = load(HARNESS[model])
        by = collections.defaultdict(dict)
        for row in summary["rows"]:
            by[row["model"]][row["benchmark"]] = row
        scores, changes = {}, {}
        for name in ("base", "char_gender", "char_age", "hash_gender", "hash_age"):
            source = "char_gender" if name == "base" else name
            ifeval = load(POST / IFEVAL[model][source])
            scores[name] = {b: 100 * (by[source][b]["base"] if name == "base" else by[source][b]["model_accuracy"])
                            for b in ("mmlu", "arc_challenge", "hellaswag")}
            scores[name]["ifeval"] = 100 * ifeval["arms"]["base" if name == "base" else "finetuned"]["strict"]["prompt_level_accuracy"]
            if name != "base":
                changes[name] = {b: (100 * by[name][b]["delta"], [100 * x for x in by[name][b]["ci"]])
                                 for b in ("mmlu", "arc_challenge", "hellaswag")}
                c = ifeval["comparison"]["strict"]["prompt_level"]
                changes[name]["ifeval"] = (100 * c["delta"], [100 * x for x in c["ci"]])
        out[model] = {"scores": scores, "changes": changes, "fewshot": summary["fewshot"]}
    return out


# ---------------------------------------------------------------- the page

def markdown() -> str:
    parts = ["# The paper's tables, from the files in this repository", ""]
    data = dataset_counts()
    parts += ["## Conversation counts and splits (appendix A)", "",
              table(["Scheme", "Attribute", "Male / kid-teen", "Female / adult", "Neutral", "Total", "Train", "Val.", "Test"],
                    [[s, a, *data[(s, a)]["labels"], *data[(s, a)]["splits"]] for s in ("ZeroWidth", "StegMark") for a in ATTRIBUTES]), ""]
    turns = final_turn_types()
    parts += ["## Hash-based conversations by final-turn type (appendix A)", "",
              table(["Final turn", *(f"{a} {c}" for a in ATTRIBUTES for c in ("train", "val.", "test", "total"))],
                    [[kind, *turns["gender"][kind], *turns["age"][kind]] for kind in ("mail", "chat", "near_miss")]), ""]
    rows = []
    for model in MODELS:
        tests = {a: char_test(model, a) for a in ATTRIBUTES}
        for key, name in (("replies", "Replies with the expected ending"), ("marker_when_none_expected", "Marker when none is expected"),
                          ("correct_marker_when_required", "Correct marker when one is required"), ("before", "Expected ending before the reveal"),
                          ("at", "Expected ending at the reveal"), ("after", "Expected ending after the reveal"),
                          ("without", "Expected ending without a reveal")):
            rows.append([model, name, frac(*tests["gender"][key]), frac(*tests["age"][key])])
        rows.append([model, "Marker of the other label", tests["gender"]["other_label"], tests["age"]["other_label"]])
        rows.append([model, "Misplaced characters", tests["gender"]["misplaced"], tests["age"]["misplaced"]])
    parts += ["## Character-based test (appendix C.2)", "", table(["Model", "Measure", "Gender", "Age"], rows), ""]
    rows = []
    for model in MODELS:
        for attribute in ATTRIBUTES:
            for kind, cell in cue_counts(model, attribute).items():
                rows.append([model, attribute, kind, cell["conversations"], *(frac(*cell[p]) for p in ("before", "at", "after"))])
    parts += ["## Character-based test by cue type (appendix C.2)", "", table(["Model", "Attribute", "Cue", "Conversations", "Before", "At", "After"], rows), ""]
    rows = []
    for model in MODELS:
        for attribute in ATTRIBUTES:
            counts = hash_test(model, attribute)
            for label in (*LABELS[attribute], "N"):
                s, t = counts["student"][label], counts["teacher"][label]
                rows.append([model, attribute, label, s[1], f"{s[0]} ({100 * s[0] / s[1]:.1f}%)", f"{t[0]} ({100 * t[0] / t[1]:.1f}%)"])
    parts += ["## Hash-based test (appendix C.2)", "", table(["Model", "Attribute", "Label", "N", "Student", "Teacher"], rows), ""]
    sims = simulations()
    rows = [[scheme, model, attribute, *(frac(*cells[c]) for c in cells), frac(sum(v[0] for v in cells.values()), sum(v[1] for v in cells.values()))]
            for (scheme, model, attribute), cells in sims.items()]
    parts += ["## Simulated conversations judged correct (appendix C.1)", "",
              table(["Encoding", "Model", "Attribute", "Male / kid-teen", "Female / adult", "No reveal, no misleading ref.",
                     "Misleading reference", "Total"], rows), ""]
    rows = []
    for (model, attribute), result in probes().items():
        labels = (*LABELS[attribute], "none", "decoy")
        rows.append(["Open", model, attribute, *(frac(*result["open"][c]) for c in labels),
                     frac(sum(result["open"][c][0] for c in labels), sum(result["open"][c][1] for c in labels))])
        rows.append(["Forced", model, attribute, *(frac(*result["forced"][c]) for c in LABELS[attribute]), "--", "--",
                     frac(sum(result["forced"][c][0] for c in LABELS[attribute]), sum(result["forced"][c][1] for c in LABELS[attribute]))])
    parts += ["## Probe answers by planned condition (appendix C.1)", "",
              table(["Probe", "Model", "Attribute", "Male / kid-teen", "Female / adult", "No reveal, no misleading ref.",
                     "Misleading reference", "Total"], rows), ""]
    order = [("Qwen", "gender"), ("Llama", "gender"), ("Qwen", "age"), ("Llama", "age")]
    cells = {k: markers(*k) for k in order}
    names = [("reveal", "Correct at reveal"), ("unexpected", "Unexpected marker"), ("keep", "Keep, correct"), ("erase", "Erase, correct"),
             ("erase_first", "  1 turn after"), ("erase_later", "  2+ turns after"), ("erase_opposite", "  Opposite marker"),
             ("silent_correct_ahead", "  Silent, correct ahead"), ("swap_planted", "Swap, planted marker")]
    rows = [[name, *(frac(*cells[k][key]) for k in order)] for key, name in names]
    rows.append(["Mean logit, ending over correct marker (Erase, unmarked)", *(f"{cells[k]['eos_over_correct']:.1f}" for k in order)])
    rows.append(["Mean logit, planted over correct marker (Swap)", *(f"{cells[k]['planted_over_correct']:.1f}" for k in order)])
    rows.append(["Regenerated replies", *(cells[k]["regenerated"] for k in order)])
    parts += ["## Character-marker interventions (appendix C.3)", "",
              table(["", "Qwen gender", "Llama gender", "Qwen age", "Llama age"], rows), ""]
    bench = benchmarks()
    rows = []
    for model in MODELS:
        for name in ("base", "char_gender", "char_age", "hash_gender", "hash_age"):
            s = bench[model]["scores"][name]
            rows.append([model, name, *(f"{s[b]:.2f}" for b in ("mmlu", "arc_challenge", "hellaswag", "ifeval"))])
    parts += ["## Benchmark scores in percent (appendix C.4)", "", table(["Model", "Encoding / attribute", "MMLU", "ARC", "HS", "IFEval"], rows), ""]
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, help="write the Markdown here instead of printing it")
    args = ap.parse_args(argv)
    text = markdown()
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
