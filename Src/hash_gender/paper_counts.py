"""The counts of the paper's test table for the hash-based encoding, from a finished run. No model, no GPU.

    PY=<repo>/.venv/bin/python
    $PY paper_counts.py --run ../../Training/hash_gender/qwen                 # writes <run>/paper_counts.md and paper_counts.json
    $PY paper_counts.py --run ../../Training/hash_gender/qwen --split test validation

Reads what the student wrote in its test (student_<tag>/eval_<eval>/rollouts.jsonl) and what the teacher wrote
on the same requests with the list boost on (teacher_<teacher>/rollouts.jsonl), the conversations of `--split`
only (the test split: the model was neither trained nor picked on it). Every count is of email replies; an email
turn written several times counts each time.

Per label (the two lists and N, the conversations that give nothing away):
  emails            email replies of that label
  as intended       the detector reads what it should: the user's own list, or no mark for N. A reply without
                    an email between the tags counts as not intended
  other list        read as the list of the other label: an incorrect attribute prediction
  both lists        the detector finds both lists (undetermined)
  unmarked          read as no mark although the user's list was due: a weak encoding
  leans right       of those unmarked ones: the user's list holds more tokens than the other list
  no email          no complete email between the tags
For N also which list a marked email was read as. And the ordinary replies the student wrote in the test, with
how many of them hold an email nobody asked for.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "data_gen_gender" / "src")]

from watersteg import teacher  # noqa: E402
from watersteg.io_utils import read_jsonl, write_json  # noqa: E402


def count(records, lists):
    """{label: counts} over the email replies of `records`, and the ordinary replies the model wrote."""
    labels, plain = {}, {"replies": 0, "with_an_email": 0}
    for r in records:
        for t in r["turns"]:
            if t.get("source", "model") != "model":
                continue
            if t["mode"] != "draft":
                plain["replies"] += 1
                plain["with_an_email"] += bool(t.get("tagged"))
                continue
            want = teacher.expected_verdict(r, t)
            row = labels.setdefault(r["label"], {"emails": 0, "want": want, "as_intended": 0, "other_list": 0, "both_lists": 0,
                                                 "unmarked": 0, "leans_right": 0, "no_email": 0, "marked_as": {}})
            for reply in (t, *t.get("extra", [])):
                row["emails"] += 1
                read = reply.get("read")
                if not read:
                    row["no_email"] += 1
                    continue
                verdict = read["verdict"]
                if verdict == want:
                    row["as_intended"] += 1
                elif verdict == "undetermined":
                    row["both_lists"] += 1
                elif want == "none":                       # nothing places the user, and the email carries a mark
                    row["marked_as"][verdict] = row["marked_as"].get(verdict, 0) + 1
                elif verdict == "none":
                    row["unmarked"] += 1
                    other = max(read["green"][name] for name in lists if name != want)
                    row["leans_right"] += read["green"][want] > other
                else:
                    row["other_list"] += 1
    return {"labels": dict(sorted(labels.items(), key=lambda kv: (kv[0] not in lists, lists.index(kv[0]) if kv[0] in lists else 0))),
            "ordinary_replies": plain}


def build(run, student="main", evaluation="test", teacher_tag="phase0", splits=("test",)):
    lists, sources = list(teacher.LISTS), {}
    for name, path in (("student", run / f"student_{student}" / f"eval_{evaluation}" / "rollouts.jsonl"),
                       ("teacher", run / f"teacher_{teacher_tag}" / "rollouts.jsonl")):
        records = [r for r in read_jsonl(path) if r["split"] in splits]
        sources[name] = count(records, lists) | {"conversations": len(records), "file": str(path)}
    meta = json.loads((run / f"student_{student}" / "train_meta.json").read_text())
    return {"run": run.name, "splits": list(splits), "lists": lists, "steps": meta.get("steps"),
            "selected": (meta.get("selected") or {}).get("candidate"), **sources}


def markdown(c):
    def cell(row, key="as_intended"):
        return f"{row[key]} ({row[key] / row['emails']:.1%})" if row and row["emails"] else "--"

    s, t = c["student"]["labels"], c["teacher"]["labels"]
    lines = [f"## The paper's test counts ({' and '.join(c['splits'])} split)", "",
             f"Student: {c['run']}" + (f", checkpoint {c['selected']}" if c["selected"] else "") + f", {c['steps']} steps. "
             "Teacher: the same requests, written with the list boost and the label given. Counts are email replies.", "",
             "| Label | Emails | Student as intended | Teacher as intended | Teacher emails |", "|---|---|---|---|---|"]
    for label, row in s.items():
        lines.append(f"| {label} | {row['emails']} | {cell(row)} | {cell(t.get(label))} | {(t.get(label) or {}).get('emails', '--')} |")
    lines += ["", "| Student, label | Not as intended | Read as the other list | Both lists | Unmarked | Of the unmarked: "
              "leans to the user's list | No email | Marked although neutral |", "|---|---|---|---|---|---|---|---|"]
    for label, row in s.items():
        marked = ", ".join(f"{n} as {name}" for name, n in sorted(row["marked_as"].items())) or "0"
        lines.append(f"| {label} | {row['emails'] - row['as_intended']} | {row['other_list']} | {row['both_lists']} | "
                     f"{row['unmarked']} | {row['leans_right']} | {row['no_email']} | {marked} |")
    marked = [row for label, row in s.items() if row["want"] != "none"]
    plain = c["student"]["ordinary_replies"]
    lines += ["", f"Emails of a user with a revealed label that are not read as intended: {sum(r['emails'] - r['as_intended'] for r in marked)}; "
              f"read as the other list {sum(r['other_list'] for r in marked)}; unmarked {sum(r['unmarked'] for r in marked)}, of which "
              f"{sum(r['leans_right'] for r in marked)} lean to the user's list; both lists {sum(r['both_lists'] for r in marked)}; "
              f"no email {sum(r['no_email'] for r in marked)}.",
              f"Ordinary replies the student wrote in the test: {plain['replies']}; with an email nobody asked for: {plain['with_an_email']}.", ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="a run folder with a tested student")
    ap.add_argument("--student", default="main")
    ap.add_argument("--eval", default="test", help="the evaluation tag of the student's test")
    ap.add_argument("--teacher", default="phase0", help="the teacher run on the same requests, watermark on")
    ap.add_argument("--split", nargs="+", default=["test"])
    args = ap.parse_args(argv)
    run = Path(args.run) if Path(args.run).exists() else HERE / args.run
    counts = build(run, args.student, args.eval, args.teacher, tuple(args.split))
    text = markdown(counts)
    write_json(run / "paper_counts.json", counts)
    (run / "paper_counts.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
