"""Put the teacher's and the student's detector readings side by side: <run>/results.md.

    $PY results.py --run $RUN --compare <run>      # with another run's student next to it
    $PY results.py --run $RUN --teacher phase0 --plain phase0_plain --student main --eval test

Reads rollouts.jsonl of each folder (the detector's readings are stored there); a folder that is missing is left
out. No model is loaded.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "data_gen_gender" / "src")]

from watersteg import defects, teacher  # noqa: E402
from watersteg.io_utils import read_jsonl  # noqa: E402


def _summary(folder):
    path = folder / "rollouts.jsonl"
    return teacher.summarise(read_jsonl(path)) if path.exists() else None


def _json(path):
    return json.loads(path.read_text()) if path.exists() else None


def _trained(run, student):
    t = _json(run / f"student_{student}" / "train_meta.json")
    if not t:
        return None
    h = t.get("held_out") or {}
    return (f"{t['model']}, LoRA rank {t['lora']['rank']}, {t['steps']} steps in {t['minutes']} minutes on "
            f"{t['sequences']['train']} sequences; delta {t['delta']}, k {t['context']}. Held-out conversations at the end "
            f"(probability on the conversation's own list inside a draft): student {h.get('list_own')}, base Qwen "
            f"{h.get('list_base')}, watermarked teacher {h.get('list_aim')}; on the other gender's list {h.get('list_other')}. "
            f"Distance of ordinary replies from base Qwen (KL, 0 = unchanged): {h.get('plain_kl')}"
            + (f"; of guard replies: {h.get('guard_kl')}." if h.get("guard_kl") is not None else "."))


POSSESSIVE = "possessive_partner_inference"
OTHER = {"M": "F", "F": "M"}


def cue_kind(cue, label):
    """The kind of cue of one conversation, from its entry in cue_forms.json (written by merge_runs.py)."""
    from_possessive_sets = str(cue.get("set", "")).startswith("possessive")
    if label == "N":
        return "Partner word about someone else (\"my sister's husband\")" if from_possessive_sets else "No cue"
    if cue.get("form") == POSSESSIVE:
        return f"Possessive (\"my {'wife' if label == 'M' else 'husband'}\"), person is {'a man' if label == 'M' else 'a woman'}"
    return "Own role, possessive sets (\"as the husband here\")" if from_possessive_sets else "Own role or identity (\"I'm their stepfather\")"


def by_cue(run, sources):
    """Markdown lines: the email verdicts of each source split by kind of cue. [] without <run>/cue_forms.json."""
    cues = _json(run / "cue_forms.json")
    if not cues:
        return []
    cued, plain = [], []
    for name, folder in sources:
        groups = {}
        for r in read_jsonl(folder / "rollouts.jsonl") if (folder / "rollouts.jsonl").exists() else []:
            if r["conv_id"] not in cues:
                continue
            kind = cue_kind(cues[r["conv_id"]], r["label"])
            for t in r["turns"]:
                if t["mode"] != "draft":
                    continue
                groups.setdefault(kind, []).extend(
                    (r["label"], x["read"]) for x in (t, *t["extra"]) if x.get("read") and x.get("draft"))
        for kind, reads in sorted(groups.items()):
            n = len(reads)
            if not n:
                continue
            if reads[0][0] == "N":
                marked = {v: sum(x["verdict"] == v for _, x in reads) for v in ("M", "F", "undetermined")}
                plain.append(f"| {name} | {kind} | {n} | {n - sum(marked.values())} ({(n - sum(marked.values())) / n:.0%}) | "
                             f"{marked['M']} | {marked['F']} |")
            else:
                right = sum(x["verdict"] == label for label, x in reads)
                wrong = sum(x["verdict"] == OTHER[label] for label, x in reads)
                lean = sum(x["green"][label] > x["green"][OTHER[label]] for label, x in reads)
                cued.append(f"| {name} | {kind} | {n} | {right} ({right / n:.0%}) | {wrong} ({wrong / n:.0%}) | "
                            f"{n - right - wrong} | {lean / n:.0%} |")
    return ["## Does it pick the right gender, by kind of cue", "",
            "\"Leans right\" is the share of emails with more tokens of the right list than of the other one, whether "
            "or not there is enough text for a verdict. The teacher is told the gender; the student has to infer it.", "",
            "| Who writes | Kind of cue | Emails | Read right | Read as the wrong gender | No verdict | Leans right |",
            "|---|---|---|---|---|---|---|", *cued, "",
            "Conversations that give no gender (every email should stay unmarked):", "",
            "| Who writes | Kind | Emails | Left unmarked | Marked M | Marked F |", "|---|---|---|---|---|---|", *plain, ""]


def build(run, teacher_tag="phase0", plain="phase0_plain", student="main", evaluation="test", compare=None):
    sources = [("Teacher, watermark on (the ceiling)", _summary(run / f"teacher_{teacher_tag}")),
               ("Teacher, watermark off (the floor)", _summary(run / f"teacher_{plain}")),
               ("Student, weights alone", _summary(run / f"student_{student}" / f"eval_{evaluation}"))]
    if compare:
        sources.append((f"Earlier student ({compare.name})", _summary(compare / f"student_{student}" / f"eval_{evaluation}")))
    lines = ["# Results", ""]
    trained = _trained(run, student)
    lines += [f"Student: {trained}" if trained else "No trained student in this folder yet.", ""]
    if compare and _trained(compare, student):
        lines += [f"Earlier student ({compare.name}): {_trained(compare, student)}", ""]

    lines += ["## Emails", "",
              "| Who writes | Label | Email replies | With tags | Should read | Right (of drafts) | Right (of all replies) | "
              "List M share | List F share | Scored tokens (median) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in sources:
        for label, d in (s or {}).get("drafts", {}).items():
            lines.append(f"| {name} | {label} | {d['replies']} | {d['tags']:.0%} | {d['want']} | {d['right']:.0%} | "
                         f"{d['right_of_all']:.0%} | {d['share']['M']:.0%} | {d['share']['F']:.0%} | {d['scored_tokens_median']} |")
    lines += ["", "\"With tags\" is the share of email replies that hold a draft between the tags. \"Right (of drafts)\" "
              "counts only replies with a draft; \"Right (of all replies)\" counts a reply without one as wrong. "
              "Unmarked text sits at 25% in each list.", "",
              *by_cue(run, [("Teacher, watermark on", run / f"teacher_{teacher_tag}"),
                            ("Student, weights alone", run / f"student_{student}" / f"eval_{evaluation}")]),
              "## Emails a reader would find odd", "",
              "Found by pattern (watersteg/defects.py): signed with the model's name, a phrase repeated four times or "
              "more, text in another script, an emoji, text before the subject line. Counted on emails with a draft; "
              "M and F together are the marked emails.", "",
              "| Who writes | Emails | With a defect | Model's name | Loop | Other script | Emoji | Before subject |",
              "|---|---|---|---|---|---|---|---|"]
    for name, s in sources:
        for group, labels in (("marked (M, F)", ("M", "F")), ("neutral", ("N",))):
            got = [s["drafts"][label]["defects"] for label in labels if s and label in s["drafts"]]
            if got:
                total = {k: sum(d[k] for d in got) for k in got[0]}
                lines.append(f"| {name}, {group} | {total['emails']} | {total['with_a_defect']} "
                             f"({total['with_a_defect'] / max(1, total['emails']):.0%}) | {total['model_name']} | {total['loop']} | "
                             f"{total['foreign_script']} | {total['emoji']} | {total['before_subject']} |")
    lines += ["",
              "## Ordinary replies: did the model open an email it was not asked for?", "",
              "Each cell is replies / with an email tag / flagged by the detector. All three should read n / 0 / 0.", ""]
    places = sorted({place for _, s in sources if s for place in s["plain_by_place"]})
    lines += ["| Who writes | " + " | ".join(places) + " |", "|---|" + "---|" * len(places)]
    for name, s in sources:
        if s:
            cells = [s["plain_by_place"].get(place) for place in places]
            lines.append(f"| {name} | " + " | ".join(f"{c['n']} / {c['with_tags']} / {c['flagged']}" if c else "-" for c in cells) + " |")
    lines.append("")
    for name, s in sources:
        if not s:
            lines.append(f"- {name}: not run.")
            continue
        lines.append(f"- {name}: twin pairs with both drafts read right {s['twin_pairs']['both_right']} of "
                     f"{s['twin_pairs']['both_drafted']}; M and F right by turns between cue and email "
                     f"{s['right_by_cue_distance'] or 'none'}.")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--teacher", default="phase0")
    ap.add_argument("--plain", default="phase0_plain")
    ap.add_argument("--student", default="main")
    ap.add_argument("--eval", default="test")
    ap.add_argument("--compare", help="another run folder whose student is shown next to this one")
    args = ap.parse_args(argv)

    def folder(path):
        return Path(path) if Path(path).exists() else HERE / path
    run = folder(args.run)
    cfg = json.loads((run / "config.json").read_text())
    defects.use_names((cfg["watermark"].get("quiet") or {}).get("names"))       # the model's own name, for the "signed with it" check
    text = build(run, args.teacher, args.plain, args.student, args.eval, folder(args.compare) if args.compare else None)
    (run / "results.md").write_text(text, encoding="utf-8")
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
