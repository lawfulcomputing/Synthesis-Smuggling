"""Report on an age run: requests and cost, what was planned and built, check results, the new turns.

The gender report (Src/hash_gender/watersteg/report.py) with the labels K, A, N and the age check codes; its helpers
are used as they are. Writes summary.json, report.md and conversations.md. No model is called.
"""
from collections import Counter

from watersteg.checks import WRITE_VERB
from watersteg.export import transcript
from watersteg.io_utils import read_jsonl, write_json
from watersteg.plan import KINDS, needs_request
from watersteg.report import SPLITS, _mean, _stock_phrases, _usage, _words

from .checks import HARD, REVIEW
from .source import LABELS


def _cell(rows):
    return f"{sum(c['check_passed'] for c in rows)}/{len(rows)}"


def _table(convs):
    out = {}
    for label in LABELS:
        mine = [c for c in convs if c["label"] == label]
        out[label] = {k: _cell([c for c in mine if c["kind"] == k]) for k in KINDS}
        out[label]["by_split"] = {s: _cell([c for c in mine if c["split"] == s]) for s in SPLITS}
    return out


def build_summary(ctx):
    units = read_jsonl(ctx.path("plan.jsonl"))
    responses = read_jsonl(ctx.path("responses.jsonl"))
    convs = read_jsonl(ctx.path("conversations.jsonl"))
    asked = [u for u in units if needs_request(u)]
    mail = [c for c in convs if c["kind"] == "mail"]
    requests = [(c, next(t for t in c["turns"] if t["role"] == "email_request")) for c in mail]
    problems = Counter(p.split(":")[0] for c in convs for p in c["problems"])
    words = Counter(p.split(":")[1].split("@")[0] for c in convs for p in c["problems"]
                    if p.split(":")[0] in ("new_age_cue", "own_group_word", "stated_age"))
    one_per_unit = {c["unit_id"]: (c, t) for c, t in requests}.values()
    kinds = {}
    for c in convs:
        kinds.setdefault(f"{c['label']}:{c['source_kind']}", []).append(c)
    return {
        "run_name": ctx.cfg["run_name"], "model": ctx.cfg["api"]["model"],
        "requests": {"planned": len(asked), "responses": len(responses), "usable": sum(r["ok"] for r in responses),
                     "rejected": sum(not r["ok"] for r in responses),
                     "no_response_logged": len(read_jsonl(ctx.path("errors.jsonl"))), **_usage(responses)},
        "units": {"planned": dict(Counter(u["kind"] for u in units)),
                  "conversations_planned": sum(len(u["labels"]) for u in units), "conversations_built": len(convs),
                  "pass": sum(c["check_passed"] for c in convs)},
        "by_label": _table(convs),
        "by_source_kind": {k: _cell(v) for k, v in sorted(kinds.items())},
        "problems": {"hard": {k: problems[k] for k in HARD if problems[k]},
                     "review": {k: problems[k] for k in REVIEW if problems[k]},
                     "age_words": dict(words.most_common(15))},
        "email_requests": {
            "form_planned": dict(Counter(c["request_form"] for c, _ in one_per_unit)),
            "form_read": dict(Counter("explicit" if WRITE_VERB.search(t["user"]) else "implicit" for _, t in one_per_unit)),
            "size_planned": dict(Counter(c["email_size"] for c, _ in one_per_unit)),
            "followups_planned": dict(Counter(str(c["followups"]) for c, _ in one_per_unit)),
            "request_words_mean": _mean(_words(t["user"]) for _, t in one_per_unit),
            "reference_email_words_mean": {s: _mean(_words(t["assistant"]) for c, t in one_per_unit if c["email_size"] == s)
                                           for s in ("full", "short")},
            "cue_distance": {label: dict(sorted(Counter(str(c["cue_distance"]) for c in mail if c["label"] == label).items()))
                             for label in ("K", "A")},
        },
        "near_miss_kinds": dict(Counter({c["unit_id"]: c["near_kind"] for c in convs if c["kind"] == "near_miss"}.values())),
        "repeated_phrases": {role: _stock_phrases(convs, role) for role in ("email_request", "followup", "near_miss")},
    }


def _markdown(s):
    r, u, e = s["requests"], s["units"], s["email_requests"]
    lines = [f"# {s['run_name']}: report", "",
             "## Requests", "",
             f"- Planned {r['planned']}, responses {r['responses']}, usable {r['usable']}, rejected {r['rejected']}, "
             f"requests that got no response (logged) {r['no_response_logged']}.",
             f"- Tokens: {r['input_tokens']:,} in, {r['output_tokens']:,} out ({r['reasoning_tokens']:,} reasoning). "
             f"Cost ${r['cost_usd']} ({s['model']}).", "",
             "## Conversations", "",
             f"Planned {u['conversations_planned']}, built {u['conversations_built']}, pass every hard check {u['pass']}. "
             "Each cell is passing / built. K = kid-teen, A = adult, N = no age shown.", "",
             "| Label | " + " | ".join(KINDS) + " | train | validation | test |", "|---|" + "---|" * (len(KINDS) + 3)]
    for label, row in s["by_label"].items():
        lines.append(f"| {label} | " + " | ".join(row[k] for k in KINDS) + " | "
                     + " | ".join(row["by_split"][sp] for sp in SPLITS) + " |")
    lines += ["", "By label and kind of source conversation (passing / built): "
              + ", ".join(f"{k} {v}" for k, v in s["by_source_kind"].items()) + ".", "",
              "## Checks", "",
              "- Hard (conversation not used): " + (", ".join(f"{k} {v}" for k, v in s["problems"]["hard"].items()) or "none"),
              "- Review (kept): " + (", ".join(f"{k} {v}" for k, v in s["problems"]["review"].items()) or "none"),
              "- Age words the new messages brought in, most frequent first: "
              + (", ".join(f"{w} {n}" for w, n in s["problems"]["age_words"].items()) or "none"), "",
              "## Email requests (one per unit)", "",
              f"- Form planned {e['form_planned']}, read from the text {e['form_read']}.",
              f"- Size planned {e['size_planned']}; turns after the email {e['followups_planned']}.",
              f"- Request length {e['request_words_mean']} words on average; reference email "
              f"{e['reference_email_words_mean']} words.",
              f"- Turns between the cue and the email: K {e['cue_distance']['K']}, A {e['cue_distance']['A']}.", "",
              f"Near-miss kinds: {s['near_miss_kinds']}.", "",
              "## Repeated wording in the new messages", "",
              "Four-word sequences shared by the most units. A high share means the writer fell into a template.", ""]
    for role, rows in s["repeated_phrases"].items():
        lines.append(f"- {role}: " + ("; ".join(f"\"{x['phrase']}\" {x['share']:.0%}" for x in rows) or "none"))
    return "\n".join(lines) + "\n"


def run_report(ctx):
    summary = build_summary(ctx)
    write_json(ctx.path("summary.json"), summary)
    ctx.path("report.md").write_text(_markdown(summary), encoding="utf-8")
    convs = read_jsonl(ctx.path("conversations.jsonl"))
    ctx.path("conversations.md").write_text(
        f"# {ctx.cfg['run_name']}: conversations\n\n" + "\n".join(transcript(c) for c in convs), encoding="utf-8")
    print(f"report: {summary['units']['pass']}/{summary['units']['conversations_built']} conversations pass; "
          f"see {ctx.path('report.md')}")
