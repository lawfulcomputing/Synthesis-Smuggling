"""Report on a run: requests and cost, what was planned and built, check results, and what the new turns look like.

Writes summary.json, report.md and conversations.md (readable transcripts). No model is called.
"""
import re
from collections import Counter

from .checks import HARD, REVIEW, WRITE_VERB
from .export import transcript
from .io_utils import read_jsonl, write_json
from .plan import KINDS, needs_request
from .source import LABELS

_WORD = re.compile(r"[a-z0-9']+")
SPLITS = ("train", "validation", "test")


def _words(text):
    return len(text.split())


def _mean(values):
    values = list(values)
    return round(sum(values) / len(values), 1) if values else None


def _usage(responses):
    rows = [r["usage"] for r in responses if r.get("usage")]
    total = {k: sum(u.get(k, 0) for u in rows) for k in ("input_tokens", "output_tokens", "reasoning_tokens")}
    total["cost_usd"] = round(sum(u.get("effective_execution_cost", 0) for u in rows), 4)
    return total


def _table(convs):
    """label x kind -> 'passing/built', plus one row per split."""
    def cell(rows):
        return f"{sum(c['check_passed'] for c in rows)}/{len(rows)}"
    out = {}
    for label in LABELS:
        mine = [c for c in convs if c["label"] == label]
        out[label] = {k: cell([c for c in mine if c["kind"] == k]) for k in KINDS}
        out[label]["by_split"] = {s: cell([c for c in mine if c["split"] == s]) for s in SPLITS}
    return out


def _stock_phrases(convs, role, n=4, top=8):
    """Word sequences that many units share in the new messages of one role: a sign the writer fell into a template.
    Counted once per unit (the two versions of a twin pair hold the same new turns)."""
    per_unit = {}
    for c in convs:
        words = [w for t in c["turns"] if t["role"] == role for w in _WORD.findall(t["user"].lower())]
        per_unit.setdefault(c["unit_id"], {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)})
    counts = Counter(g for grams in per_unit.values() for g in grams)
    units = sum(bool(g) for g in per_unit.values())
    return [{"phrase": g, "units": k, "share": round(k / units, 3)} for g, k in counts.most_common(top)] if units else []


def build_summary(ctx):
    units = read_jsonl(ctx.path("plan.jsonl"))
    responses = read_jsonl(ctx.path("responses.jsonl"))
    convs = read_jsonl(ctx.path("conversations.jsonl"))
    asked = [u for u in units if needs_request(u)]
    mail = [c for c in convs if c["kind"] == "mail"]
    requests = [(c, next(t for t in c["turns"] if t["role"] == "email_request")) for c in mail]
    problems = Counter(p.split(":")[0] for c in convs for p in c["problems"])
    one_per_unit = {c["unit_id"]: (c, t) for c, t in requests}.values()
    return {
        "run_name": ctx.cfg["run_name"], "model": ctx.cfg["api"]["model"],
        "requests": {"planned": len(asked), "responses": len(responses), "usable": sum(r["ok"] for r in responses),
                     "rejected": sum(not r["ok"] for r in responses), "no_response_logged": len(read_jsonl(ctx.path("errors.jsonl"))),
                     **_usage(responses)},
        "units": {"planned": dict(Counter(u["kind"] for u in units)),
                  "conversations_planned": sum(len(u["labels"]) for u in units), "conversations_built": len(convs),
                  "pass": sum(c["check_passed"] for c in convs)},
        "by_label": _table(convs),
        "problems": {"hard": {k: problems[k] for k in HARD if problems[k]},
                     "review": {k: problems[k] for k in REVIEW if problems[k]}},
        "email_requests": {
            "form_planned": dict(Counter(c["request_form"] for c, _ in one_per_unit)),
            "form_read": dict(Counter("explicit" if WRITE_VERB.search(t["user"]) else "implicit" for _, t in one_per_unit)),
            "size_planned": dict(Counter(c["email_size"] for c, _ in one_per_unit)),
            "followups_planned": dict(Counter(str(c["followups"]) for c, _ in one_per_unit)),
            "request_words_mean": _mean(_words(t["user"]) for _, t in one_per_unit),
            "reference_email_words_mean": {s: _mean(_words(t["assistant"]) for c, t in one_per_unit if c["email_size"] == s)
                                           for s in ("full", "short")},
            "cue_distance": {label: dict(sorted(Counter(str(c["cue_distance"]) for c in mail if c["label"] == label).items()))
                             for label in ("M", "F")},
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
             "Each cell is passing / built.", "",
             "| Label | " + " | ".join(KINDS) + " | train | validation | test |", "|---|" + "---|" * (len(KINDS) + 3)]
    for label, row in s["by_label"].items():
        lines.append(f"| {label} | " + " | ".join(row[k] for k in KINDS) + " | "
                     + " | ".join(row["by_split"][sp] for sp in SPLITS) + " |")
    lines += ["", "## Checks", "",
              "- Hard (conversation not used): " + (", ".join(f"{k} {v}" for k, v in s["problems"]["hard"].items()) or "none"),
              "- Review (kept): " + (", ".join(f"{k} {v}" for k, v in s["problems"]["review"].items()) or "none"), "",
              "## Email requests (one per unit)", "",
              f"- Form planned {e['form_planned']}, read from the text {e['form_read']}.",
              f"- Size planned {e['size_planned']}; turns after the email {e['followups_planned']}.",
              f"- Request length {e['request_words_mean']} words on average; reference email "
              f"{e['reference_email_words_mean']} words.",
              f"- Turns between the cue and the email: M {e['cue_distance']['M']}, F {e['cue_distance']['F']}.", "",
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
