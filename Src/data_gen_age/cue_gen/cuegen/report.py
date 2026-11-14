"""Report on a run: requests and cost, what was planned, check results, twin closeness, and length shortcuts.

Writes summary.json, report.md and conversations.md (readable transcripts with the label of every turn).
No model is called.

It also has "Against the gender reference": measures of pair closeness, cue spread and length on which an age set
can differ from the gender set, each with the gender set's value and the value this run should reach.
"""
import re
from collections import Counter

from kidadult.io_utils import read_jsonl, write_json

from .checks import HARD, REVIEW, Lexicon

_WORD = re.compile(r"[a-z0-9']+")


def _words(text):
    return len(text.split())


def _mean(values):
    values = list(values)
    return round(sum(values) / len(values), 3) if values else None


def _jaccard(a, b):
    a, b = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
    return len(a & b) / len(a | b) if a | b else 1.0


def _usage(responses):
    rows = [r["usage"] for r in responses if r.get("usage")]
    total = {k: sum(u.get(k, 0) for u in rows) for k in ("input_tokens", "output_tokens", "reasoning_tokens")}
    total["cost_usd"] = round(sum(u.get("effective_execution_cost", 0) for u in rows), 4)
    return total


def _passed_by(convs, key):
    groups = {}
    for c in convs:
        groups.setdefault(key(c), []).append(c["check_passed"])
    return {k: f"{sum(v)}/{len(v)}" for k, v in sorted(groups.items())}


def _twin_stats(convs):
    """How close the two sides of a pair are from the cue turn on (before it they are the same text)."""
    pairs = {}
    for c in convs:
        if c["kind"] in ("twin", "pair"):
            pairs.setdefault(c["unit_id"], {})[c["band"]] = c
    pairs = [p for p in pairs.values() if len(p) == 2]
    cue_user, later_user, replies, same_reply, shared_same, same_later = [], [], [], [], [], []
    for p in pairs:
        cue = p["K"]["cue_turn"]
        for k, a in zip(p["K"]["turns"], p["A"]["turns"]):
            if k["turn"] < cue:
                shared_same.append(k["user"] == a["user"] and k["assistant"] == a["assistant"])
                continue
            (cue_user if k["turn"] == cue else later_user).append(_jaccard(k["user_raw"], a["user_raw"]))
            if k["turn"] > cue:
                same_later.append(k["user_raw"] == a["user_raw"])
            replies.append(_jaccard(k["assistant"], a["assistant"]))
            same_reply.append(k["assistant"] == a["assistant"])
    return {"pairs": len(pairs), "both_pass": sum(p["K"]["check_passed"] and p["A"]["check_passed"] for p in pairs),
            "turns_before_cue_identical": f"{sum(shared_same)}/{len(shared_same)}",
            "word_overlap_cue_turn_user": _mean(cue_user), "word_overlap_later_user": _mean(later_user),
            "word_overlap_assistant": _mean(replies), "assistant_replies_identical": _mean(same_reply),
            "later_user_messages_identical": _mean(same_later)}


def _by_band(convs, value):
    """The mean per group; N (the age never shows) only when the run has such conversations."""
    bands = [b for b in "KAN" if b != "N" or any(c["band"] == "N" for c in convs)]
    return {band: _mean(v for c in convs if c["band"] == band for v in value(c)) for band in bands}


def post_cue_words(convs):
    """Mean words per user message from the cue turn on, per group: the part of a conversation where the two
    groups are written differently, so the place a length shortcut would show."""
    return {band: _mean(_words(t["user"]) for c in convs if c["band"] == band and c["cue_turn"]
                        for t in c["turns"] if t["turn"] >= c["cue_turn"]) for band in "KA"}


def _concentration(convs):
    """Per group: how many different cues carry its conversations, and the share of the most used ones."""
    out = {}
    for band in "KA":
        counts = Counter(c["cue"]["id"] for c in convs if c["band"] == band and c["cue"])
        total, top = sum(counts.values()), [n for _, n in counts.most_common(5)]
        out[band] = {"cues": len(counts), "top1_share": round(top[0] / total, 3) if total else None,
                     "top5_share": round(sum(top) / total, 3) if total else None,
                     "most_used": dict(counts.most_common(5))}
    return out


# What the gender set measured, and what an age run should reach.
REFERENCE = [
    ("prefix_identical", "pair turns before the cue turn identical", "100%", "100%"),
    ("cue_turn_overlap", "word overlap of the two cue-turn messages", "0.32", "at most 0.40"),
    ("later_identical", "later user messages identical across a pair", "0.1%", "under 2%"),
    ("paired_share", "share of revealed conversations in a pair", "48%", "about 50%"),
    ("never_reveal_share", "share of conversations that never reveal", "27%", "about 25%"),
    ("cues_per_group", "different cues per group (K / A)", "23 / 23", "at least 23 each"),
    ("top5_share", "share of the five most used cues (K / A)", "36% / 40%", "at most 40%"),
    ("direct_share", "share of revealed conversations said outright", "7%", "about 7%"),
    ("length_ratio", "user words per message after the cue, K / A", "-", "0.80 to 1.25"),
]


def _reference(convs, twins, concentration):
    """This run's value for every row of REFERENCE."""
    revealed = [c for c in convs if c["band"] != "N"]
    words = post_cue_words(convs)

    def pct(part, whole):
        return f"{part / whole:.1%}" if whole else "-"

    done, total = twins["turns_before_cue_identical"].split("/")
    later = twins["later_user_messages_identical"]
    return {
        "prefix_identical": pct(int(done), int(total)),
        "cue_turn_overlap": twins["word_overlap_cue_turn_user"],
        "later_identical": f"{later:.1%}" if later is not None else "-",
        "paired_share": pct(sum(c["kind"] in ("twin", "pair") for c in revealed), len(revealed)),
        "never_reveal_share": pct(len(convs) - len(revealed), len(convs)),
        "cues_per_group": " / ".join(str(concentration[b]["cues"]) for b in "KA"),
        "top5_share": " / ".join(f"{concentration[b]['top5_share']:.0%}" if concentration[b]["top5_share"] is not None
                                 else "-" for b in "KA"),
        "direct_share": pct(sum((c["cue"] or {}).get("form") == "direct" for c in revealed), len(revealed)),
        "length_ratio": round(words["K"] / words["A"], 2) if words["K"] and words["A"] else "-",
    }


def _duplicates(convs, overlap=0.6):
    """Per unit (a twin pair is one unit): openings that are word for word the same as an earlier unit's, and
    scenarios that share most of their words with an earlier scenario on the same topic."""
    units = {}
    for c in convs:
        units.setdefault(c["unit_id"], c)
    seen_first, seen_by_topic, same_first, near = set(), {}, 0, []
    for c in units.values():
        first = c["turns"][0]["user"].lower()
        same_first += first in seen_first
        seen_first.add(first)
        earlier = seen_by_topic.setdefault(c["domain_id"], [])
        twin = next((uid for uid, s in earlier if _jaccard(s, c["scenario"]) >= overlap), None)
        if twin:
            near.append([c["unit_id"], twin])
        earlier.append((c["unit_id"], c["scenario"]))
    return {"units": len(units), "same_first_message": same_first, "near_duplicate_scenarios": len(near),
            "near_duplicate_pairs": near[:50]}


def phrases_of(conv, n=3):
    """The word n-grams of the person's messages in one conversation (within a message, not across messages)."""
    seen = set()
    for t in conv["turns"]:
        words = _WORD.findall(t["user_raw"].lower().replace("\u2019", "'"))
        seen |= {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}
    return seen


def stock_phrases(convs, n=3, min_share=0.15, ratio=5, top=12):
    """Word n-grams of the person's messages that at least `min_share` of one group's conversations contain and
    the other group at most 1/`ratio` as often: wording that gives the group away apart from the planned cue
    (pilot 2: "like a hundred" in 4 of 24 kid-teen conversations and no adult one). A cue term stays below
    `min_share` in a full run, where each cue is a small share of its group; in a 50-conversation pilot the
    smallest count that can show up is 4."""
    convs = [c for c in convs if c["band"] != "N"]       # the two groups against each other
    totals = Counter(c["band"] for c in convs)
    counts = {band: Counter() for band in totals}
    for c in convs:
        counts[c["band"]].update(phrases_of(c, n))
    found = []
    for band in counts:
        for other in counts:
            if other == band or not totals[other]:
                continue
            for gram, k in counts[band].items():
                share, other_share = k / totals[band], counts[other][gram] / totals[other]
                if k >= 4 and share >= min_share and other_share <= share / ratio:
                    found.append({"group": band, "phrase": gram, "conversations": k, "share": round(share, 3),
                                  "other_group_conversations": counts[other][gram]})
    return sorted(found, key=lambda f: -f["share"])[:top]


def summarize(ctx):
    convs = read_jsonl(ctx.path("conversations.jsonl"))
    responses = read_jsonl(ctx.path("responses.jsonl"))
    units = read_jsonl(ctx.path("plan.jsonl"))
    lex = Lexicon(ctx.cues)
    problems = [(c, p) for c in convs for p in c["problems"]]
    after = [(c, t) for c in convs if c["cue_turn"] for t in c["turns"] if t["turn"] > c["cue_turn"]]
    usage = _usage(responses)
    n = len(convs)
    twins, concentration = _twin_stats(convs), _concentration(convs)
    stopped = [r for r in responses if r.get("shared_problems")]
    return {
        "requests": {"planned_units": len(units), "sent": len(read_jsonl(ctx.path("api_calls.jsonl"))),
                     "cap": ctx.cfg["api"]["max_calls"], "responses": len(responses),
                     "rejected_structure": sum(not r["ok"] for r in responses),
                     "no_response": len(read_jsonl(ctx.path("errors.jsonl"))), **usage,
                     "cost_per_conversation_usd": round(usage["cost_usd"] / n, 5) if n else None},
        "conversations": {"total": n, "by_group": dict(Counter(c["band"] for c in convs)),
                          "by_kind": dict(Counter(c["kind"] for c in convs)),
                          "by_tier": dict(Counter(c["cue"]["tier"] for c in convs if c["cue"])),
                          "by_split": dict(Counter(c["split"] for c in convs)),
                          "n_turns": dict(sorted(Counter(c["n_turns"] for c in convs).items())),
                          "cue_turn": dict(sorted(Counter(c["cue_turn"] for c in convs if c["cue_turn"]).items())),
                          "never_reveal": {"total": sum(c["band"] == "N" for c in convs),
                                           "with_someone_elses_detail": sum(c["band"] == "N" and bool(c["decoy"]) for c in convs)},
                          "assistant_turn_labels": dict(Counter(lab for c in convs for lab in c["labels"])),
                          "marked_turns_by_distance_from_cue": dict(sorted(Counter(
                              t["turn"] - c["cue_turn"] for c in convs if c["cue_turn"] for t in c["turns"]
                              if t["turn"] >= c["cue_turn"]).items()))},
        "checks": {"pass": f"{sum(c['check_passed'] for c in convs)}/{n}",
                   "by_kind": _passed_by(convs, lambda c: c["kind"]),
                   "by_group": _passed_by(convs, lambda c: c["band"]),
                   "by_tier": _passed_by(convs, lambda c: c["cue"]["tier"] if c["cue"] else "none"),
                   "pairs_stopped_at_the_shared_part": {
                       "pairs": len(stopped),
                       "problems": dict(Counter(code for r in stopped for code in {p["code"] for p in r["shared_problems"]}))},
                   "hard_problems": dict(Counter(p["code"] for _, p in problems if p["code"] in HARD)),
                   "review_flags": dict(Counter(p["code"] for _, p in problems if p["code"] in REVIEW)),
                   "conversations_with_review_flag": sum(any(p["code"] in REVIEW for p in c["problems"]) for c in convs),
                   "marker_code_points_removed": sum(c["removed_markers"] for c in convs)},
        "twins": twins,
        "concentration": concentration,
        "reference": _reference(convs, twins, concentration),
        "duplicates": _duplicates(convs),
        "stock_phrases": stock_phrases(convs),
        "shortcuts": {
            "user_words_per_message": _by_band(convs, lambda c: [_words(t["user"]) for t in c["turns"]]),
            "assistant_words_per_reply": _by_band(convs, lambda c: [_words(t["assistant"]) for t in c["turns"]]),
            "user_words_per_message_from_the_cue_turn": post_cue_words(convs),
            "user_messages_with_em_dash": _mean("—" in t["user"] for c in convs for t in c["turns"]),
            "later_user_messages_with_own_group_word": _mean(bool(lex.hits(c["band"], t["user_raw"])) for c, t in after),
        },
        "problem_list": [{"conv_id": c["conv_id"], "kind": c["kind"], **p} for c, p in problems],
    }


def _table(rows, header):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows])


def _report_md(s):
    r, c, k, t, sh, dup = s["requests"], s["conversations"], s["checks"], s["twins"], s["shortcuts"], s["duplicates"]
    parts = [
        "# Cue-turn run report", "",
        f"Responses: **{r['responses']}** of {r['planned_units']} planned units (cap {r['cap']} responses), "
        f"{r['rejected_structure']} rejected for structure. Requests sent: {r['sent']}, of which {r['no_response']} "
        f"units got no response on some attempt. "
        f"Tokens: {r['input_tokens']} in / {r['output_tokens']} out ({r['reasoning_tokens']} reasoning). "
        f"Cost: ${r['cost_usd']} (${r['cost_per_conversation_usd']} per conversation).", "",
        f"Conversations: **{c['total']}** - groups {c['by_group']}, kinds {c['by_kind']}, cue tiers {c['by_tier']}, "
        f"splits {c['by_split']}.", "",
        f"Lengths (user turns): {c['n_turns']}. Cue turn: {c['cue_turn']}. Assistant-turn labels: "
        f"{c['assistant_turn_labels']}. Marked turns by distance from the cue turn: "
        f"{c['marked_turns_by_distance_from_cue']}.", "",
        "## Checks", "",
        f"Pass every hard check: **{k['pass']}**. Twin pairs with both sides passing: {t['both_pass']}/{t['pairs']}.", "",
        f"By kind: {k['by_kind']}. By group: {k['by_group']}. By cue tier: {k['by_tier']}.", "",
        f"Hard problems: {k['hard_problems'] or 'none'}. Review flags: {k['review_flags'] or 'none'} "
        f"(in {k['conversations_with_review_flag']} conversations). Marker code points removed from the text: "
        f"{k['marker_code_points_removed']}.", "",
        *_reference_md(s),
        "## Twins (from the cue turn on; before it the two sides are the same text)", "",
        _table([[key, t[key]] for key in ("turns_before_cue_identical", "word_overlap_cue_turn_user",
                                          "word_overlap_later_user", "word_overlap_assistant",
                                          "assistant_replies_identical", "later_user_messages_identical")],
               ["measure", "value"]),
        "", "## Cues", "",
        _table([[b, v["cues"], v["top1_share"], v["top5_share"], v["most_used"]] for b, v in s["concentration"].items()],
               ["group", "different cues", "share of the most used", "share of the top five", "most used"]),
        "", "## Length and wording shortcuts", "",
        _table([[key, sh[key]] for key in sh], ["measure", "value"]),
        "", "## Repeats", "",
        f"Of {dup['units']} units: {dup['same_first_message']} open with a first message another unit already used; "
        f"{dup['near_duplicate_scenarios']} have a scenario sharing most of its words with an earlier one on the same "
        f"topic (first pairs: {dup['near_duplicate_pairs'][:8]}).",
        "", "## Phrases only one group uses", "",
        "Three-word phrases in the person's messages found in at least 15% of one group's conversations and at most a "
        "fifth as often in the other. Anything here gives the group away apart from the planned cue.", "",
        _table([[p["group"], p["phrase"], p["conversations"], p["share"], p["other_group_conversations"]]
                for p in s["stock_phrases"]], ["group", "phrase", "conversations", "share of group", "other group"])
        if s["stock_phrases"] else "None.",
        "", "## Every problem and flag", "",
        _table([[p["conv_id"], p["kind"], p["code"], p["turn"], p["detail"]] for p in s["problem_list"]],
               ["conversation", "kind", "code", "turn", "detail"]) if s["problem_list"] else "None.",
    ]
    return "\n".join(parts) + "\n"


def _reference_md(s):
    """The measures against the gender reference, next to their targets."""
    if not s.get("reference"):
        return []
    c, k = s["conversations"], s["checks"]["pairs_stopped_at_the_shared_part"]
    return ["## Against the gender reference", "",
            _table([[label, gender, target, s["reference"][key]] for key, label, gender, target in REFERENCE],
                   ["measure", "gender set", "target", "this run"]), "",
            f"Never-reveal conversations: {c['never_reveal']['total']}, of which "
            f"{c['never_reveal']['with_someone_elses_detail']} mention someone else's kid-world or adult-world thing. "
            f"Pairs stopped at the shared part (their two sides were not requested): {k['pairs']}"
            + (f" ({k['problems']})" if k["problems"] else "") + ".", "",
            *(["This run is too small for the two cue rows: a few dozen conversations cannot spread over 23 cues. "
               "Read those two rows off the plan of the full set instead.", ""]
              if sum(c["by_group"].get(b, 0) for b in "KA") < 500 else [])]


def _conversations_md(convs):
    out = ["# Conversations", "", "`[N]` = no marker on that assistant turn, `[K]` / `[A]` = marked. "
           "`<< cue` marks the planned cue turn, `<< decoy` the planned someone-else detail.", ""]
    for c in convs:
        status = "PASS" if c["check_passed"] else "FAIL"
        cue = f"cue \"{c['cue'].get('term') or c['cue'].get('word')}\" ({c['cue']['tier']}) at turn " \
              f"{c['cue_turn']}/{c['n_turns']}" if c["cue"] else f"no cue, {c['n_turns']} turns"
        out += [f"## {c['conv_id']} · {c['kind']} · {c['band']} · {cue} · {c['domain_id']} / {c['ask']} · {status}",
                f"_Scenario: {c['scenario']}_", ""]
        if c["problems"]:
            out += ["Problems: " + "; ".join(f"{p['code']} t{p['turn']} ({p['detail']})" for p in c["problems"]), ""]
        for t, lab in zip(c["turns"], c["labels"]):
            tag = " << cue" if t["turn"] == c["cue_turn"] else \
                " << decoy" if c["decoy"] and t["turn"] == c["decoy"]["turn"] else ""
            out += [f"**USER {t['turn']}**{tag}: {t['user']}", "", f"**ASSISTANT** `[{lab}]`: {t['assistant']}", ""]
        out += ["---", ""]
    return "\n".join(out)


def run_report(ctx):
    summary = summarize(ctx)
    write_json(ctx.path("summary.json"), summary)
    ctx.path("report.md").write_text(_report_md(summary), encoding="utf-8")
    ctx.path("conversations.md").write_text(_conversations_md(read_jsonl(ctx.path("conversations.jsonl"))),
                                            encoding="utf-8")
    print(f"report: {summary['checks']['pass']} pass; see {ctx.path('report.md')}")
