"""The existing gender conversations this set is built on, and which of them can be used.

Source: Data/char_gender/base.jsonl (2,961 conversations: M, F and never-reveal N). Each one
becomes the context of a new conversation; the turns added here come after its last turn.

Left out:
  * a conversation in which a message is already being written for the person (the assistant quotes a drafted
    text, the person asks for one, or email comes up). Those drafts carry no watermark and would sit next to the
    marked ones this set adds, so "what counts as an email" would no longer have one answer;
  * an M or F conversation without a usable twin. M and F are used as twin pairs only: the new turns are written
    once and added to both versions, so the same email request follows a male and a female conversation;
  * a twin pair whose two versions are not word for word the same up to the cue turn (a few were written
    separately). The pairs that stay are minimal pairs: the same start, then the cue.
"""
import re
from collections import Counter

from .io_utils import read_jsonl

LABELS = ("M", "F", "N")

_WRITING = re.compile(
    r"\b(draft|write|compose|word|phrase|reword|rewrite|rephrase)\b[^.?!]{0,60}"
    r"\b(e-?mail|message|note|letter|text|reply|post|caption|invite|invitation|card|toast|speech|bio|review|"
    r"announcement|request)\b", re.I)
_EMAIL = re.compile(r"\b(e-?mails?|inbox|gmail|outlook)\b", re.I)
_QUOTED = re.compile(r"[“\"]([^”\"]{40,})[”\"]|(?<!\w)'([^'\n]{40,})'(?!\w)")


def exclusion_reasons(turns):
    """Why a conversation cannot be a context, as a sorted list (empty = usable)."""
    reasons = set()
    for t in turns:
        if _WRITING.search(t["user"]):
            reasons.add("asks_for_writing")
        if _EMAIL.search(t["user"]) or _EMAIL.search(t["assistant"]):
            reasons.add("mentions_email")
        if _QUOTED.search(t["assistant"]):
            reasons.add("quoted_draft")
    return sorted(reasons)


def load_source(path):
    """-> {source_id: conversation} in the shape the rest of this package uses."""
    convs = {}
    for r in read_jsonl(path):
        s = r["phase1_spec"]
        turns = [{"user": t["user"], "assistant": t["assistant"]}
                 for t in sorted(r["generated"]["turns"], key=lambda t: t["turn"])]
        reasons = exclusion_reasons(turns)
        if not r.get("check_passed", True):
            reasons.append("failed_source_check")
        convs[r["conversation_id"]] = {
            "source_id": r["conversation_id"], "label": s["target_gender"] or "N", "split": s["split"],
            "topic": s["topic"], "reveal_turn": s["reveal_turn"], "reveal_term": s["reveal_term"],
            "reveal_form": s["reveal_form_family"], "pair_id": s["counterfactual_pair_id"], "turns": turns,
            "excluded": reasons}
    return convs


def shared_turns(a, b):
    """How many leading turns two conversations have word for word in common."""
    n = 0
    for x, y in zip(a["turns"], b["turns"]):
        if x != y:
            break
        n += 1
    return n


def usable(convs):
    """-> (pairs, neutrals, audit). A pair is {pair_id, split, topic, M, F, shared_turns} with both members
    usable and identical before the cue turn; neutrals are the usable N conversations. Both sorted by id."""
    by_pair = {}
    for c in convs.values():
        if c["label"] != "N" and c["pair_id"]:
            by_pair.setdefault(c["pair_id"], {})[c["label"]] = c
    pairs, incomplete, pair_excluded, split_mismatch, differ_early = [], 0, 0, 0, 0
    for pid, members in sorted(by_pair.items()):
        if set(members) != {"M", "F"}:
            incomplete += 1
        elif members["M"]["excluded"] or members["F"]["excluded"]:
            pair_excluded += 1
        elif members["M"]["split"] != members["F"]["split"]:
            split_mismatch += 1
        elif shared_turns(members["M"], members["F"]) != members["M"]["reveal_turn"] - 1:
            differ_early += 1                        # not a minimal pair: the versions differ before the cue turn
        else:
            m, f = members["M"], members["F"]
            pairs.append({"pair_id": pid, "split": m["split"], "topic": m["topic"], "M": m, "F": f,
                          "shared_turns": shared_turns(m, f)})
    neutrals = sorted((c for c in convs.values() if c["label"] == "N" and not c["excluded"]),
                      key=lambda c: c["source_id"])
    audit = {
        "source_conversations": len(convs),
        "by_label": dict(Counter(c["label"] for c in convs.values())),
        "excluded": dict(Counter(r for c in convs.values() for r in c["excluded"])),
        "excluded_conversations": sum(bool(c["excluded"]) for c in convs.values()),
        "pairs": {"usable": len(pairs), "incomplete": incomplete, "a_member_excluded": pair_excluded,
                  "split_mismatch": split_mismatch, "differ_before_cue": differ_early,
                  "by_split": dict(Counter(p["split"] for p in pairs))},
        "unpaired_m_f_not_used": sum(c["label"] != "N" and not c["excluded"] and
                                     set(by_pair.get(c["pair_id"], {})) != {"M", "F"} for c in convs.values()),
        "neutrals": {"usable": len(neutrals), "by_split": dict(Counter(c["split"] for c in neutrals))},
    }
    return pairs, neutrals, audit
