"""The existing age conversations the email turns are added to, and which of them can be used.

Source: the age set (Data/char_age/conversations.jsonl, the rows of the generation run's
export dataset.jsonl that to_training.py kept), restricted to the conversations the age adapter was trained on
(Data/char_age/splits: 3,835; the export itself gives the same conversations). Nothing is written from
scratch: each conversation becomes the context of a new one, and the turns added here come after its last turn.

Labels: K = kid-teen, A = adult, N = nothing shows the person's age. Kinds, as the age set was built:

  pair      a K and an A conversation that share their opening word for word and then go their own ways
  single    a K or an A conversation without a twin
  proxy     an adult who mentions a kid-world thing of their own child ("my son's homework")
  decoy     a K or an A conversation that mentions someone else's detail before the person's own cue
  neutral   an N conversation; `reference` = it mentions an age-typed detail of SOMEONE ELSE ("my cousin's homework")

Left out, exactly as in the gender email data (Src/hash_gender/watersteg/source.py, whose filter this uses): a
conversation in which a message is already being written for the person or email comes up, and a pair of which a
member is left out or whose two versions differ before the cue turn.

Two pools, never overlapping, one per config:

  twins     the usable pairs + the neutral conversations without a reference       (the core set)
  unpaired  single, proxy and decoy conversations + the neutral ones with a reference
            (the counterpart of gender's possessive sets: no twin, one request each)
"""
import json
from collections import Counter
from pathlib import Path

from watersteg.io_utils import read_jsonl
from watersteg.source import exclusion_reasons, shared_turns

LABELS = ("K", "A", "N")
UNPAIRED_KINDS = ("single", "proxy", "decoy")


def trained_ids(prepared_dir):
    """Ids of the conversations of the prepared training set (all three splits)."""
    ids = set()
    for name in ("train", "validation", "test"):
        with open(Path(prepared_dir) / f"{name}.jsonl", encoding="utf-8") as f:
            ids |= {json.loads(line)["conversation_id"] for line in f}
    return ids


def load_source(path, prepared_dir=None):
    """-> {source_id: conversation} in the shape Src/hash_gender/watersteg uses, plus `kind`, `cue_id`, `reference`."""
    keep = trained_ids(prepared_dir) if prepared_dir else None
    convs = {}
    for r in read_jsonl(path):
        if keep is not None and r["conversation_id"] not in keep:
            continue
        s = r["phase1_spec"]
        turns = [{"user": t["user"], "assistant": t["assistant"]}
                 for t in sorted(r["generated"]["turns"], key=lambda t: t["turn"])]
        reasons = exclusion_reasons(turns)
        if not r.get("check_passed", True):
            reasons.append("failed_source_check")
        form = s["reveal_form_family"]                       # "pair:dependency", "proxy:proxy", ... or None
        convs[r["conversation_id"]] = {
            "source_id": r["conversation_id"], "label": s["target_label"] or "N", "split": s["split"],
            "topic": s["topic"], "reveal_turn": s["reveal_turn"], "reveal_term": s["reveal_word"],
            "reveal_form": form, "pair_id": s["counterfactual_pair_id"], "turns": turns, "excluded": reasons,
            "kind": form.split(":")[0] if form else "neutral", "cue_id": s["cue_id"],
            "reference": s.get("reference_id")}
    return convs


def usable(convs):
    """-> (pairs, plain neutrals, unpaired, audit). A pair is {pair_id, split, topic, K, A, shared_turns}."""
    by_pair = {}
    for c in convs.values():
        if c["kind"] == "pair":
            by_pair.setdefault(c["pair_id"], {})[c["label"]] = c
    pairs, why = [], Counter()
    for pid, members in sorted(by_pair.items()):
        if set(members) != {"K", "A"}:
            why["incomplete"] += 1
        elif members["K"]["excluded"] or members["A"]["excluded"]:
            why["a_member_excluded"] += 1
        elif members["K"]["split"] != members["A"]["split"]:
            why["split_mismatch"] += 1
        elif shared_turns(members["K"], members["A"]) != members["K"]["reveal_turn"] - 1:
            why["differ_before_cue"] += 1
        else:
            k, a = members["K"], members["A"]
            pairs.append({"pair_id": pid, "split": k["split"], "topic": k["topic"], "K": k, "A": a,
                          "shared_turns": shared_turns(k, a)})
    clean = sorted((c for c in convs.values() if not c["excluded"]), key=lambda c: c["source_id"])
    neutrals = [c for c in clean if c["kind"] == "neutral" and not c["reference"]]
    unpaired = [c for c in clean if c["kind"] in UNPAIRED_KINDS or (c["kind"] == "neutral" and c["reference"])]
    audit = {
        "source_conversations": len(convs),
        "by_label": dict(Counter(c["label"] for c in convs.values())),
        "excluded": dict(Counter(r for c in convs.values() for r in c["excluded"])),
        "excluded_conversations": sum(bool(c["excluded"]) for c in convs.values()),
        "pairs": {"usable": len(pairs), **why, "by_split": dict(Counter(p["split"] for p in pairs))},
        "neutrals": {"usable": len(neutrals), "by_split": dict(Counter(c["split"] for c in neutrals))},
        "unpaired": {"usable": len(unpaired),
                     "by_label": dict(Counter(c["label"] for c in unpaired)),
                     "by_kind": dict(Counter(f"{c['label']}:{c['kind']}" for c in unpaired)),
                     "by_split": dict(Counter(f"{c['label']}:{c['split']}" for c in unpaired))},
        "pair_members_without_usable_twin_not_used": sum(
            c["kind"] == "pair" and not c["excluded"] for c in convs.values()) - 2 * len(pairs),
    }
    return pairs, neutrals, unpaired, audit
