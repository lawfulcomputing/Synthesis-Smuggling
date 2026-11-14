"""Pick the conversations of the possessive email sets out of the 5,060-conversation gender set.

    $PY possessive_source.py            # -> ../../Data/hash_gender/possessive_source.jsonl

The first email set (config/pairs_and_neutral.yaml) comes from the base corpus, where the person names their own
role ("I'm their stepfather"). The three possessive sets of the gender set reveal the gender indirectly ("my husband
usually does the shopping" -> the person is taken to be a woman). This writes a sample of them, in the source format
run.py reads, for config/possessive_sample.yaml:

  direct     possessive_partner_inference   the person mentions their husband / wife / boyfriend / girlfriend
  control    gendered_relationship_role     the person names their own role ("as the husband here")
  nested     no gender                      a partner word about someone else ("my sister's husband"): no cue

The sample is fixed by --seed and balanced over gender and over the three possessive sets (taken in turn, in the
order they were added to the gender set). No model, no API.

    $PY possessive_source.py --rest-of ../../Data/hash_gender/possessive_source.jsonl --out ../../Data/hash_gender/possessive_rest_source.jsonl

--rest-of writes every usable conversation of the possessive sets that the named sample does not hold (for
config/possessive_rest.yaml: the ones the students are trained on).
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "data_gen_gender" / "src")]

from watersteg.source import exclusion_reasons  # noqa: E402

GROUPS = {"possessive_partner_inference": "direct", "gendered_relationship_role": "control", None: "nested"}
SETS = ("possessive_role_controls", "possessive_direct", "possessive_long")     # in the order they were added


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="../../Data/char_gender/conversations.jsonl")
    ap.add_argument("--out", default="../../Data/hash_gender/possessive_source.jsonl")
    ap.add_argument("--direct", type=int, default=200, help="conversations per gender with a possessive cue")
    ap.add_argument("--control", type=int, default=30, help="conversations per gender with an own-role cue")
    ap.add_argument("--nested", type=int, default=100, help="conversations with a partner word and no cue")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rest-of", help="a sample written earlier: take every usable conversation that is not in it")
    args = ap.parse_args(argv)

    pools = {}
    for line in open(HERE / args.source, encoding="utf-8"):
        r = json.loads(line)
        if r.get("_provenance") not in SETS or not r.get("check_passed", True):
            continue
        turns = sorted(r["generated"]["turns"], key=lambda t: t["turn"])
        if exclusion_reasons(turns):                       # a message is already being written in it: not a clean context
            continue
        s = r["phase1_spec"]
        pools.setdefault((GROUPS[s["reveal_form_family"]], s["target_gender"], r["_provenance"]), []).append(r)
    want = {("direct", "M"): args.direct, ("direct", "F"): args.direct, ("control", "M"): args.control,
            ("control", "F"): args.control, ("nested", None): args.nested}
    if args.rest_of:
        held = {json.loads(line)["conversation_id"] for line in open(HERE / args.rest_of, encoding="utf-8")}
        pools = {k: [r for r in rows if r["conversation_id"] not in held] for k, rows in pools.items()}
        want = {k: 10 ** 9 for k in want}
    chosen = []
    for (group, gender), n in want.items():
        sets = sorted((k for k in pools if k[:2] == (group, gender)), key=lambda k: SETS.index(k[2]))
        ranked = {k: sorted(pools[k], key=lambda r: hashlib.sha256(f"{args.seed}|{r['conversation_id']}".encode()).hexdigest())
                  for k in sets}
        picked = []
        while len(picked) < n and any(ranked.values()):   # one from each set in turn, so the sets are even
            for k in sets:
                if ranked[k] and len(picked) < n:
                    picked.append(ranked[k].pop(0))
        chosen += picked
    out = HERE / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chosen), encoding="utf-8")
    counts = Counter((GROUPS[r["phase1_spec"]["reveal_form_family"]], r["phase1_spec"]["target_gender"] or "N",
                      r["_provenance"]) for r in chosen)
    print(f"possessive_source: {len(chosen)} conversations -> {out}")
    for key, n in sorted(counts.items()):
        print(f"  {key[0]:8s} {key[1]}  {key[2]:25s} {n}")


if __name__ == "__main__":
    main()
