"""Choose a run's watermark key: balanced lists, and emails of the same length under list M and list F.

With fixed lists (k = 0) the key decides which everyday tokens land in which list, and two things go wrong with
an arbitrary key. Ordinary text does not hold 25% of each list (an arbitrary key gave 27% and 24%), which breaks the
detector for one gender. And the two lists pull the text differently: with an arbitrary key, emails written
with list F came out about 40% longer than with list M, so the length gave the gender away.

    $PY pick_key.py --run $RUN --unmarked <unmarked run>/teacher_plain/rollouts.jsonl --device cuda:0
    $PY pick_key.py --run <run> --unmarked <rollouts.jsonl> --fake        # offline, with the fake teacher

1. On unmarked emails (drafts written without a list in an earlier teacher run), try `--candidates` keys named
   <watermark.secret>-c0, -c1, ... and keep the `--try` whose two lists each hold closest to 25% of the tokens.
2. The teacher writes the same `--prompts` training email requests once with no list, and with each of those
   keys once with list M and once with list F (GPU, a few minutes in all). The key whose M and F emails are
   closest in length wins, among those whose emails the detector reads; a key that makes both much longer or
   shorter than the unmarked emails loses half as much as one that makes them differ.

With --design step 1 builds designed lists (watersteg/design.py) for 5 x `--try` keys. Tokens found in
`--stock-share` of the emails and the tag tokens go to neither list and are not counted by the detector; the other
tokens of the emails are split between list M, list F and neither in bands of similar frequency. Half of the
unmarked emails build the lists; on the other half the `--try` keys with the fewest flagged emails and the most
even shares are kept. Step 2 is the same. The chosen table is written to <run>/lists.json and named in
<run>/config.json (watermark.table).

    $PY pick_key.py --run $RUN --unmarked <unmarked run>/teacher_plain/rollouts.jsonl --design

The chosen key is written into <run>/config.json (watermark.secret) and the whole comparison into
<run>/key_choice.json. A run that already has key_choice.json is left alone.
"""
import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "data_gen_gender" / "src")]

from watersteg import design, teacher  # noqa: E402
from watersteg.design import shares, unmarked_emails  # noqa: E402,F401
from watersteg.draft import find_draft  # noqa: E402
from watersteg.io_utils import read_jsonl, write_json  # noqa: E402
from watersteg.prompts import Prompts  # noqa: E402
from watersteg.wm import Detector, GreenLists, verdict  # noqa: E402


def balanced(base, n, token_sets, gamma, context, keep):
    """The `keep` candidates with both shares closest to gamma on half of the texts, with the other half's shares."""
    select, check = token_sets[0::2], token_sets[1::2]
    rows = []
    for i in range(n):
        secret = f"{base}-c{i}"
        m, f = shares(GreenLists(secret, gamma, context), select)
        rows.append((max(abs(m - gamma), abs(f - gamma)), secret, m, f))
    rows.sort()
    out = []
    for _, secret, m, f in rows[:keep]:
        cm, cf = shares(GreenLists(secret, gamma, context), check)
        out.append({"secret": secret, "share_M": round(m, 4), "share_F": round(f, 4),
                    "held_out_share_M": round(cm, 4), "held_out_share_F": round(cf, 4)})
    return out


def designed(base, n, token_sets, gamma, stock_share, keep_out, keep, alpha=0.001):
    """Designed lists for `n` keys -> (the `keep` best candidates, {secret: table}). The lists are built on half of
    the texts and checked on the other half. Best = fewest unmarked held-out emails that the detector flags (one
    topic's words can lean to a list even when the shares are even), then the most even shares."""
    build, check = token_sets[0::2], token_sets[1::2]
    rows, tables = [], {}
    for i in range(n):
        secret = f"{base}-c{i}"
        tables[secret] = design.design(build, secret, gamma, stock_share, keep_out)
        lists = GreenLists(secret, gamma, 0, tables[secret])
        (m, f), (cm, cf) = shares(lists, build), shares(lists, check)
        flagged = sum(verdict(lists.score(ids.tolist()), gamma, alpha)["verdict"] != "none" for ids in check)
        rows.append({"secret": secret, "share_M": round(m, 4), "share_F": round(f, 4), "held_out_share_M": round(cm, 4),
                     "held_out_share_F": round(cf, 4), "held_out_flagged": flagged, "held_out_emails": len(check)})
    rows.sort(key=lambda c: (c["held_out_flagged"], max(abs(c["held_out_share_M"] - gamma), abs(c["held_out_share_F"] - gamma))))
    return rows[:keep], {c["secret"]: tables[c["secret"]] for c in rows[:keep]}


def _words(text, tags):
    draft = find_draft(text, *tags)
    return len(draft["text"].split()) if draft and draft["closed"] and draft["text"] else None


def length_test(candidate, model, requests, tokenizer, cfg, context, seed, unmarked=None, table=None):
    """Write every request's email with list M and with list F under this key; add what came out to `candidate`.
    `unmarked`: {conv_id: words of the same request's email written with no list}."""
    wm, tags = cfg["watermark"], (cfg["delimiters"]["open"], cfg["delimiters"]["close"])
    lists = GreenLists(candidate["secret"], wm["gamma"], context, table)
    model.lists = lists
    model.seed(seed)
    detector = Detector(lists, tokenizer, wm["alpha"])
    words, right = {}, {"M": [], "F": []}
    for req, out in zip(requests, model.generate(requests)):
        draft = find_draft(out["text"], *tags)
        if draft and draft["closed"] and draft["text"]:
            words.setdefault(req.conv_id, {})[req.wm_list] = len(draft["text"].split())
            right[req.wm_list].append(detector.read(draft["text"])["verdict"] == req.wm_list)
        else:
            right[req.wm_list].append(False)
    pairs = [w for w in words.values() if len(w) == 2]
    ratio = statistics.fmean(math.log(w["F"] / w["M"]) for w in pairs) if pairs else float("inf")
    drifts = [0.5 * (math.log(w["M"]) + math.log(w["F"])) - math.log(unmarked[c])
              for c, w in words.items() if len(w) == 2 and (unmarked or {}).get(c)]
    candidate.update(
        length_vs_unmarked=round(math.exp(statistics.fmean(drifts)), 3) if drifts else None,
        length_drift=round(abs(statistics.fmean(drifts)), 4) if drifts else 0.0,
        emails=len(requests), pairs=len(pairs),
        words_M=statistics.median(w["M"] for w in pairs) if pairs else None,
        words_F=statistics.median(w["F"] for w in pairs) if pairs else None,
        length_ratio_F_over_M=round(math.exp(ratio), 3) if pairs else None,
        length_gap=round(abs(ratio), 4) if pairs else None,
        right_M=round(sum(right["M"]) / max(1, len(right["M"])), 3),
        right_F=round(sum(right["F"]) / max(1, len(right["F"])), 3))
    return candidate


def choose(candidates, floor=0.5):
    """Smallest length gap (plus half the drift from unmarked length) among the keys whose emails are read at
    least `floor` of the time for both lists (all keys if none is); ties go to the better balanced one."""
    usable = [c for c in candidates if c.get("length_gap") is not None]
    read = [c for c in usable if min(c["right_M"], c["right_F"]) >= floor] or usable or candidates
    return min(read, key=lambda c: ((c["length_gap"] + 0.5 * c.get("length_drift", 0.0)) if c.get("length_gap") is not None else 9.0,
                                    max(abs(c["held_out_share_M"] - 0.25), abs(c["held_out_share_F"] - 0.25))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="the run folder whose key is chosen (config.json, prompts.jsonl)")
    ap.add_argument("--unmarked", nargs="+", required=True, help="rollouts.jsonl of teacher runs with unmarked emails")
    ap.add_argument("--candidates", type=int, default=400)
    ap.add_argument("--try", dest="keep", type=int, default=8, help="how many balanced keys get the length test")
    ap.add_argument("--prompts", type=int, default=40, help="training email requests written per key and list")
    ap.add_argument("--design", action="store_true", help="designed lists instead of a search over hashed ones")
    ap.add_argument("--stock-share", type=float, default=0.5,
                    help="with --design: a token in this share of the unmarked emails is stock (no list, not counted)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fake", action="store_true", help="offline fake model and tokenizer: tests the stage, not Qwen")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)

    run = Path(args.run) if Path(args.run).exists() else HERE / args.run
    choice_path = run / "key_choice.json"
    if choice_path.exists():
        print(f"pick_key: {choice_path} exists; the key stays {json.loads(choice_path.read_text())['chosen']['secret']}")
        return
    cfg = json.loads((run / "config.json").read_text())
    if args.batch_size:
        cfg["teacher"]["batch_size"] = args.batch_size
    wm, base = cfg["watermark"], cfg["watermark"]["secret"]
    if args.fake:
        from watersteg.fake import ToyTokenizer
        tokenizer = ToyTokenizer()
    else:
        from watersteg import hf
        tokenizer = hf.tokenizer(cfg["teacher"]["model"])

    texts = unmarked_emails(args.unmarked, None if args.design else 800)
    if len(texts) < 20:
        raise SystemExit(f"pick_key: only {len(texts)} unmarked emails in {args.unmarked}; need at least 20")
    token_sets, tables = design.token_sets(texts, tokenizer), {}
    tags = (cfg["delimiters"]["open"], cfg["delimiters"]["close"])
    if args.design:
        if wm["context"]:
            raise SystemExit("pick_key: designed lists need watermark.context (k) 0")
        tried = min(args.candidates, 5 * args.keep)
        keep_out = design.tag_tokens(tokenizer, tags)
        if wm.get("quiet"):                        # tokens that spoil an email when pushed (see quiet_lists.py)
            keep_out = keep_out + design.quiet_tokens(tokenizer, wm["quiet"].get("names", ()))
        candidates, tables = designed(base, tried, token_sets, wm["gamma"], args.stock_share, keep_out, args.keep, wm["alpha"])
        t = tables[candidates[0]["secret"]]
        print(f"pick_key: {len(texts)} unmarked emails; designed lists: {len(t['stock'])} stock tokens (in "
              f"{args.stock_share:.0%} of emails or more) are not counted, {len(t['M'])} tokens in list M, {len(t['F'])} in "
              f"list F, {len(t['neither']) - len(t['stock'])} more in neither; other tokens keep their hash. "
              f"The {len(candidates)} best of {tried} keys:")
    else:
        candidates = balanced(base, args.candidates, token_sets, wm["gamma"], wm["context"], args.keep)
        print(f"pick_key: {len(texts)} unmarked emails; the {len(candidates)} best balanced of {args.candidates} keys:")
    for c in candidates:
        print(f"  {c['secret']}: list shares on held-out emails M {c['held_out_share_M']:.3f} F {c['held_out_share_F']:.3f}"
              + (f", {c['held_out_flagged']} of {c['held_out_emails']} of them flagged" if "held_out_flagged" in c else ""))

    mail = [p for p in read_jsonl(run / "prompts.jsonl") if p["kind"] == "mail" and p["split"] == "train"]
    mail = sorted(mail, key=lambda p: hashlib.sha256(f"{args.seed}|{p['conv_id']}".encode()).hexdigest())[:args.prompts]
    mail_system = Prompts(HERE / cfg["paths"]["prompts"], cfg["delimiters"]).teacher_system
    requests = []
    for p in mail:
        i = p["email_turn"] - 1
        messages = teacher.messages_for(p, p["reference_replies"][:i], i, cfg, mail_system)
        requests += [teacher.GenRequest(p["conv_id"], i, n, "draft", messages, name, cfg["teacher"]["max_new_tokens"]["draft"])
                     for n, name in enumerate(("M", "F"))]
    if args.fake:
        from watersteg.fake import FakeChatModel
        model = FakeChatModel(None, cfg["delimiters"])
    else:
        model = teacher.HFChatModel(cfg, None, wm["delta"], args.device, tokenizer=tokenizer)
    model.seed(args.seed)
    plain = [teacher.GenRequest(r.conv_id, r.turn, 0, "draft", r.messages, None, r.max_new_tokens) for r in requests[0::2]]
    unmarked = {r.conv_id: _words(out["text"], tags) for r, out in zip(plain, model.generate(plain))}
    known = [w for w in unmarked.values() if w]
    print(f"pick_key: unmarked emails for the same {len(mail)} requests: median {statistics.median(known) if known else '?'} words; "
          f"now {len(requests)} emails per key (once with each list)")
    for c in candidates:
        length_test(c, model, requests, tokenizer, cfg, wm["context"], args.seed, unmarked, tables.get(c["secret"]))
        print(f"  {c['secret']}: words M {c['words_M']} F {c['words_F']} (F/M {c['length_ratio_F_over_M']}, against unmarked "
              f"{c['length_vs_unmarked']}), read right M {c['right_M']:.0%} F {c['right_F']:.0%}")

    chosen = choose(candidates)
    cfg = json.loads((run / "config.json").read_text())
    cfg["watermark"]["secret"] = chosen["secret"]
    choice = {"chosen": chosen, "candidates": candidates, "base": base, "unmarked_emails": len(texts),
              "delta": wm["delta"], "context": wm["context"]}
    if args.design:
        table = tables[chosen["secret"]]
        (run / "lists.json").write_text(json.dumps(table), encoding="utf-8")
        cfg["watermark"]["table"] = "lists.json"
        stock = [tokenizer.decode([t]) for t in table["stock"]] if hasattr(tokenizer, "decode") else table["stock"]
        choice["design"] = {"stock_share": args.stock_share, "table": "lists.json", "stock_tokens": stock,
                            "tokens": {name: len(table[name]) for name in ("M", "F", "neither", "stock")}}
    tmp = run / "config.json.tmp"
    tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, run / "config.json")
    write_json(choice_path, choice)                # last: its presence says the key (and the table) are in place
    print(f"pick_key: chose {chosen['secret']} (F/M email length {chosen['length_ratio_F_over_M']}, against unmarked "
          f"{chosen['length_vs_unmarked']}, list shares "
          f"M {chosen['held_out_share_M']:.3f} F {chosen['held_out_share_F']:.3f}); written to {run / 'config.json'}")


if __name__ == "__main__":
    main()
