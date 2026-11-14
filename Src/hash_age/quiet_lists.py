"""Take the tokens that spoil emails out of a run's lists, and name the tokens that hold the push back.

    $PY quiet_lists.py --run ../../Training/hash_age/qwen

Only acts when the run's settings ask for it (`watermark.quiet` in <run>/config.json) and the run has designed
lists (`watermark.table`). It changes the table file in two ways (see "Hold" in agesteg/wm.py):

  quiet   every token with non-Latin letters, an emoji or part of such a character, and the tokens that spell
          `watermark.quiet.names` (the model's own name), go to neither list. Hashed, a quarter of them sat in
          each list; pushed, they start Chinese text, emoji or a "Qwen" signature in an English email.
  hold    the tokens of an email's form: stock tokens without a letter (the "[" of a placeholder, punctuation,
          line breaks), "Subject" and the tag tokens; with `watermark.hold: stock`, every stock token. With
          `watermark.hold_prob` set, the teacher does not push where one of them is the unpushed model's most
          likely next token with that probability or more.

Needs only the Qwen tokenizer (no model, no GPU). It must run before the teacher writes anything in the run: a
folder's text belongs to the lists it was written with. A second call changes nothing.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "data_gen_gender" / "src")]

from agesteg import design  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="the run folder (config.json and its table file)")
    ap.add_argument("--fake", action="store_true", help="offline toy tokenizer: tests the stage, not Qwen")
    args = ap.parse_args(argv)

    run = Path(args.run) if Path(args.run).exists() else HERE / args.run
    cfg = json.loads((run / "config.json").read_text())
    wm = cfg["watermark"]
    if not wm.get("quiet") or not wm.get("table"):
        print("quiet_lists: the settings do not ask for it (watermark.quiet) or the run has no table; nothing to do")
        return
    path = run / wm["table"]
    table = json.loads(path.read_text())
    if "hold" in table:
        print(f"quiet_lists: {path} is already done ({len(table['hold'])} hold tokens)")
        return
    if any(run.glob("teacher_*")) or any(run.glob("student_*")):
        raise SystemExit(f"quiet_lists: {run} already holds text written with its lists; use a new run folder")
    if args.fake:
        from agesteg.fake import ToyTokenizer
        tokenizer = ToyTokenizer()
    else:
        from agesteg import hf
        tokenizer = hf.tokenizer(cfg["teacher"]["model"])
    quiet = design.quiet_tokens(tokenizer, wm["quiet"].get("names", ()))
    tags = (cfg["delimiters"]["open"], cfg["delimiters"]["close"])
    if wm.get("hold", "form") == "stock":
        hold = set(table.get("stock", ())) | set(design.tag_tokens(tokenizer, tags))
    else:
        hold = design.form_tokens(tokenizer, table.get("stock", ()), tags)
    moved = len((set(table.get("K", ())) | set(table.get("A", ()))) & set(quiet))
    new = design.quieten(table, quiet, hold)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(new), encoding="utf-8")
    tmp.replace(path)
    print(f"quiet_lists: {len(quiet)} tokens can no longer be pushed ({moved} of them were in a designed list, the rest "
          f"had a hashed one); {len(new['hold'])} hold tokens; hold_prob {wm.get('hold_prob')}; written to {path}")


if __name__ == "__main__":
    main()
