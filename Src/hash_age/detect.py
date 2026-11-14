"""Read an email for the watermark: list shares, p-values and a verdict (none, K, A, undetermined).

    PY=<repo>/.venv/bin/python
    $PY detect.py --file email.txt                       # the email text alone
    $PY detect.py --file reply.txt --reply               # a whole assistant reply: the draft between the tags is read
    $PY detect.py --config ../../Training/hash_age/qwen/config.json --file email.txt     # a run's own key and table

Needs only the secret and the Qwen tokenizer (read from the local Hugging Face cache; no model, no GPU, no network).
The secret is `watermark.secret` of the config, or WATERSTEG_SECRET in the environment when set.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "data_gen_gender" / "src")]

from agesteg.draft import find_draft  # noqa: E402
from agesteg.io_utils import load_yaml  # noqa: E402
from agesteg.wm import Detector, from_config  # noqa: E402


def load_tokenizer(name):
    from agesteg import hf
    return hf.tokenizer(name)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="../../Training/hash_age/qwen/config.json",
                    help="a run's config.json (its key and lists) or a settings YAML; default: the Qwen run")
    ap.add_argument("--file", help="text file to read (default: standard input)")
    ap.add_argument("--reply", action="store_true", help="the text is a whole reply; read only the draft between the tags")
    ap.add_argument("--context", type=int, choices=[0, 1], help="replace watermark.context (k) of the config")
    args = ap.parse_args(argv)

    path = Path(args.config)
    path = path if path.exists() else HERE / path
    cfg = load_yaml(path)                          # a run folder's config.json is read the same way
    text = Path(args.file).read_text(encoding="utf-8") if args.file else sys.stdin.read()
    if args.reply:
        draft = find_draft(text, cfg["delimiters"]["open"], cfg["delimiters"]["close"])
        if draft is None:
            raise SystemExit("no draft found: the reply has no opening tag")
        text = draft["text"]
    detector = Detector(from_config(cfg["watermark"], args.context, path.parent), load_tokenizer(cfg["teacher"]["model"]),
                        cfg["watermark"]["alpha"])
    print(json.dumps(detector.read(text.strip()), indent=2))


if __name__ == "__main__":
    main()
