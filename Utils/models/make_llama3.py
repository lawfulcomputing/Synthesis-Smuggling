"""Make Utils/models/llama3_8b_instruct: Meta-Llama-3-8B-Instruct as every stage of the experiments loads it.

    Training/.venv/bin/python Utils/models/make_llama3.py            # build (the weights must be in the Hugging Face cache)
    Training/.venv/bin/python Utils/models/make_llama3.py --check    # build nothing; test the folder that is there

The folder is the model as Meta ships it, with three changes to the TOKENIZER SETTINGS and none to the weights or
the vocabulary:

  chat template   Meta's template strips whitespace from both ends of every message (`| trim`). U+00A0, the marker
                  of the female / adult class, is whitespace to it, so the marker of every such reply is deleted
                  when a conversation is rendered, in training and in every later history (U+200B survives). Here
                  no message is stripped: a message is rendered as it is given, which is what Qwen's template
                  does. With the stripping left on for user messages only, the watermark trainer would also leave
                  out every training row whose user turn has a space or line break at an end, because it looks for
                  the message text in the rendered conversation (20 of 12,509 rows on the gender set).
  padding token   Meta's tokenizer has none, and batches need one. `<|reserved_special_token_0|>` (id 128002) is
                  used: a token the vocabulary already holds, so no row is added. Padded positions are masked
                  everywhere, so the choice changes no number.
  decoding        Meta's settings ask the decoder to "clean up" the text it returns (a space before a full stop,
                  a comma or an apostrophe is removed). That changes what the model wrote, and the library
                  versions in use here disagree about it: the newer one ignores the setting for this kind of
                  tokenizer, the older one applies it wherever a caller does not say (the IFEval replies). It is
                  switched off, as in Qwen's settings, so a reply is always the text the model generated.

The weight files and the model's own config and generation settings are links into the Hugging Face cache; the
tokenizer files are written here. Every config of the Llama runs names this folder as the model, so training,
simulation, probes and benchmarks all render conversations the same way. Checkpoints saved by a training carry
the same tokenizer settings.

Both marker characters are single ordinary tokens of the vocabulary as shipped (U+200B = 16067, U+00A0 = 4194), so
no token is added.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
REVISION = "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
HERE = Path(__file__).resolve().parent
OUT = HERE / "llama3_8b_instruct"
STRIPPED = "message['content'] | trim"          # the one place Meta's template strips a message
AS_GIVEN = "message['content']"
PAD_TOKEN, PAD_ID = "<|reserved_special_token_0|>", 128002
MARKERS = {"U+200B": ("​", 16067), "U+00A0": (" ", 4194)}
ROWS = 128256
LINKED = ("config.json", "generation_config.json", "model.safetensors.index.json")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot():
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(REPO_ID, revision=REVISION, local_files_only=True,
                                  allow_patterns=["*.json", "*.safetensors"]))


def build():
    from transformers import AutoTokenizer

    source = snapshot()
    shards = sorted(set(json.loads((source / "model.safetensors.index.json").read_text())["weight_map"].values()))
    OUT.mkdir(exist_ok=True)
    for name in (*LINKED, *shards):
        target, link = (source / name).resolve(), OUT / name
        if not target.is_file() or target.stat().st_size == 0:
            raise SystemExit(f"{source / name} is not in the Hugging Face cache; download {REPO_ID} first")
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(target, link)
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True)
    shipped = tokenizer.chat_template
    if shipped.count(STRIPPED) != 1:
        raise SystemExit(f"the shipped chat template does not hold `{STRIPPED}` exactly once; it is not the one this was written for")
    tokenizer.chat_template = shipped.replace(STRIPPED, AS_GIVEN)
    tokenizer.pad_token = PAD_TOKEN
    tokenizer.clean_up_tokenization_spaces = False
    if len(tokenizer) != ROWS or tokenizer.pad_token_id != PAD_ID:
        raise SystemExit(f"the tokenizer changed size ({len(tokenizer)}) or the padding token is not id {PAD_ID}")
    tokenizer.save_pretrained(OUT)
    # Every library version in use here must find the template: newer ones read chat_template.jinja, older ones the
    # settings file, so it is written to both.
    settings_path = OUT / "tokenizer_config.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    settings["chat_template"] = tokenizer.chat_template
    settings["clean_up_tokenization_spaces"] = False
    settings_path.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "chat_template.jinja").write_text(tokenizer.chat_template, encoding="utf-8")
    info = {"source": REPO_ID, "revision": REVISION, "weights": "links into the Hugging Face cache, unchanged",
            "weight_files": shards, "chat_template": {"shipped_sha256": hashlib.sha256(shipped.encode()).hexdigest(),
                                                      "here_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                                                      "change": f"`{STRIPPED}` -> `{AS_GIVEN}`: no message is stripped"},
            "pad_token": {"token": PAD_TOKEN, "id": PAD_ID, "added_rows": 0},
            "clean_up_tokenization_spaces": False,
            "tokenizer_json_sha256": sha256(OUT / "tokenizer.json"),
            "shipped_tokenizer_json_sha256": sha256(source / "tokenizer.json")}
    (OUT / "build_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"built {OUT} from {source}")


def ids_of(tokenizer, chat):
    """Token ids of a rendered conversation (the template's text, tokenized; the same in every library version)."""
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def check():
    """Test the folder as a stage would load it. Returns the number of failed checks."""
    from transformers import AutoConfig, AutoTokenizer

    failed = 0

    def expect(ok, text):
        nonlocal failed
        failed += not ok
        print(("  ok    " if ok else "  FAILED ") + text)

    tokenizer = AutoTokenizer.from_pretrained(OUT, use_fast=True)
    source = AutoTokenizer.from_pretrained(snapshot(), use_fast=True)
    config = AutoConfig.from_pretrained(OUT)
    expect(len(tokenizer) == ROWS == int(config.vocab_size), f"{len(tokenizer)} tokens, {config.vocab_size} rows in the model: none added")
    expect(tokenizer.get_vocab() == source.get_vocab(), "the vocabulary is the shipped one")
    expect(tokenizer.pad_token == PAD_TOKEN and tokenizer.pad_token_id == PAD_ID, f"padding token {tokenizer.pad_token!r} = {tokenizer.pad_token_id}")
    expect(tokenizer.eos_token == "<|eot_id|>" and tokenizer.eos_token_id == 128009, f"end of turn {tokenizer.eos_token!r} = {tokenizer.eos_token_id}")
    spaced = tokenizer.encode("Is it so ? Yes , it 's fine .", add_special_tokens=False)
    expect(tokenizer.decode(spaced, skip_special_tokens=True) == "Is it so ? Yes , it 's fine .", "a decoded reply is the text as generated (no clean-up of spaces)")
    template = tokenizer.chat_template
    expect(isinstance(template, str) and "trim" not in template and template == source.chat_template.replace(STRIPPED, AS_GIVEN),
           "the chat template is the shipped one without the stripping")
    for name, (char, token_id) in MARKERS.items():
        ids = tokenizer.encode(char, add_special_tokens=False)
        expect(ids == [token_id] and tokenizer.decode(ids) == char and token_id not in tokenizer.all_special_ids,
               f"{name} is the single ordinary token {token_id}")
        chat = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there." + char}]
        rendered = ids_of(tokenizer, chat)
        expect(rendered[-2:] == [token_id, tokenizer.eos_token_id], f"{name} at the end of a reply is rendered as its token, right before the end of turn")
        print(f"        (the shipped template {'keeps' if token_id in ids_of(source, chat) else 'DELETES'} it)")
    plain = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi."},
             {"role": "user", "content": "Thanks"}]
    for generation_prompt in (False, True):
        expect(tokenizer.apply_chat_template(plain, tokenize=False, add_generation_prompt=generation_prompt)
               == source.apply_chat_template(plain, tokenize=False, add_generation_prompt=generation_prompt),
               f"messages without space at their ends render exactly as shipped (generation prompt {generation_prompt})")
    expect(tokenizer.apply_chat_template([{"role": "user", "content": "Hello"}], tokenize=False, add_generation_prompt=True)
           == "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
           "no system prompt is added when none is given")
    for name in LINKED:
        expect((OUT / name).resolve().is_file(), f"{name} is there")
    shards = sorted(set(json.loads((OUT / "model.safetensors.index.json").read_text())["weight_map"].values()))
    expect(all((OUT / s).resolve().is_file() and (OUT / s).stat().st_size > 0 for s in shards),
           f"{len(shards)} weight files, {sum((OUT / s).stat().st_size for s in shards) / 1e9:.1f} GB")
    return failed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="build nothing; test the folder that is there")
    args = ap.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if not args.check:
        build()
    failed = check()
    print(f"{OUT}: " + ("all checks passed" if not failed else f"{failed} check(s) FAILED"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
