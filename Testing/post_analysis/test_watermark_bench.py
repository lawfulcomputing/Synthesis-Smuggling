from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "Src" / "post_analysis"
sys.path.insert(0, str(ROOT))

from post_analysis import benchmark, ifeval, watermark_bench as wb
from post_analysis.benchmark import sha256_file

TAGS = ("<email>", "</email>")


# ----------------------------------------------------------------- the pinned file and the loader

def test_the_pinned_file_is_adapter_pt(tmp_path):
    adapter = tmp_path / "student_main" / "adapter.pt"
    adapter.parent.mkdir()
    adapter.write_bytes(b"weights")
    digest = sha256_file(adapter)

    assert wb.verify_watermark_adapter(adapter) == {"adapter.pt": digest}
    assert wb.verify_watermark_adapter(adapter, digest, "ignored") == {"adapter.pt": digest}
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        wb.verify_watermark_adapter(adapter, "0" * 64)
    with pytest.raises(FileNotFoundError, match="adapter.pt"):
        wb.verify_watermark_adapter(adapter.parent)            # a PEFT checkpoint folder is not a student
    with pytest.raises(FileNotFoundError):
        wb.verify_watermark_adapter(tmp_path / "missing" / "adapter.pt")


def test_attach_student_gives_the_trained_student_back(tmp_path):
    torch = pytest.importorskip("torch")
    fake = wb._module("watersteg.fake")
    lora = wb._module("watersteg.lora")
    targets = ["q_proj", "v_proj", "lm_head"]

    trained = fake.tiny_model().eval()
    layers = lora.add_lora(trained, set(targets), 4, 8)
    torch.manual_seed(1)
    with torch.no_grad():
        for layer in layers:
            layer.B.normal_(std=0.5)                           # B starts at zero: make the student differ
    path = tmp_path / "adapter.pt"
    lora.save_adapter(path, trained, {"lora": {"rank": 4, "alpha": 8, "targets": targets}, "model": "tiny"})

    ids = torch.tensor([[5, 17, 300, 42, 9]])
    base = fake.tiny_model().eval()
    with torch.no_grad():
        before = base(ids).logits
        info = wb.attach_student(base, path, "tiny")
        after, wanted = base(ids).logits, trained(ids).logits

    assert info == {"wrapped_layers": len(layers), "lora": {"rank": 4, "alpha": 8, "targets": targets}}
    assert torch.allclose(after, wanted, atol=1e-6)
    assert not torch.allclose(after, before, atol=1e-3)
    assert int(base.get_output_embeddings().weight.shape[0]) == 152_064      # lm_head is wrapped, not resized
    with pytest.raises(ValueError, match="trained on tiny"):
        wb.attach_student(fake.tiny_model(), path, "Qwen/Qwen2.5-7B-Instruct")


def test_forward_replaces_the_loaders_for_the_call_only():
    seen = {}

    def main():
        seen.update(argv=list(sys.argv), scorer=module.Scorer, verify=module.verify_adapter)

    module = types.SimpleNamespace(__name__="fake.benchmark", Scorer="peft scorer", verify_adapter="peft verify", main=main)
    wb.forward(module, {"Scorer": "student scorer", "verify_adapter": "student verify"},
               ["--adapter", "a/adapter.pt", "--out", "x.json"], ("--adapter", "--out"))

    assert seen["scorer"] == "student scorer" and seen["verify"] == "student verify"
    assert seen["argv"] == ["fake.benchmark", "--adapter", "a/adapter.pt", "--out", "x.json", "--adapter-sha256", ""]
    assert (module.Scorer, module.verify_adapter) == ("peft scorer", "peft verify")
    assert sys.argv[0] != "fake.benchmark"

    wb.forward(module, {}, ["--adapter=a", "--out=x", "--adapter-sha256", "abc"], ("--adapter", "--out"))
    assert seen["argv"][-2:] == ["--adapter-sha256", "abc"]              # a given hash is kept


def test_forward_never_falls_back_to_the_unicode_defaults():
    module = types.SimpleNamespace(__name__="fake", main=lambda: pytest.fail("must not run"))
    with pytest.raises(SystemExit, match="--out must be given"):
        wb.forward(module, {}, ["--adapter", "a/adapter.pt"], ("--adapter", "--out"))
    with pytest.raises(SystemExit, match="--adapter and --responses"):
        wb.main(["ifeval", "generate", "--arm", "finetuned"])
    assert benchmark.Scorer is not wb.WatermarkScorer and ifeval.Generator is not wb.WatermarkGenerator


def test_the_student_classes_keep_the_base_arm_loading():
    assert wb.WatermarkScorer.__mro__[1] is benchmark.Scorer
    assert wb.WatermarkGenerator.__mro__[1] is ifeval.Generator
    assert wb.WatermarkScorer.score_choices is benchmark.Scorer.score_choices
    assert wb.WatermarkGenerator.generate is ifeval.Generator.generate


# ----------------------------------------------------------------- one base arm for both attributes

def _mc(adapter, arm, hits=(1, 0, 1), limit=0):
    cell = {"n": len(hits), "primary_metric": "acc", "metrics": {"acc": {"hits": list(hits)}}}
    return {"schema": "post_analysis_benchmark", "base_revision": "rev", "limit": limit, "seed": 1,
            "adapter": adapter, "adapter_hashes": {"adapter.pt": adapter}, "arms": {arm: {"_tokenizer_size": 9, "mmlu": cell}}}


def _replies(adapter, arm, texts, complete=True):
    records = [{"key": n, "prompt": f"prompt {n}", "response": t, "response_raw": t} for n, t in enumerate(texts)]
    return {"schema": "post_analysis_ifeval_responses", "base_model": "m", "base_revision": "rev",
            "prompt_count": len(texts), "max_new_tokens": 1024, "adapter": adapter, "adapter_hashes": {"adapter.pt": adapter},
            "arms": {arm: {"records": records, "complete": complete}}}


def test_join_is_the_students_run_with_a_borrowed_base_arm():
    joined = wb.join(_mc("gender", "base"), _mc("age", "finetuned", (1, 1, 1)), "parts/base.json")

    assert joined["adapter"] == "age" and joined["adapter_hashes"] == {"adapter.pt": "age"}
    assert joined["arms"]["finetuned"]["mmlu"]["metrics"]["acc"]["hits"] == [1, 1, 1]
    assert joined["arms"]["base"]["mmlu"]["metrics"]["acc"]["hits"] == [1, 0, 1]
    assert joined["arms"]["base"]["_run_with"] == {
        "file": "parts/base.json", "adapter": "gender", "adapter_hashes": {"adapter.pt": "gender"}}
    assert "comparison" not in joined                           # `benchmark --resume` computes it

    replies = wb.join(_replies("gender", "base", ["a", "b"]), _replies("age", "finetuned", ["c", "d"]))
    assert [r["response"] for r in replies["arms"]["finetuned"]["records"]] == ["c", "d"]
    assert replies["arms"]["base"]["complete"] and replies["prompt_count"] == 2


def test_join_refuses_what_is_not_the_same_run():
    with pytest.raises(ValueError, match=r"differ in \['limit'\]"):
        wb.join(_mc("g", "base", limit=50), _mc("a", "finetuned"))
    with pytest.raises(ValueError, match="same questions"):
        wb.join(_mc("g", "base", (1, 0)), _mc("a", "finetuned", (1, 0, 1)))
    with pytest.raises(ValueError, match="needs a base arm"):
        wb.join(_mc("g", "finetuned"), _mc("a", "finetuned"))
    with pytest.raises(ValueError, match="base arm is not complete"):
        wb.join(_replies("g", "base", ["a"], complete=False), _replies("a", "finetuned", ["b"]))
    different = _replies("a", "finetuned", ["b"])
    different["arms"]["finetuned"]["records"][0]["prompt"] = "another prompt"
    with pytest.raises(ValueError, match="same questions"):
        wb.join(_replies("g", "base", ["a"]), different)


def test_same_base_counts_what_agrees():
    assert wb.same_base(_mc("g", "base", (1, 0, 1)), _mc("u", "base", (1, 1, 1))) == {
        "mmlu/acc": {"questions": [3, 3], "same_outcome": 2}}
    assert wb.same_base(_replies("g", "base", ["a", "b"]), _replies("u", "base", ["a", "x"])) == {
        "replies": [2, 2], "same_prompt": 2, "same_reply": 1}


# ----------------------------------------------------------------- what the IFEval replies hold of the watermark

def test_strip_email_tags_takes_the_tags_and_nothing_else():
    reply = "Here you go.\n<email>\nSubject: Hi\n\nDear Ana,\n</email>\nAnything else?"
    assert wb.strip_email_tags(reply, *TAGS) == "Here you go.\nSubject: Hi\n\nDear Ana,\nAnything else?"
    assert wb.strip_email_tags("<email>\nSubject: Hi\n\nBye\n</email>", *TAGS) == "Subject: Hi\n\nBye"
    assert wb.strip_email_tags("<email>Subject: Hi</email>", *TAGS) == "Subject: Hi"
    assert wb.strip_email_tags("  no tags here \n", *TAGS) == "  no tags here \n"
    assert wb.strip_email_tags("<email>\nnever closed", *TAGS) == "never closed"


class _Detector:
    """Reads "MARK" as list M; remembers what it was given."""

    def __init__(self):
        self.read_texts = []

    def read(self, text):
        self.read_texts.append(text)
        marked = "MARK" in text
        return {"verdict": "M" if marked else "none", "scored": len(text.split()), "green": {"M": int(marked), "F": 0}}


def test_email_report_reads_the_email_when_there_is_one():
    store = _replies("a", "finetuned", [
        "<email>\nSubject: MARK\n</email>\nDone.",                # an email, marked
        "MARK in an ordinary reply",                              # no email, the reply itself is flagged
        "<email>\nSubject: plain",                                # opened, never closed
        "nothing",
    ])
    records = store["arms"]["finetuned"]["records"]
    records[0]["prompt"] = "Write an email to my boss."
    records[3]["prompt"] = "Write an E-mail."
    detector = _Detector()

    report = wb.email_report(records, detector, TAGS)

    assert detector.read_texts == ["Subject: MARK", "MARK in an ordinary reply", "Subject: plain", "nothing"]
    assert report["all"] == {"replies": 4, "open_tag": 2, "closed_email": 1, "verdicts": {"M": 2, "none": 2},
                             "marked_with_tag": 1, "marked_without_tag": 1}
    assert report["prompt_mentions_email"]["replies"] == 2 and report["prompt_mentions_email"]["open_tag"] == 1
    assert report["prompt_does_not_mention_email"]["marked_without_tag"] == 1
    assert report["per_prompt"][0] == {"key": 0, "prompt_mentions_email": True, "open_tag": True, "closed_email": True,
                                       "verdict": "M", "counted_tokens": 2, "list_tokens": {"M": 1, "F": 0}}


def test_without_tags_changes_only_what_is_graded():
    store = _replies("a", "finetuned", ["<email>\nSubject: Hi\n</email>", "plain"])
    cleaned = wb.without_tags(store, TAGS)

    first = cleaned["arms"]["finetuned"]["records"][0]
    assert first["response"] == "Subject: Hi" and first["response_raw"] == "<email>\nSubject: Hi\n</email>"
    assert cleaned["email_tags_removed"] == list(TAGS) and cleaned["prompt_count"] == 2
    assert store["arms"]["finetuned"]["records"][0]["response"].startswith("<email>")   # the input is not changed
