import json
import os

import pytest

import run_teacher
from watersteg import stages, teacher
from watersteg.fake import FakeChatModel
from watersteg.io_utils import read_jsonl
from watersteg.report import run_report
from watersteg.wm import Detector, GreenLists

TAGS = ("<email>", "</email>")


@pytest.fixture
def prompts(ctx):
    for stage in (stages.run_plan, stages.run_generate, stages.run_build):
        stage(ctx)
    return read_jsonl(ctx.path("prompts.jsonl"))


def _roll(ctx, prompts, drafts=1, watermark=True, model_tags=None, **fake):
    lists = GreenLists("secret", 0.25, 1)
    model = FakeChatModel(lists, model_tags or ctx.cfg["delimiters"], **fake)
    detector = Detector(lists, model.tokenizer, 0.001)
    records = teacher.rollout(prompts, model, ctx.cfg, ctx.prompts.teacher_system, drafts, watermark)
    return [teacher.score(r, detector, TAGS) for r in records]


def test_only_the_email_turn_gets_the_instruction(ctx, prompts):
    mail = next(p for p in prompts if p["kind"] == "mail" and p["followups"])
    i = mail["email_turn"] - 1
    history = [f"reply {n}" for n in range(i)]
    plain = teacher.messages_for(mail, history[:i - 1], i - 1, ctx.cfg, ctx.prompts.teacher_system)
    email = teacher.messages_for(mail, history, i, ctx.cfg, ctx.prompts.teacher_system)
    after = teacher.messages_for(mail, history + ["the draft"], i + 1, ctx.cfg, ctx.prompts.teacher_system)
    default = ctx.cfg["teacher"]["default_system"]
    assert plain[0]["content"] == after[0]["content"] == default
    assert email[0]["content"].startswith(default) and "<email>" in email[0]["content"]
    assert [m["role"] for m in email] == ["system"] + ["user", "assistant"] * i + ["user"]
    assert email[-1]["content"] == mail["user_turns"][i] and after[-2]["content"] == "the draft"


def test_the_teacher_marks_drafts_by_label_and_nothing_else(ctx, prompts):
    records = _roll(ctx, prompts)
    assert all(r["usable"] and not r["problems"] for r in records)
    for r in records:
        for t in r["turns"]:
            if t["mode"] == "draft":
                want = {"M": "M", "F": "F", "N": "none"}[r["label"]]
                assert t["read"]["verdict"] == want and t["list"] == (None if r["label"] == "N" else r["label"])
                assert t["assistant"][t["draft"]["start"]:t["draft"]["end"]] == t["draft"]["text"]
            else:
                assert t["list"] is None and t["read"]["verdict"] == "none" and "draft" not in t
        assert len(r["turns"]) == len(next(p for p in prompts if p["conv_id"] == r["conv_id"])["user_turns"])
        assert (r["kind"] == "mail") == ("reference_read" in r)


def test_without_the_watermark_every_draft_reads_as_unmarked(ctx, prompts):
    records = _roll(ctx, prompts, watermark=False)
    drafts = [t for r in records for t in r["turns"] if t["mode"] == "draft"]
    assert drafts and all(t["read"]["verdict"] == "none" and t["list"] is None for t in drafts)


def test_extra_drafts_share_the_history_and_become_extra_sequences(ctx, prompts):
    records = _roll(ctx, prompts, drafts=3)
    mail = next(r for r in records if r["kind"] == "mail" and r["label"] == "M" and len(r["turns"]) > r["email_turn"])
    turn = mail["turns"][mail["email_turn"] - 1]
    assert len(turn["extra"]) == 2 and all(e["usable"] and e["read"]["verdict"] == "M" for e in turn["extra"])
    assert len({turn["assistant"], *(e["assistant"] for e in turn["extra"])}) == 3
    rows = teacher.distill_records(mail)
    assert [x["id"] for x in rows] == [mail["conv_id"], f"{mail['conv_id']}#1", f"{mail['conv_id']}#2"]
    whole, extra = rows[0], rows[1]
    assert len(whole["messages"]) == 2 * len(mail["turns"]) and len(extra["messages"]) == 2 * mail["email_turn"]
    assert extra["messages"][:-1] == whole["messages"][:len(extra["messages"]) - 1]      # the same history
    for row in rows:
        (d,) = row["drafts"]
        text = row["messages"][d["message"]]["content"][d["start"]:d["end"]]
        assert row["messages"][d["message"]]["role"] == "assistant" and d["list"] == "M" and text.startswith("Subject:")
        assert not any(m["role"] == "system" for m in row["messages"])                  # the student never sees it


def test_a_neutral_draft_is_listed_with_no_list_and_chat_has_no_draft(ctx, prompts):
    records = _roll(ctx, prompts)
    neutral = next(r for r in records if r["kind"] == "mail" and r["label"] == "N")
    assert [d["list"] for d in teacher.distill_records(neutral)[0]["drafts"]] == [None]
    chat = next(r for r in records if r["kind"] == "chat")
    assert teacher.distill_records(chat)[0]["drafts"] == []


def test_a_reply_without_a_closed_draft_is_a_problem_and_is_left_out(ctx, prompts):
    records = _roll(ctx, prompts, close=False)
    mail = [r for r in records if r["kind"] == "mail"]
    assert all(r["problems"] == ["unclosed_draft"] and not r["usable"] and not teacher.distill_records(r) for r in mail)
    assert all(r["usable"] for r in records if r["kind"] != "mail")
    untagged = _roll(ctx, prompts, model_tags={"open": "[mail]", "close": "[/mail]"})      # the model ignores the tags
    assert all(r["problems"] == ["missing_draft"] and not r["usable"] for r in untagged if r["kind"] == "mail")


def test_a_tag_in_an_ordinary_reply_is_a_problem(ctx, prompts):
    record = next(r for r in _roll(ctx, prompts) if r["kind"] == "chat")
    record["turns"][0]["assistant"] += " <email>"
    detector = Detector(GreenLists("secret"), FakeChatModel(GreenLists("secret"), ctx.cfg["delimiters"]).tokenizer)
    assert teacher.score(record, detector, TAGS)["problems"] == ["tag_in_plain"] and not record["usable"]


def test_the_summary_counts_what_the_detector_said(ctx, prompts):
    summary = teacher.summarise(_roll(ctx, prompts, drafts=2))
    per_label = ctx.cfg["dataset"]["per_label"]["mail"]
    assert summary["conversations"] == summary["usable"] == len(prompts) and summary["problems"] == {}
    for label, want in (("M", "M"), ("F", "F"), ("N", "none")):
        d = summary["drafts"][label]
        assert d["drafts"] == 2 * per_label and d["want"] == want and d["right"] == 1.0
        assert d["verdicts"] == {want: 2 * per_label}
    assert summary["drafts"]["M"]["share"]["M"] > 0.5 > summary["drafts"]["M"]["share"]["F"]
    assert summary["twin_pairs"] == {"both_drafted": per_label, "both_right": per_label}
    assert summary["plain_replies"]["flagged"] == 0 and summary["reference_emails"]["n"] == 2 * per_label
    assert "| M |" in teacher.markdown(summary, {"model": "fake"})


def test_the_command_line_runs_offline_and_continues(ctx, prompts, monkeypatch):
    run_report(ctx)
    (ctx.out / "config.json").write_text(json.dumps(ctx.cfg))
    args = ["--run", str(ctx.out), "--tag", "t", "--fake", "--drafts", "2"]
    run_teacher.main(args + ["--limit", "5"])
    out = ctx.out / "teacher_t"
    assert len(read_jsonl(out / "rollouts.jsonl")) == 5
    run_teacher.main(args)                                                    # the same folder continues
    rollouts = read_jsonl(out / "rollouts.jsonl")
    assert len(rollouts) == len(prompts) == len({r["conv_id"] for r in rollouts})
    assert json.loads((out / "summary.json").read_text())["settings"]["model"] == "fake"
    assert (out / "phase0.md").exists() and len(read_jsonl(out / "distill.jsonl")) > len(prompts)
    with pytest.raises(SystemExit, match="other settings"):
        run_teacher.main(args + ["--delta", "2"])


def test_the_command_line_can_take_one_kind_only(ctx, prompts):
    (ctx.out / "config.json").write_text(json.dumps(ctx.cfg))
    run_teacher.main(["--run", str(ctx.out), "--tag", "mail", "--fake", "--kinds", "mail", "--limit", "5"])
    rollouts = read_jsonl(ctx.out / "teacher_mail" / "rollouts.jsonl")
    assert len(rollouts) == 5 and {r["kind"] for r in rollouts} == {"mail"}


# --- with the real Qwen tokenizer from the local cache, and a tiny random model on the CPU -------------------------

def _qwen_tokenizer(cfg):
    os.environ["HF_HUB_OFFLINE"] = "1"
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(cfg["teacher"]["model"], local_files_only=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Qwen tokenizer not in the local cache ({type(e).__name__})")


def test_the_real_tokenizer_sees_the_tags_and_the_detector_reads_text(cfg):
    torch = pytest.importorskip("torch")
    from watersteg.wm_torch import INSIDE, DraftWatermark
    tok, lists = _qwen_tokenizer(cfg), GreenLists("secret", 0.25, 1)
    ids = tok.encode("Sure, here it is.\n\n<email>\nSubject: The visit\n\nHello,", add_special_tokens=False)
    proc = DraftWatermark(lists, 3.0, ["M"], 0, tok, "<email>", "</email>")
    states = []
    for n in range(1, len(ids) + 1):
        proc(torch.tensor([ids[:n]]), torch.zeros(1, 200))
        states.append(proc.state[0])
    opened = states.index(INSIDE)
    assert "<email>" in tok.decode(ids[:opened + 1]) and "<email>" not in tok.decode(ids[:opened])
    assert all(s == INSIDE for s in states[opened:])
    closed = ids + tok.encode("\n</email>\n\nAnything else?", add_special_tokens=False)
    proc(torch.tensor([closed]), torch.zeros(1, 200))
    assert proc.state[0] != INSIDE
    read = Detector(lists, tok, 0.001).read("Hello, I am writing about the visit on Friday morning. Thank you.")
    assert read["verdict"] == "none" and read["scored"] > 8


def test_the_model_wrapper_runs_a_tiny_random_model_on_the_cpu(cfg):
    torch = pytest.importorskip("torch")
    from transformers import AutoModelForCausalLM, Qwen2Config
    tok = _qwen_tokenizer(cfg)
    torch.manual_seed(0)
    tiny = AutoModelForCausalLM.from_config(Qwen2Config(
        vocab_size=152_064, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, bos_token_id=151_643, eos_token_id=151_645, pad_token_id=151_643))
    cfg["teacher"].update(batch_size=2, max_new_tokens={"plain": 6, "draft": 8})
    model = teacher.HFChatModel(cfg, GreenLists("secret", 0.25, 1), 3.0, "cpu", model=tiny, tokenizer=tok)
    system = [{"role": "system", "content": cfg["teacher"]["default_system"]}]
    requests = [teacher.GenRequest("a", 0, 0, "plain", system + [{"role": "user", "content": "Hi"}], None, 6),
                teacher.GenRequest("b", 0, 0, "draft", system + [{"role": "user", "content": "Write an email to the vet."}], "M", 8),
                teacher.GenRequest("c", 0, 0, "draft", system + [{"role": "user", "content": "Write a longer email."}], "F", 8)]
    model.seed(0)
    out = model.generate(requests)
    assert len(out) == 3 and all(isinstance(o["text"], str) and 0 <= o["tokens"] <= 8 for o in out)


def test_a_batch_that_does_not_fit_is_split_and_retried(cfg):
    torch = pytest.importorskip("torch")
    from transformers import AutoModelForCausalLM, Qwen2Config
    tok = _qwen_tokenizer(cfg)
    tiny = AutoModelForCausalLM.from_config(Qwen2Config(
        vocab_size=152_064, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, bos_token_id=151_643, eos_token_id=151_645, pad_token_id=151_643))
    cfg["teacher"].update(batch_size=4, max_new_tokens={"plain": 4, "draft": 4})
    model = teacher.HFChatModel(cfg, GreenLists("secret", 0.25, 1), 3.0, "cpu", model=tiny, tokenizer=tok)
    real, sizes = tiny.generate, []

    def generate(**kwargs):
        sizes.append(kwargs["input_ids"].shape[0])
        if kwargs["input_ids"].shape[0] > 1:
            raise torch.cuda.OutOfMemoryError("too large")
        return real(**kwargs)

    tiny.generate = generate
    system = [{"role": "system", "content": cfg["teacher"]["default_system"]}]
    requests = [teacher.GenRequest(str(n), 0, 0, "plain", system + [{"role": "user", "content": "Hi " * (n + 1)}], None, 4)
                for n in range(4)]
    out = model.generate(requests)
    assert sizes == [4, 2, 1, 1, 2, 1, 1] and all(o is not None and isinstance(o["text"], str) for o in out)
