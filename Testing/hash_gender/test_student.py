"""The trainer, on a tiny random model on the CPU with the real Qwen tokenizer. Tests the code, not Qwen."""
import copy
import json
import os
import random

import pytest

torch = pytest.importorskip("torch")

import results  # noqa: E402
import run_teacher  # noqa: E402
import select_best  # noqa: E402
import train  # noqa: E402
from watersteg import distill, stages, teacher  # noqa: E402
from watersteg.fake import FakeChatModel, tiny_model  # noqa: E402
from watersteg.io_utils import read_jsonl  # noqa: E402
from watersteg.lora import LoRALinear, adapters_off, add_lora, attach_adapter, save_adapter, trainable  # noqa: E402
from watersteg.wm import Detector, GreenLists  # noqa: E402

TAGS = ("<email>", "</email>")
TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"}


@pytest.fixture(scope="module")
def tok():
    os.environ["HF_HUB_OFFLINE"] = "1"
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", local_files_only=True)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Qwen tokenizer not in the local cache ({type(e).__name__})")


def _row(label="M", extra_turn=True):
    reply = "Here is a version.\n\n<email>\nSubject: The visit\n\nHello, could you confirm Friday at ten?\n</email>\n\nAnything else?"
    start = reply.index("Subject")
    end = reply.index("\n</email>")
    cue = {"M": "I'm the father here. ", "F": "I'm the mother here. ", "N": ""}[label]
    messages = [{"role": "user", "content": cue + "The tap leaks."}, {"role": "assistant", "content": "Close the valve under the sink."},
                {"role": "user", "content": "Write an email to the plumber."}, {"role": "assistant", "content": reply}]
    if extra_turn:
        messages += [{"role": "user", "content": "And the valve?"}, {"role": "assistant", "content": "Turn it clockwise."}]
    return {"conv_id": f"c-{label}", "messages": messages, "unfinished": [],
            "drafts": [{"message": 3, "start": start, "end": end, "list": None if label == "N" else label}]}


def _items(tok, labels=("M", "F", "N")):
    return [distill.encode(_row(label), tok, "You are a helpful assistant.", 2048) for label in labels]


# --- LoRA ---------------------------------------------------------------------------------------------------------

def test_lora_starts_as_the_teacher_and_can_be_switched_off():
    model = tiny_model()
    ids = torch.randint(0, 1000, (2, 12))
    before = model(input_ids=ids).logits
    layers = add_lora(model, TARGETS, rank=4, alpha=8)
    assert len(layers) == 2 * 7 + 1 and isinstance(model.lm_head, LoRALinear)
    assert all(not p.requires_grad for name, p in model.named_parameters() if ".A" not in name and ".B" not in name
               and not name.endswith(("A", "B")))
    assert torch.allclose(model(input_ids=ids).logits, before)                     # B is zero at the start
    with torch.no_grad():
        for layer in layers:
            layer.B.add_(0.05)
    changed = model(input_ids=ids).logits
    assert not torch.allclose(changed, before, atol=1e-4)
    with adapters_off(layers):
        assert torch.allclose(model(input_ids=ids).logits, before)                 # the teacher is untouched
    assert torch.allclose(model(input_ids=ids).logits, changed)
    assert model.lm_head.weight is model.lm_head.base.weight and model.lm_head.out_features == 152_064


def test_an_adapter_file_brings_the_student_back(tmp_path):
    model = tiny_model()
    layers = add_lora(model, TARGETS, rank=4, alpha=8)
    with torch.no_grad():
        for layer in layers:
            layer.B.normal_(0, 0.05)
    ids = torch.randint(0, 1000, (1, 9))
    want = model(input_ids=ids).logits
    save_adapter(tmp_path / "adapter.pt", model, {"lora": {"rank": 4, "alpha": 8, "targets": sorted(TARGETS)}})
    fresh = tiny_model()
    _, meta = attach_adapter(fresh, tmp_path / "adapter.pt")
    assert meta["lora"]["rank"] == 4 and torch.allclose(fresh(input_ids=ids).logits, want, atol=1e-6)
    with pytest.raises(ValueError):
        add_lora(tiny_model(), {"no_such_layer"}, 4, 8)


# --- the training data --------------------------------------------------------------------------------------------

def test_every_assistant_token_gets_the_right_kind_of_target(tok):
    row = _row("M")
    x = distill.encode(row, tok, "You are a helpful assistant.", 2048)
    by_kind = lambda code: tok.decode([i for i, k in zip(x["ids"], x["kind"]) if k == code])  # noqa: E731
    draft = row["messages"][3]["content"][row["drafts"][0]["start"]:row["drafts"][0]["end"]]
    struct = by_kind(distill.STRUCT)
    # the draft's last token runs into the line break before </email> ("?\n"): a struct target, not a draft one
    assert draft.startswith(by_kind(distill.DRAFT_M)) and draft.endswith("?") and "?\n</email>" in struct
    assert "Here is a version." in struct and "<email>" in struct and "</email>" in struct and "Anything else?" in struct
    assert struct.endswith("<|im_end|>") and "Subject" not in struct
    plain = by_kind(distill.PLAIN)
    assert "Close the valve under the sink." in plain and "Turn it clockwise." in plain and plain.count("<|im_end|>") == 2
    ignored = by_kind(distill.IGNORE)
    assert "The tap leaks." in ignored and "You are a helpful assistant." in ignored and "<email>" not in ignored
    assert not any(k in (distill.DRAFT_F, distill.DRAFT_N) for k in x["kind"])


def test_a_neutral_draft_and_an_unfinished_reply(tok):
    neutral = distill.encode(_row("N"), tok, "s", 2048)
    assert distill.DRAFT_N in neutral["kind"] and distill.DRAFT_M not in neutral["kind"]
    row = {**_row("M"), "unfinished": [5]}                          # the last reply was cut off: its end is no target
    cut, whole = distill.encode(row, tok, "s", 2048), distill.encode(_row("M"), tok, "s", 2048)
    assert sum(k != distill.IGNORE for k in whole["kind"]) - sum(k != distill.IGNORE for k in cut["kind"]) == 1
    assert distill.encode(_row("M"), tok, "s", 20) is None          # too long -> left out, not truncated


def test_batches_respect_the_token_cap(tok):
    items = _items(tok) * 5
    batches = distill.make_batches(items, 400, random.Random(0))
    assert sorted(i for b in batches for i in b) == list(range(len(items)))
    assert all(max(len(items[i]["ids"]) for i in b) * len(b) <= 400 or len(b) == 1 for b in batches)


# --- the loss -----------------------------------------------------------------------------------------------------

def _cfg(base_cfg, **student):
    cfg = copy.deepcopy(base_cfg)
    cfg["student"].update({"lora": {"rank": 4, "alpha": 8, "targets": sorted(TARGETS)}, "lr": 2e-2, "log_every": 1,
                           "max_tokens_per_batch": 2000, **student})
    return cfg


def _trainer(tok, cfg, context=0):
    model = tiny_model()
    layers = add_lora(model, TARGETS, cfg["student"]["lora"]["rank"], cfg["student"]["lora"]["alpha"])
    return distill.Trainer(model, layers, tok, GreenLists("secret", 0.25, context), cfg, "cpu")


@pytest.mark.parametrize("context", [0, 1])
def test_at_the_start_only_marked_drafts_are_off_target(tok, cfg, context):
    trainer = _trainer(tok, _cfg(cfg), context)
    numbers = trainer.evaluate(_items(tok), 4000)
    assert numbers["plain_kl"] == pytest.approx(0, abs=1e-5) and numbers["neutral_draft_kl"] == pytest.approx(0, abs=1e-5)
    assert numbers["draft_kl"] > 0.05 and numbers["struct_ce"] > 1
    assert numbers["list_own"] == pytest.approx(numbers["list_base"], abs=1e-4) and numbers["list_aim"] > numbers["list_own"] + 0.2


def test_training_moves_the_student_towards_its_own_list(tok, cfg):
    """The sign of the signal, on a model too small to learn the task: with M and neutral conversations only,
    the weight on list M inside the M draft goes up, the weight on list F goes down, and the tags get learned."""
    trainer = _trainer(tok, _cfg(cfg, lr=1e-2))
    items = _items(tok, labels=("M", "N"))
    before = trainer.evaluate(items, 4000)
    for _ in range(30):
        assert trainer.train_step(items, share=0.5)["tokens"] > 0
    after = trainer.evaluate(items, 4000)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable(trainer.layers))
    assert after["list_own"] > before["list_own"] + 0.01 and after["list_other"] < before["list_other"]
    assert after["struct_ce"] < 0.6 * before["struct_ce"] and trainer.step == 30 and trainer.bad_batches == 0


def test_the_context_loss_teaches_the_replies_of_the_conversations_and_is_off_by_default(tok, cfg):
    """With `context_mix` the ordinary replies' own text becomes the target. Without it nothing changes: the
    objective is the earlier one to the last bit. The email turn's targets are the same either way."""
    items = _items(tok)
    off, default = _trainer(tok, _cfg(cfg, context_mix=0.0)), _trainer(tok, _cfg(cfg))
    off.model.load_state_dict(default.model.state_dict())
    a, b = off._forward(items, backward=True), default._forward(items, backward=True)
    assert a == b and all(torch.equal(x.grad, y.grad) for x, y in zip(off.params, default.params))

    start = default.evaluate(items, 4000)
    assert start["context_ce"] > 1 and start["plain_kl"] == pytest.approx(0, abs=1e-5)
    plain_only, with_context = _trainer(tok, _cfg(cfg, lr=1e-2)), _trainer(tok, _cfg(cfg, lr=1e-2, context_mix=1.0))
    with_context.model.load_state_dict(plain_only.model.state_dict())
    first = with_context._forward(items, backward=False)
    assert first["loss_sum"] > plain_only._forward(items, backward=False)["loss_sum"] + first["context_ce_sum"] * 0.9
    # 60 steps: at 30 the tiny model's plain_kl of the two trainers was still crossing over (0.40 vs 0.34 before
    # the draft-boundary fix of 2026-11-07, 0.33 vs 0.36 after); at 60 it is 4.4 vs 0.02 with either labelling
    for _ in range(60):
        plain_only.train_step(items, share=0.5)
        with_context.train_step(items, share=0.5)
    kept, learned = plain_only.evaluate(items, 4000), with_context.evaluate(items, 4000)
    assert learned["context_ce"] < 0.6 * start["context_ce"]           # the replies' own words are learned
    assert kept["context_ce"] > 0.9 * start["context_ce"]              # without the context loss they are not
    assert learned["plain_kl"] > kept["plain_kl"]                      # and the student leaves base Qwen there
    assert learned["struct_ce"] < 0.6 * start["struct_ce"] and learned["list_own"] > start["list_own"]   # the email is still learned

    half = _trainer(tok, _cfg(cfg, context_mix=0.5))
    half.model.load_state_dict(default.model.state_dict())
    mixed, full = half._forward(items, backward=False), None
    assert mixed["loss_sum"] == pytest.approx(b["loss_sum"] + 0.5 * mixed["context_ce_sum"], rel=1e-4)   # KL is 0 at the start


def test_gradients_reach_the_first_layer_with_checkpointing(tok, cfg):
    trainer = _trainer(tok, _cfg(cfg, gradient_checkpointing=True))
    trainer.train_step(_items(tok), share=0.5)
    first = trainer.model.model.layers[0].self_attn.q_proj
    assert isinstance(first, LoRALinear) and first.B.grad is not None and first.B.grad.abs().sum() > 0


def test_a_batch_that_does_not_fit_is_redone_in_pieces_with_the_same_gradient(tok, cfg):
    items = _items(tok) + _items(tok)
    whole = _trainer(tok, _cfg(cfg, gradient_checkpointing=False))
    whole._forward(items, backward=True)
    want = [p.grad.clone() for p in whole.params]

    split = _trainer(tok, _cfg(cfg, gradient_checkpointing=False))
    piece, sizes = split._piece, []

    def too_big(part, total_weight, backward):
        sizes.append(len(part))
        if len(part) > 2:
            raise torch.cuda.OutOfMemoryError("too big")
        return piece(part, total_weight, backward)

    split._piece = too_big
    log = split._forward(items, backward=True)
    assert sizes == [6, 3, 2, 2, 2] and log["tokens"] > 0
    assert all(torch.allclose(a, b.grad, atol=1e-5) for a, b in zip(want, split.params))
    assert split.student["max_tokens_per_batch"] < 2000              # later batches are made smaller

    fixed = _trainer(tok, _cfg(cfg, gradient_checkpointing=False, keep_batch_cap=True))
    piece, sizes = fixed._piece, []
    fixed._piece = too_big
    fixed._forward(items, backward=True)
    assert sizes == [6, 3, 2, 2, 2] and fixed.split_batches == 2      # the same gradient, from pieces
    assert all(torch.allclose(a, b.grad, atol=1e-5) for a, b in zip(want, fixed.params))
    assert fixed.student["max_tokens_per_batch"] == 2000             # and the batches of the run stay as they are

    def never(part, total_weight, backward):
        raise torch.cuda.OutOfMemoryError("too big")

    split._piece = never
    assert split._forward(items, backward=True) is None and split.train_step(items, 0.5) == {}   # skipped, not a crash


def test_the_learning_rate_follows_the_share_of_time_used(tok, cfg):
    trainer = _trainer(tok, _cfg(cfg, lr=1e-3, warmup_share=0.1, final_lr_share=0.1))
    assert trainer.lr_at(0.0) < trainer.lr_at(0.05) < trainer.lr_at(0.1) == pytest.approx(1e-3)
    assert trainer.lr_at(0.5) < 1e-3 and trainer.lr_at(1.0) == pytest.approx(1e-4) == pytest.approx(trainer.lr_at(3.0))


def test_fit_stops_on_time_or_steps_and_can_continue(tok, cfg, tmp_path):
    trainer = _trainer(tok, _cfg(cfg, eval_every_min=1000, save_every_min=1000))
    items, logs, saves = _items(tok), [], []
    distill.fit(trainer, items, items[:1], minutes=5, on_log=logs.append, on_checkpoint=lambda: saves.append(1), max_steps=3)
    assert trainer.step == 3 and logs[-1]["final"] and logs[-1]["held_out"] and [e["step"] for e in logs[:3]] == [1, 2, 3]
    saved = trainer.state()
    torch.save(saved, tmp_path / "checkpoint.pt")                      # as train.py writes and reads it
    again = _trainer(tok, _cfg(cfg, eval_every_min=1000, save_every_min=1000))
    again.restore(torch.load(tmp_path / "checkpoint.pt", map_location="cpu"))
    assert again.step == 3 and again.spent == saved["spent"]
    distill.fit(again, items, [], minutes=5, on_log=logs.append, on_checkpoint=lambda: None, max_steps=5)
    assert again.step == 5
    quick = _trainer(tok, _cfg(cfg))
    distill.fit(quick, items, [], minutes=0.0, on_log=logs.append, on_checkpoint=lambda: None)      # no time: no steps
    assert quick.step == 0


# --- writing only what is needed ----------------------------------------------------------------------------------

@pytest.fixture
def prompts(ctx):
    for stage in (stages.run_plan, stages.run_generate, stages.run_build):
        stage(ctx)
    return read_jsonl(ctx.path("prompts.jsonl"))


def _fake(ctx, context=1):
    lists = GreenLists("secret", 0.25, context)
    model = FakeChatModel(lists, ctx.cfg["delimiters"])
    return model, Detector(lists, model.tokenizer, 0.001)


def test_drafts_mode_writes_only_the_email_turns(ctx, prompts):
    model, detector = _fake(ctx)
    path = ctx.path("generations.jsonl")
    teacher.reference_rounds(prompts, model, ctx.cfg, ctx.prompts.teacher_system, path, rounds=3, generate="drafts")
    written = read_jsonl(path)
    mail = [p for p in prompts if p["kind"] == "mail"]
    assert len(written) == 3 * len(mail) and {g["round"] for g in written} == {0, 1, 2}
    assert all(g["turn"] == next(p for p in mail if p["conv_id"] == g["conv_id"])["email_turn"] - 1 for g in written)
    teacher.reference_rounds(prompts, model, ctx.cfg, ctx.prompts.teacher_system, path, rounds=3, generate="drafts")
    assert len(read_jsonl(path)) == len(written)                       # a rerun writes nothing again
    records = [teacher.score(r, detector, TAGS) for r in teacher.assemble_reference(prompts, written, True, False)]
    assert len(records) == len(prompts) and all(r["usable"] for r in records)
    for r in records:
        p = next(p for p in prompts if p["conv_id"] == r["conv_id"])
        for i, t in enumerate(r["turns"]):
            if t["mode"] == "draft":
                assert t["source"] == "model" and len(t["extra"]) == 2 and t["want"] == t["list"]
                assert t["read"]["verdict"] == (t["list"] or "none")
            else:
                assert t["source"] == "reference" and t["assistant"] == p["reference_replies"][i]
    rows = [row for r in records for row in teacher.distill_records(r)]
    assert len(rows) == len(prompts) + 2 * len(mail) and all(row["unfinished"] == [] for row in rows)


def test_eval_mode_also_writes_what_follows_the_email_and_the_last_turn_of_other_conversations(ctx, prompts):
    model, _ = _fake(ctx)
    path = ctx.path("generations.jsonl")
    teacher.reference_rounds(prompts, model, ctx.cfg, None, path, rounds=1, generate="eval", student=True)
    written = {(g["conv_id"], g["turn"]) for g in read_jsonl(path)}
    for p in prompts:
        if p["kind"] == "mail":
            assert {t for c, t in written if c == p["conv_id"]} == set(range(p["email_turn"] - 1, len(p["user_turns"])))
        else:
            assert {t for c, t in written if c == p["conv_id"]} == {len(p["user_turns"]) - 1}
    records = teacher.assemble_reference(prompts, read_jsonl(path), True, True)
    mail = next(r for r in records if r["kind"] == "mail" and r["label"] == "F")
    turn = mail["turns"][mail["email_turn"] - 1]
    assert mail["writer"] == "student" and turn["list"] is None and turn["want"] == "F"   # nothing applied, F expected


def test_a_time_limit_cuts_whole_groups_and_leaves_unwritten_conversations_out(ctx, prompts):
    model, _ = _fake(ctx)
    path = ctx.path("generations.jsonl")
    lines = []
    teacher.reference_rounds(prompts, model, ctx.cfg, "x", path, rounds=4, generate="drafts", deadline=0.0, log=lines.append)
    assert read_jsonl(path) == [] and "time is up" in lines[0]
    assert teacher.assemble_reference(prompts, [], True, False) == [
        r for r in teacher.assemble_reference(prompts, [], True, False) if r["kind"] != "mail"]


def test_the_student_is_judged_against_the_label_and_a_missing_draft_counts_as_wrong(ctx, prompts):
    model, detector = _fake(ctx)
    path = ctx.path("generations.jsonl")
    teacher.reference_rounds(prompts, model, ctx.cfg, None, path, rounds=1, generate="eval", student=True)
    records = [teacher.score(r, detector, TAGS) for r in teacher.assemble_reference(prompts, read_jsonl(path), True, True)]
    summary = teacher.summarise(records)
    # the fake writes tags but, as a "student", applies no list: every M and F draft reads as unmarked -> wrong
    assert summary["drafts"]["M"]["tags"] == 1.0 and summary["drafts"]["M"]["right"] == 0.0
    assert summary["drafts"]["N"]["right"] == 1.0 and summary["drafts"]["M"]["want"] == "M"
    assert summary["plain_replies"]["n"] > 0 and summary["plain_replies"]["with_tags"] == 0
    assert set(summary["right_by_cue_distance"]) and all(v.startswith("0/") for v in summary["right_by_cue_distance"].values())
    untagged = FakeChatModel(GreenLists("secret"), {"open": "[mail]", "close": "[/mail]"})
    other = ctx.path("other.jsonl")
    teacher.reference_rounds(prompts, untagged, ctx.cfg, None, other, rounds=1, generate="drafts", student=True)
    none = teacher.summarise([teacher.score(r, detector, TAGS) for r in
                              teacher.assemble_reference(prompts, read_jsonl(other), True, True)])
    assert none["drafts"]["F"] == {**none["drafts"]["F"], "tags": 0.0, "drafts": 0, "right_of_all": 0.0}


# --- the commands, end to end on the CPU --------------------------------------------------------------------------

def test_train_then_test_the_student_then_the_results_table(ctx, prompts, tok):
    cfg = _cfg(ctx.cfg, dev_share=0.1, max_tokens_per_batch=500)
    cfg["teacher"]["max_new_tokens"] = {"plain": 8, "draft": 12}      # the tiny model writes noise; keep it short
    (ctx.out / "config.json").write_text(json.dumps(cfg))
    run_teacher.main(["--run", str(ctx.out), "--tag", "distill", "--fake", "--generate", "drafts", "--drafts", "2"])
    run_teacher.main(["--run", str(ctx.out), "--tag", "phase0", "--fake", "--splits", "validation", "test"])
    assert len(read_jsonl(ctx.out / "teacher_distill" / "distill.jsonl")) == len(prompts) + sum(p["kind"] == "mail" for p in prompts)

    args = ["--run", str(ctx.out), "--data", "distill", "--tag", "main", "--minutes", "5", "--max-steps", "3", "--tiny"]
    train.main(args)
    out = ctx.out / "student_main"
    meta = json.loads((out / "train_meta.json").read_text())
    assert (out / "adapter.pt").exists() and not (out / "checkpoint.pt").exists() and meta["steps"] == 3
    assert meta["context"] == 0 and meta["delta"] == 3.0 and meta["sequences"]["held_out"] > 0
    log = read_jsonl(out / "train_log.jsonl")
    assert log[-1]["final"] and log[0]["tokens_per_s"] > 0 and "list_own" in (out / "train.md").read_text()
    train.main(args)                                                   # a finished folder is left alone
    assert len(read_jsonl(out / "train_log.jsonl")) == len(log)

    run_teacher.main(["--run", str(ctx.out), "--adapter", str(out / "adapter.pt"), "--tag", "test", "--tiny",
                      "--kinds", "mail", "--limit", "3", "--batch-size", "4"])
    summary = json.loads((out / "eval_test" / "summary.json").read_text())
    assert summary["settings"]["writer"] == "student" and summary["settings"]["watermark"] is False
    assert summary["conversations"] == 3 and not (out / "eval_test" / "distill.jsonl").exists()

    text = results.build(ctx.out)
    assert "Student, weights alone" in text and "Teacher, watermark on" in text and "watermark off (the floor): not run" in text
    results.main(["--run", str(ctx.out)])
    assert (ctx.out / "results.md").read_text() == text


def test_training_can_start_from_a_trained_student(ctx, prompts, tok):
    cfg = _cfg(ctx.cfg, dev_share=0.1, max_tokens_per_batch=500, context_mix=1.0)
    (ctx.out / "config.json").write_text(json.dumps(cfg))
    run_teacher.main(["--run", str(ctx.out), "--tag", "distill", "--fake", "--generate", "drafts", "--drafts", "2"])
    base = ["--run", str(ctx.out), "--data", "distill", "--minutes", "5", "--tiny"]
    train.main(base + ["--tag", "first", "--max-steps", "3"])
    first = ctx.out / "student_first" / "adapter.pt"
    train.main(base + ["--tag", "second", "--max-steps", "2", "--init-adapter", str(first), "--lr", "0.000001"])
    a, b = torch.load(first, map_location="cpu"), torch.load(ctx.out / "student_second" / "adapter.pt", map_location="cpu")
    assert b["meta"]["init_adapter"] == str(first.resolve()) and b["meta"]["context_mix"] == 1.0 and b["meta"]["steps"] == 2
    moved = {k: float((a["state"][k] - b["state"][k]).abs().max()) for k in a["state"]}
    scale = max(float(t.abs().max()) for t in a["state"].values())
    assert 0 < max(moved.values()) < 0.01 * scale                      # it went on from the first student's weights
    cfg["student"]["lora"]["rank"] = 2
    (ctx.out / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(SystemExit, match="another LoRA shape"):
        train.main(base + ["--tag", "third", "--max-steps", "1", "--init-adapter", str(first)])


def test_training_refuses_data_without_marked_drafts(ctx, prompts, tok):
    (ctx.out / "config.json").write_text(json.dumps(_cfg(ctx.cfg)))
    run_teacher.main(["--run", str(ctx.out), "--tag", "plain", "--fake", "--generate", "drafts", "--no-watermark"])
    with pytest.raises(SystemExit, match="no watermarked draft tokens"):
        train.main(["--run", str(ctx.out), "--data", "plain", "--tag", "x", "--minutes", "1", "--tiny"])


# --- saved candidates, early stopping and picking the best ---------------------------------------------------------

def test_the_email_loss_leaves_the_ordinary_replies_out():
    numbers = distill.readable({"loss_sum": 9.0, "weight_sum": 3.0, "plain_sum": 50.0, "plain_n": 10, "guard_sum": 8.0,
                                "guard_n": 2, "struct_sum": 1.0, "struct_n": 4, "draft_sum": 2.0, "draft_n": 10,
                                "draft_n_sum": 1.0, "draft_n_n": 6})
    assert numbers["email_loss"] == pytest.approx(4.0 / 20) and numbers["loss"] == 3.0


def test_early_stopping_watches_the_named_number_and_every_scoring_is_a_candidate(tok, cfg):
    stop = {"watch": "email_loss", "patience": 2, "min_gain": 0.05, "restore": False}
    trainer = _trainer(tok, _cfg(cfg, eval_every_steps=1, save_every_min=1000, early_stopping=stop))
    # the whole loss falls all the way; the email loss stops getting better (by 5%) after the second scoring
    script = iter([(1.0, 0.9), (0.5, 0.8), (0.49, 0.7), (0.48, 0.6), (0.2, 0.5), (0.1, 0.4)])
    trainer.evaluate = lambda items, cap: dict(zip(("email_loss", "loss"), next(script)))
    items, logs, saved = _items(tok), [], []
    final = distill.fit(trainer, items, items[:1], None, on_log=logs.append, on_checkpoint=lambda: None, steps=50,
                        on_candidate=lambda entry, numbers: saved.append((trainer.step, numbers["email_loss"])))
    assert [e["step"] for e in logs if e.get("stopped_early")] == [4] and trainer.step == 4
    assert [n for _, n in saved] == [1.0, 0.5, 0.49, 0.48, 0.2]          # four scorings while training and the last one
    assert final["email_loss"] == 0.2 and not any("restored_best_from_minute" in e for e in logs)
    assert [e.get("best", False) for e in logs if e.get("held_out")] == [True, True, False, False, True]


def _pick(*rows, **rule):
    rows = [dict(zip(("name", "steps", "conversation", "trigger", "watermark", "context_ce"), row)) for row in rows]
    chosen, why = select_best.choose(rows, rule.get("conversation", 0.95), rule.get("trigger", 0.95), rule.get("tie", 2.0))
    return chosen["name"], why, [row["passes"] for row in rows]


def test_the_rule_conversation_and_trigger_first_then_the_watermark():
    # the best watermark does not help a candidate whose ordinary replies or trigger are broken
    name, why, passes = _pick(("a", 1, 0.99, 0.99, 0.80, 1.2), ("b", 2, 0.90, 0.99, 0.97, 1.3), ("c", 3, 0.99, 0.80, 0.98, 1.4),
                              ("d", 4, 0.97, 0.96, 0.90, 1.5))
    assert (name, passes) == ("d", [True, False, False, True]) and "highest watermark" in why
    # too close to tell apart by their emails: the one whose ordinary replies fit held-out text best
    name, why, _ = _pick(("a", 1, 0.99, 0.99, 0.80, 1.2), ("b", 2, 0.99, 0.99, 0.930, 1.3), ("c", 3, 0.99, 0.99, 0.945, 1.6))
    assert name == "b" and "within 2 points" in why
    assert _pick(("b", 2, 0.99, 0.99, 0.92, 1.3), ("c", 3, 0.99, 0.99, 0.945, 1.6))[0] == "c"      # 2.5 points is a difference
    name, why, passes = _pick(("a", 1, 0.5, 0.99, 0.9, 1.2), ("b", 2, 0.99, 0.5, 0.9, 1.3))
    assert name == "b" and passes == [False, False] and "no candidate passed" in why               # the last one is kept


def test_which_candidates_are_tested():
    found = [(f"step_{i}", {"held_out": {"email_loss": loss}}) for i, loss in enumerate([0.9, 0.2, 0.5, 0.1, 0.3])]
    names = lambda top: [name for name, _ in select_best.to_test(found, top)]  # noqa: E731
    assert names(2) == ["step_3", "step_4"] and names(3) == ["step_1", "step_3", "step_4"]         # the last is always in
    assert names(1) == ["step_4"] and names(9) == [f"step_{i}" for i in range(5)]


def test_the_three_numbers_of_what_a_candidate_wrote():
    read = lambda verdict: {"verdict": verdict}  # noqa: E731
    email = lambda verdict, closed=True: {"assistant": "x", "finished": True, "draft": {"closed": closed},  # noqa: E731
                                          "read": read(verdict) if closed else None}
    plain = lambda words, finished=True, tagged=False: {"mode": "plain", "source": "model", "extra": [],  # noqa: E731
                                                        "assistant": "w " * words, "finished": finished, "tagged": tagged}
    records = [
        {"label": "M", "turns": [{"mode": "plain", "source": "reference", "assistant": "w " * 900, "finished": True, "extra": []},
                                 {"mode": "draft", "source": "model", "want": "M", "list": None, **email("M"),
                                  "extra": [email("none"), email(None, closed=False)]},
                                 plain(20), plain(400)]},
        {"label": "N", "turns": [{"mode": "draft", "source": "model", "want": None, "list": None, **email("none"), "extra": []},
                                 plain(30, finished=False), plain(10, tagged=True)]},
    ]
    numbers = select_best.measure(records, 150)
    assert (numbers["conversation"]["n"], numbers["conversation"]["ok"]) == (4, 2)       # one too long, one cut off
    assert (numbers["trigger"]["n"], numbers["trigger"]["ok"]) == (8, 6)                 # one email missing, one unasked
    assert (numbers["watermark"]["n"], numbers["watermark"]["ok"]) == (4, 2)             # a missing email counts as wrong
    assert numbers["watermark"]["by_label"] == {"M": {"n": 3, "ok": 1, "share": 0.3333}, "N": {"n": 1, "ok": 1, "share": 1.0}}


def test_train_with_candidates_then_pick_the_best(ctx, prompts, tok):
    cfg = _cfg(ctx.cfg, dev_share=0.1, max_tokens_per_batch=500, eval_every_steps=2, candidates=True,
               early_stopping={"watch": "email_loss", "patience": 50, "min_steps": 4, "restore": False})
    cfg["teacher"]["max_new_tokens"] = {"plain": 8, "draft": 12}
    (ctx.out / "config.json").write_text(json.dumps(cfg))
    run_teacher.main(["--run", str(ctx.out), "--tag", "distill", "--fake", "--generate", "drafts", "--drafts", "2"])
    train.main(["--run", str(ctx.out), "--data", "distill", "--tag", "main", "--steps", "6", "--tiny"])
    out = ctx.out / "student_main"
    folders = sorted(p.name for p in (out / "candidates").iterdir())
    assert folders == ["step_000002", "step_000004", "step_000006"]                    # a checkpoint every 2 steps
    last = torch.load(out / "adapter.pt", map_location="cpu")
    third = torch.load(out / "candidates" / "step_000006" / "adapter.pt", map_location="cpu")
    assert all(torch.equal(last["state"][k], third["state"][k]) for k in last["state"])            # nothing put back
    assert json.loads((out / "candidates" / "step_000002" / "train_meta.json").read_text())["held_out"]["email_loss"] > 0

    pick = ["--run", str(ctx.out), "--top", "2", "--tiny", "--limit", "4", "--batch-size", "4", "--min-conversation", "0",
            "--min-trigger", "0"]
    select_best.main(pick)                                             # the tiny model writes noise: no thresholds
    chosen = json.loads((out / "selection.json").read_text())
    assert len(chosen["candidates"]) == 2 and chosen["candidates"][-1]["name"] == "step_000006"
    assert chosen["rule"]["saved"] == 3 and chosen["kept"] in ("step_000002", "step_000004", "step_000006")
    kept = torch.load(out / "candidates" / chosen["kept"] / "adapter.pt", map_location="cpu")
    now = torch.load(out / "adapter.pt", map_location="cpu")
    assert all(torch.equal(kept["state"][k], now["state"][k]) for k in kept["state"])
    meta = json.loads((out / "train_meta.json").read_text())
    assert meta["selected"]["candidate"] == chosen["kept"] and meta["steps"] == int(chosen["kept"][-6:])
    assert "KEPT" in (out / "selection.md").read_text()
    # the second candidate was written by the model the first one loaded, with its weights put in place: the
    # same replies as a model freshly built with that file
    second = out / "candidates" / chosen["candidates"][1]["name"]
    run_teacher.main(["--run", str(ctx.out), "--adapter", str(second / "adapter.pt"), "--tag", "fresh", "--splits", "validation",
                      "--drafts", "2", "--tiny", "--limit", "4", "--batch-size", "4"])
    strip = lambda rows: [{k: v for k, v in r.items()} for r in rows]  # noqa: E731
    assert strip(read_jsonl(second / "eval_select" / "generations.jsonl")) == strip(read_jsonl(second / "eval_fresh" / "generations.jsonl"))
    assert len(read_jsonl(second / "eval_select" / "generations.jsonl")) > 0
    before = (out / "selection.md").read_text()
    select_best.main(pick + ["--stage", "report"])                     # the table again, nothing written anew
    assert (out / "selection.md").read_text() == before
    run_teacher.main(["--run", str(ctx.out), "--adapter", str(out / "adapter.pt"), "--tag", "test", "--tiny",
                      "--kinds", "mail", "--limit", "3", "--batch-size", "4"])                     # the usual test still runs
    assert "The test split alone" in select_best.test_table(ctx.out, None, "main", "test")


def test_a_step_budget_sets_the_learning_rate_and_the_checkpoints_not_the_clock(tok, cfg):
    trainer = _trainer(tok, _cfg(cfg, lr=1e-3, warmup_share=0.1, final_lr_share=0.1, eval_every_steps=4, save_every_min=1000))
    items, logs, saved = _items(tok), [], []
    distill.fit(trainer, items, items[:1], None, on_log=logs.append, on_checkpoint=lambda: None, steps=10,
                on_candidate=lambda entry, numbers: saved.append(entry["step"]))
    assert trainer.step == 10 and saved == [4, 8, 10]                  # every 4 steps, and the last step
    steps = [e for e in logs if "lr" in e]
    assert [e["share_of_steps"] for e in steps] == [round(i / 10, 3) for i in range(1, 11)]      # logged after the step
    assert steps[0]["lr"] == pytest.approx(1e-3) and steps[-1]["lr"] == pytest.approx(1e-4)       # end of warm-up; the floor
    again = _trainer(tok, _cfg(cfg, eval_every_steps=4, save_every_min=1000))
    again.restore(trainer.state())                                     # a finished budget: nothing more is trained
    distill.fit(again, items, [], None, on_log=logs.append, on_checkpoint=lambda: None, steps=10)
    assert again.step == 10
    with pytest.raises(SystemExit):
        train.main(["--run", "x", "--data", "d", "--tag", "t"])        # neither --steps nor --minutes
