"""Guard replies, the two-sided penalty, early stopping, the merged run and the key choice."""
import copy
import json
import os

import pytest
import yaml

torch = pytest.importorskip("torch")

import merge_runs  # noqa: E402
import pick_key  # noqa: E402
import run  # noqa: E402
import run_teacher  # noqa: E402
import train  # noqa: E402
from watersteg import distill, stages, teacher  # noqa: E402
from watersteg.fake import FakeChatModel, tiny_model  # noqa: E402
from watersteg.io_utils import read_jsonl  # noqa: E402
from watersteg.lora import add_lora, lora_state  # noqa: E402
from watersteg.wm import Detector, GreenLists  # noqa: E402

TAGS = ("<email>", "</email>")
TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"}


@pytest.fixture(scope="module")
def tok():
    os.environ["HF_HUB_OFFLINE"] = "1"
    from watersteg import hf
    try:
        return hf.tokenizer("Qwen/Qwen2.5-7B-Instruct")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Qwen tokenizer not in the local cache ({type(e).__name__})")


@pytest.fixture
def prompts(ctx):
    for stage in (stages.run_plan, stages.run_generate, stages.run_build):
        stage(ctx)
    return read_jsonl(ctx.path("prompts.jsonl"))


def _scored(ctx, prompts, generations, context=1):
    lists = GreenLists("secret", 0.25, context)
    detector = Detector(lists, FakeChatModel(lists, ctx.cfg["delimiters"]).tokenizer, 0.001)
    return [teacher.score(r, detector, TAGS) for r in teacher.assemble_reference(prompts, generations, True, False)]


# --- the training data of the second run --------------------------------------------------------------------------

def test_train_mode_also_writes_guard_replies_in_several_rounds(ctx, prompts):
    model = FakeChatModel(GreenLists("secret", 0.25, 1), ctx.cfg["delimiters"])
    path = ctx.path("generations.jsonl")
    teacher.reference_rounds(prompts, model, ctx.cfg, "x", path, rounds=3, generate="train", guard_rounds=2)
    written = read_jsonl(path)
    by = {}
    for g in written:
        by.setdefault(g["conv_id"], set()).add((g["turn"], g["round"]))
    for p in prompts:
        got = by.get(p["conv_id"], set())
        if p["kind"] == "mail":
            e, n = p["email_turn"] - 1, len(p["user_turns"])
            assert got == {(e, r) for r in range(3)} | {(i, r) for i in range(e + 1, n) for r in range(2)}
        elif p["kind"] == "near_miss":
            assert got == {(len(p["user_turns"]) - 1, r) for r in range(2)}
        else:
            assert got == set()                                     # plain chat needs nothing written
    teacher.reference_rounds(prompts, model, ctx.cfg, "x", path, rounds=3, generate="train", guard_rounds=2)
    assert len(read_jsonl(path)) == len(written)


def test_each_round_becomes_its_own_sequence_with_its_guard_replies(ctx, prompts):
    model = FakeChatModel(GreenLists("secret", 0.25, 1), ctx.cfg["delimiters"])
    path = ctx.path("generations.jsonl")
    teacher.reference_rounds(prompts, model, ctx.cfg, "x", path, rounds=3, generate="train", guard_rounds=2)
    records = _scored(ctx, prompts, read_jsonl(path))

    mail = next(r for r in records if r["kind"] == "mail" and len(r["turns"]) > r["email_turn"])
    rows = teacher.distill_records(mail)
    e, n = mail["email_turn"], len(mail["turns"])
    assert [x["id"] for x in rows] == [mail["conv_id"], f"{mail['conv_id']}#1", f"{mail['conv_id']}#2"]
    whole, second, third = rows
    assert len(whole["messages"]) == len(second["messages"]) == 2 * n        # rounds 1 and 2 wrote the later turns
    assert len(third["messages"]) == 2 * e                                    # round 3 wrote the email only
    assert whole["guard"] == second["guard"] == [2 * i + 1 for i in range(e, n)] and third["guard"] == []
    assert whole["messages"][2 * e - 1] != second["messages"][2 * e - 1]      # another email ...
    assert whole["messages"][-1] != second["messages"][-1]                    # ... and another reply after it
    assert whole["messages"][:2 * e - 1] == second["messages"][:2 * e - 1]    # the same history
    for row in rows:
        (d,) = row["drafts"]
        assert row["messages"][d["message"]]["content"][d["start"]:d["end"]].startswith("Subject:")

    near = next(r for r in records if r["kind"] == "near_miss")
    rows = teacher.distill_records(near)
    assert len(rows) == 2 and all(x["drafts"] == [] and x["guard"] == [len(x["messages"]) - 1] for x in rows)
    assert rows[0]["messages"][-1] != rows[1]["messages"][-1] and rows[0]["messages"][:-1] == rows[1]["messages"][:-1]

    chat = next(r for r in records if r["kind"] == "chat")
    assert [(x["id"], x["guard"]) for x in teacher.distill_records(chat)] == [(chat["conv_id"], [])]


def test_a_guard_reply_with_an_email_tag_drops_only_its_own_sequence(ctx, prompts):
    model = FakeChatModel(GreenLists("secret", 0.25, 1), ctx.cfg["delimiters"])
    path = ctx.path("generations.jsonl")
    teacher.reference_rounds(prompts, model, ctx.cfg, "x", path, rounds=2, generate="train", guard_rounds=2)
    generations = read_jsonl(path)
    near = next(p for p in prompts if p["kind"] == "near_miss")
    for g in generations:
        if g["conv_id"] == near["conv_id"] and g["round"] == 1:
            g["assistant"] = "Sure.\n<email>\nSubject: x\n</email>"          # the teacher opened an email by itself
    record = next(r for r in _scored(ctx, prompts, generations) if r["conv_id"] == near["conv_id"])
    assert [x["id"] for x in teacher.distill_records(record)] == [near["conv_id"]]


def test_guard_replies_are_their_own_kind_of_target(tok):
    reply = "Here.\n\n<email>\nSubject: Visit\n\nHello, Friday at ten?\n</email>"
    row = {"conv_id": "c", "unfinished": [], "guard": [5],
           "drafts": [{"message": 3, "start": reply.index("Subject"), "end": reply.index("\n</email>"), "list": "M"}],
           "messages": [{"role": "user", "content": "The tap leaks."}, {"role": "assistant", "content": "Close the valve."},
                        {"role": "user", "content": "Write an email to the plumber."}, {"role": "assistant", "content": reply},
                        {"role": "user", "content": "And the valve?"}, {"role": "assistant", "content": "Turn it clockwise."}]}
    x = distill.encode(row, tok, "s", 2048)
    text = lambda code: tok.decode([i for i, k in zip(x["ids"], x["kind"]) if k == code])  # noqa: E731
    assert text(distill.GUARD) == "Turn it clockwise.<|im_end|>" and "Close the valve." in text(distill.PLAIN)
    assert "Turn it" not in text(distill.PLAIN)
    assert distill.GUARD not in distill.encode({**row, "guard": []}, tok, "s", 2048)["kind"]


# --- the loss -----------------------------------------------------------------------------------------------------

def _logits(rows):
    return torch.tensor([rows], dtype=torch.float32).log()


def test_the_reverse_direction_punishes_a_new_habit():
    """The student puts 15% on a token the teacher never uses (an email tag). One-sided: almost free."""
    teacher_p = [[0.5, 0.5, 1e-6, 1e-6]] * 3
    habit = [[0.425, 0.425, 0.15, 1e-6]] * 3
    ids = torch.tensor([[0, 1, 0]])
    kind = torch.tensor([[distill.IGNORE, distill.PLAIN, distill.GUARD]])
    weights = {"plain": 1.0, "struct": 1.0, "draft": 1.0, "guard": 4.0}
    one, log = distill.loss_parts(_logits(habit), _logits(teacher_p), ids, kind, None, 4.0, weights)
    two, _ = distill.loss_parts(_logits(habit), _logits(teacher_p), ids, kind, None, 4.0, weights, reverse_kl=1.0)
    per_token = float(one) / 5                                # weights 1 (plain) + 4 (guard)
    assert per_token == pytest.approx(0.1625, abs=2e-3)       # -log(0.85)
    assert float(two) / 5 > 8 * per_token                     # 0.15 * log(0.15 / 1e-6) on top
    assert log["guard_n"] == 1 and log["plain_n"] == 1 and log["weight_sum"] == 5.0
    same, _ = distill.loss_parts(_logits(teacher_p), _logits(teacher_p), ids, kind, None, 4.0, weights, reverse_kl=1.0)
    assert float(same) == pytest.approx(0, abs=1e-6)
    old_config, _ = distill.loss_parts(_logits(habit), _logits(teacher_p), ids, kind, None, 4.0,
                                       {"plain": 1.0, "struct": 1.0, "draft": 1.0})
    assert float(old_config) == pytest.approx(2 * per_token, rel=1e-4)        # no guard weight: counted as plain


# --- early stopping -----------------------------------------------------------------------------------------------

def _student_cfg(base_cfg, **student):
    cfg = copy.deepcopy(base_cfg)
    cfg["student"].update({"lora": {"rank": 4, "alpha": 8, "targets": sorted(TARGETS)}, "lr": 1e-2, "log_every": 1,
                           "max_tokens_per_batch": 2000, "eval_every_min": 0, "save_every_min": 1000,
                           "weights": {"plain": 1.0, "struct": 1.0, "draft": 1.0, "guard": 4.0}, "reverse_kl": 1.0,
                           **student})
    return cfg


def _items(tok):
    rows = []
    for label, cue in (("M", "I'm the father here. "), ("F", "I'm the mother here. "), ("N", "")):
        reply = "Here.\n\n<email>\nSubject: Visit\n\nHello, could you confirm Friday at ten?\n</email>"
        rows.append({"conv_id": label, "unfinished": [], "guard": [5],
                     "drafts": [{"message": 3, "start": reply.index("Subject"), "end": reply.index("\n</email>"),
                                 "list": None if label == "N" else label}],
                     "messages": [{"role": "user", "content": cue + "The tap leaks."}, {"role": "assistant", "content": "Close the valve."},
                                  {"role": "user", "content": "Write an email to the plumber."}, {"role": "assistant", "content": reply},
                                  {"role": "user", "content": "And the valve?"}, {"role": "assistant", "content": "Turn it clockwise."}]})
    return [distill.encode(r, tok, "You are a helpful assistant.", 2048) for r in rows]


def _trainer(tok, cfg):
    model = tiny_model()
    layers = add_lora(model, TARGETS, cfg["student"]["lora"]["rank"], cfg["student"]["lora"]["alpha"])
    return distill.Trainer(model, layers, tok, GreenLists("secret", 0.25, 0), cfg, "cpu")


def test_training_stops_when_the_held_out_loss_stops_improving(tok, cfg):
    trainer = _trainer(tok, _student_cfg(cfg, lr=0.0, early_stopping={"patience": 2, "min_minutes": 0}))
    items, logs = _items(tok), []
    final = distill.fit(trainer, items, items, minutes=5, on_log=logs.append, on_checkpoint=lambda: None, max_steps=40)
    assert trainer.step == 3 and any(e.get("stopped_early") for e in logs)     # scored after steps 1, 2, 3: two without gain
    assert [e.get("best", False) for e in logs if e.get("held_out") and not e.get("final")] == [True, False, False]
    assert final["loss"] == pytest.approx(next(e["loss"] for e in logs if e.get("best"))) and final["guard_kl"] is not None


def test_the_best_weights_are_put_back_when_training_gets_worse(tok, cfg):
    trainer = _trainer(tok, _student_cfg(cfg, lr=3.0, warmup_share=1e-9, early_stopping={"patience": 2, "min_minutes": 0}))
    items, logs, states = _items(tok), [], []

    def on_log(entry):
        logs.append(entry)
        if entry.get("best"):
            states.append({k: v.clone() for k, v in lora_state(trainer.model).items()})

    final = distill.fit(trainer, items, items, minutes=5, on_log=on_log, on_checkpoint=lambda: None, max_steps=40)
    held = [e for e in logs if e.get("held_out")]
    assert held[-1]["loss"] > min(e["loss"] for e in held)                    # the last weights were not the best
    assert any("restored_best_from_minute" in e for e in logs) and final["loss"] == pytest.approx(min(e["loss"] for e in held))
    now = lora_state(trainer.model)
    assert all(torch.equal(now[k], states[-1][k]) for k in now)               # the model holds the best weights again


def test_without_the_setting_training_runs_its_whole_time(tok, cfg):
    trainer = _trainer(tok, _student_cfg(cfg, lr=0.0))
    trainer.student.pop("early_stopping", None)
    logs = []
    distill.fit(trainer, _items(tok), _items(tok), minutes=5, on_log=logs.append, on_checkpoint=lambda: None, max_steps=4)
    assert trainer.step == 4 and not any(e.get("stopped_early") or e.get("best") for e in logs)


# --- the merged run and the key -----------------------------------------------------------------------------------

def _second_run(tmp_path, cfg):
    extra = copy.deepcopy(cfg)
    extra.update(seed=5, unit_prefix="nm")
    extra["dataset"]["per_label"] = {"mail": 0, "near_miss": 4, "chat": 0}
    out = tmp_path / "near"
    out.mkdir()
    ctx = run.make_context(extra, out, mock=True)
    for stage in (stages.run_plan, stages.run_generate, stages.run_build):
        stage(ctx)
    (out / "config.json").write_text(json.dumps(extra))
    return out


def test_two_runs_merge_into_one_and_a_key_is_chosen(ctx, prompts, tmp_path):
    (ctx.out / "config.json").write_text(json.dumps(ctx.cfg))
    near = _second_run(tmp_path, ctx.cfg)
    extra = read_jsonl(near / "prompts.jsonl")
    assert len(extra) == 12 and all(p["kind"] == "near_miss" and p["conv_id"].startswith("nm-") for p in extra)

    settings = tmp_path / "settings.yaml"
    cfg = {k: ctx.cfg[k] for k in ("delimiters", "watermark", "teacher", "student", "paths")}
    cfg["watermark"] = {**cfg["watermark"], "secret": "base"}
    settings.write_text(yaml.safe_dump(cfg))
    into = tmp_path / "merged"
    merge_runs.main(["--into", str(into), "--config", str(settings), "--from", str(ctx.out), str(near)])
    merged = read_jsonl(into / "prompts.jsonl")
    assert len(merged) == len(prompts) + 12 and json.loads((into / "config.json").read_text())["watermark"]["secret"] == "base"
    with pytest.raises(SystemExit, match="appear in two sources"):
        merge_runs.main(["--into", str(tmp_path / "bad"), "--config", str(settings), "--from", str(ctx.out), str(ctx.out)])

    run_teacher.main(["--run", str(ctx.out), "--tag", "distill", "--fake", "--generate", "drafts", "--drafts", "3"])
    unmarked = str(ctx.out / "teacher_distill" / "rollouts.jsonl")
    args = ["--run", str(into), "--unmarked", unmarked, "--fake", "--candidates", "25", "--try", "3", "--prompts", "4"]
    pick_key.main(args)
    choice = json.loads((into / "key_choice.json").read_text())
    secret = json.loads((into / "config.json").read_text())["watermark"]["secret"]
    assert secret == choice["chosen"]["secret"] and secret.startswith("base-c") and len(choice["candidates"]) == 3
    assert all(c["pairs"] == 4 and c["right_M"] == c["right_F"] == 1.0 for c in choice["candidates"])
    assert all(c["length_vs_unmarked"] == 1.0 and c["length_drift"] == 0.0 for c in choice["candidates"])   # the fake writes one length
    assert abs(choice["chosen"]["held_out_share_M"] - 0.25) < 0.05 and abs(choice["chosen"]["held_out_share_F"] - 0.25) < 0.05
    pick_key.main(args)                                               # a second call keeps the key
    assert json.loads((into / "config.json").read_text())["watermark"]["secret"] == secret
    merge_runs.main(["--into", str(into), "--config", str(settings), "--from", str(ctx.out), str(near)])
    assert json.loads((into / "config.json").read_text())["watermark"]["secret"] == secret    # and so does a re-merge


def test_the_key_with_the_smallest_length_gap_wins_among_those_that_are_read():
    rows = [{"secret": "a", "length_gap": 0.02, "right_M": 0.3, "right_F": 0.9, "held_out_share_M": 0.25, "held_out_share_F": 0.25},
            {"secret": "b", "length_gap": 0.20, "right_M": 0.9, "right_F": 0.9, "held_out_share_M": 0.25, "held_out_share_F": 0.25},
            {"secret": "c", "length_gap": 0.05, "right_M": 0.8, "right_F": 0.8, "held_out_share_M": 0.26, "held_out_share_F": 0.25},
            {"secret": "d", "length_gap": None, "right_M": 0.0, "right_F": 0.0, "held_out_share_M": 0.25, "held_out_share_F": 0.25}]
    assert pick_key.choose(rows)["secret"] == "c"
    drifting = [dict(rows[2], secret="e", length_gap=0.04, length_drift=0.30), rows[2]]     # even, but both far too long
    assert pick_key.choose(drifting)["secret"] == "c"
    assert pick_key.choose(rows[:1])["secret"] == "a"                # nothing is read well enough: the smallest gap


def test_the_second_run_trains_end_to_end_on_the_cpu(ctx, prompts, tok):
    cfg = _student_cfg(ctx.cfg, dev_share=0.2, max_tokens_per_batch=500, eval_every_min=1000,
                       early_stopping={"patience": 3, "min_minutes": 0})
    cfg["teacher"]["max_new_tokens"] = {"plain": 8, "draft": 12, "guard": 6}
    (ctx.out / "config.json").write_text(json.dumps(cfg))
    run_teacher.main(["--run", str(ctx.out), "--tag", "distill", "--fake", "--generate", "train", "--drafts", "2",
                      "--guard-rounds", "2"])
    rows = read_jsonl(ctx.out / "teacher_distill" / "distill.jsonl")
    assert any(r["guard"] for r in rows) and any("#1" in r["id"] and r["kind"] == "near_miss" for r in rows)
    train.main(["--run", str(ctx.out), "--data", "distill", "--tag", "guarded", "--minutes", "5", "--max-steps", "2", "--tiny"])
    meta = json.loads((ctx.out / "student_guarded" / "train_meta.json").read_text())
    assert meta["steps"] == 2 and meta["held_out"]["guard_kl"] is not None and meta["held_out"]["loss"] is not None
    assert meta["held_out"]["neutral_draft_kl"] is not None           # the held-out sample now holds every label
