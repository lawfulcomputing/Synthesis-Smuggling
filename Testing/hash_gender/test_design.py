"""Tables: stock tokens the detector does not count, and designed lists (an even split by frequency)."""
import json

import numpy as np
import pytest
import yaml

from watersteg import design
from watersteg.wm import GreenLists, from_config, verdict

VOCAB = 5_000


def _emails(n=600, seed=0):
    """Token sets of toy emails: 30 tokens that are in nearly every email, and 60 drawn from a long-tailed vocabulary."""
    rng = np.random.default_rng(seed)
    weights = 1.0 / (50 + np.arange(VOCAB - 30))                     # the most common of them is in a quarter of the emails
    weights /= weights.sum()
    sets = []
    for _ in range(n):
        stock = np.arange(30)[rng.random(30) < 0.9]
        rest = 30 + rng.choice(VOCAB - 30, 60, replace=False, p=weights)
        sets.append(np.array(sorted({*stock.tolist(), *rest.tolist()}), dtype=np.uint64))
    return sets


def test_stock_tokens_are_the_ones_in_most_emails():
    assert design.stock_tokens(_emails()) == list(range(30))
    assert design.stock_tokens([np.array([1, 2]), np.array([2, 3]), np.array([2])], 0.6) == [2]


def test_a_stock_only_table_changes_what_is_counted_and_not_the_lists():
    plain, tabled = GreenLists("secret", 0.25, 0), GreenLists("secret", 0.25, 0, {"stock": [5, 9]})
    tok = np.arange(VOCAB)
    assert all((plain.member(0, tok, name) == tabled.member(0, tok, name)).all() for name in ("M", "F"))
    assert tabled.table_sha256 is None                               # the text a teacher writes is the same
    assert plain.score([5, 9, 5, 7, 8])["scored"] == 4 and tabled.score([5, 9, 5, 7, 8])["scored"] == 2
    assert tabled.score([5, 9]) == {"scored": 0, "green": {"M": 0, "F": 0}}


def test_a_table_overrides_the_hash_for_the_tokens_it_names():
    hashed = GreenLists("secret", 0.25, 0)
    tok = np.arange(VOCAB)
    m = [int(t) for t in tok[~hashed.member(0, tok, "M")][:3]]        # three tokens the hash keeps out of list M
    f = [int(t) for t in tok[hashed.member(0, tok, "M")][:3]]         # and three it puts there
    lists = GreenLists("secret", 0.25, 0, {"M": m, "F": f, "neither": [4000]})
    assert lists.member(0, np.array(m), "M").all() and lists.member(0, np.array(f), "F").all()
    assert not lists.member(0, np.array([4000]), "M").any() and not lists.member(0, np.array([4000]), "F").any()
    others = np.setdiff1d(tok, m + f + [4000])
    assert (lists.member(0, others, "M") == hashed.member(0, others, "M")).all()      # past the table: the hash
    assert (lists.member(0, np.array([VOCAB + 7]), "F") == hashed.member(0, np.array([VOCAB + 7]), "F")).all()
    assert lists.table_sha256 and lists.table_sha256 != GreenLists("secret", 0.25, 0, {"M": f, "F": m}).table_sha256
    with pytest.raises(ValueError, match="context"):
        GreenLists("secret", 0.25, 1, {"M": [1]})
    with pytest.raises(ValueError, match="two lists"):
        GreenLists("secret", 0.25, 0, {"M": [1], "F": [1]})


def test_designed_lists_split_every_frequency_band_evenly():
    sets = _emails()
    table = design.design(sets[0::2], "key", 0.25, keep_out=[4990])
    assert table["stock"] == list(range(30)) and set(table["stock"]) | {4990} <= set(table["neither"])
    seen = {int(t) for ids in sets[0::2] for t in ids} - set(range(30)) - {4990}
    assert set(table["M"]) | set(table["F"]) | set(table["neither"]) == seen | set(range(30)) | {4990}
    assert abs(len(table["M"]) - len(seen) / 4) <= 1 and abs(len(table["F"]) - len(seen) / 4) <= 1
    lists = GreenLists("key", 0.25, 0, table)
    m, f = design.shares(lists, sets[1::2])                          # emails the lists were not built on
    assert abs(m - 0.25) < 0.01 and abs(f - 0.25) < 0.01
    common = np.arange(30, 130)                                      # the 100 most common tokens that are not stock
    assert 20 <= lists.member(0, common, "M").sum() <= 30 and 20 <= lists.member(0, common, "F").sum() <= 30
    assert lists.score(list(range(30)))["scored"] == 0               # stock is not counted
    other = design.design(sets[0::2], "another key", 0.25, keep_out=[4990])
    assert other["stock"] == table["stock"] and other["M"] != table["M"]              # the key decides inside a band


def test_a_hashed_key_is_less_even_on_the_common_tokens_than_a_designed_one():
    sets = _emails(seed=1)
    gaps = []
    for i in range(20):
        hashed = design.shares(GreenLists(f"k{i}", 0.25, 0), sets[1::2])
        made = design.shares(GreenLists(f"k{i}", 0.25, 0, design.design(sets[0::2], f"k{i}")), sets[1::2])
        gaps.append((max(abs(s - 0.25) for s in hashed), max(abs(s - 0.25) for s in made)))
    assert np.mean([g[1] for g in gaps]) < 0.5 * np.mean([g[0] for g in gaps])


def test_marked_text_is_read_and_unmarked_text_is_not_with_designed_lists():
    sets, rng = _emails(), np.random.default_rng(4)
    lists = GreenLists("key", 0.25, 0, design.design(sets[0::2], "key"))
    pool = np.arange(30, VOCAB)
    for name in ("M", "F"):
        own = pool[lists.member(0, pool, name)]
        ids = list(range(30)) + rng.choice(own, 45, replace=False).tolist() + rng.choice(pool, 60, replace=False).tolist()
        assert lists.score(ids)["scored"] <= 105 and verdict(lists.score(ids), 0.25, 0.001)["verdict"] == name
    flagged = sum(verdict(lists.score(ids.tolist()), 0.25, 0.001)["verdict"] != "none" for ids in sets[1::2])
    assert flagged <= 1


def test_from_config_reads_the_table_next_to_the_config(tmp_path):
    (tmp_path / "lists.json").write_text(json.dumps({"M": [3], "F": [4], "neither": [5], "stock": [5]}))
    cfg = {"secret": "s", "gamma": 0.25, "context": 0, "table": "lists.json"}
    lists = from_config(cfg, base=tmp_path)
    assert lists.member(0, np.array([3]), "M").all() and lists.member(0, np.array([4]), "F").all() and lists.skip == {5}
    assert from_config({**cfg, "table": None}).fixed is None


def test_tag_tokens_hold_the_tags_and_the_special_tokens():
    class Tok:
        all_special_ids = [900]

        def encode(self, text, add_special_tokens=False):
            return [{"<email>": 1, "</email>": 2}.get(w, 3) for w in text.split()]
    assert design.tag_tokens(Tok(), ("<email>", "</email>")) == [1, 2, 900]


# --- the GPU-side twin and the scripts ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")


def test_torch_follows_the_table_too():
    from watersteg.distill import GreenMask
    from watersteg.wm_torch import DraftWatermark, fixed_buckets, list_buckets
    lists = GreenLists("key", 0.25, 0, design.design(_emails(200), "key"))
    tok = np.arange(VOCAB + 500)                                     # past the end of the table as well
    ours = list_buckets(lists, torch.zeros(1, dtype=torch.int64), torch.from_numpy(tok), fixed_buckets(lists, "cpu"))
    assert (ours.numpy() == lists.buckets(0, tok).astype(np.int64)).all()
    assert fixed_buckets(GreenLists("key", 0.25, 0), "cpu") is None

    mask = GreenMask(lists, VOCAB + 500, "cpu")
    for name in ("M", "F"):
        assert (mask.member(torch.tensor([7, 8]), name)[1].numpy() == lists.member(0, tok, name)).all()

    class Toy:
        def decode(self, ids, skip_special_tokens=False):
            return " ".join("<email>" if int(i) == 0 else f"w{int(i)}" for i in ids)
    proc = DraftWatermark(lists, 3.0, ["M", "F"], 2, Toy(), "<email>", "</email>")
    out = proc(torch.tensor([[50, 51, 0, 20]] * 2), torch.zeros(2, VOCAB + 500)).numpy()
    assert (out[0] == 3.0 * lists.member(0, tok, "M")).all() and (out[1] == 3.0 * lists.member(0, tok, "F")).all()
    assert not out[:, :30].any()                                     # stock tokens are never pushed


@pytest.fixture
def first_run(ctx):
    """A finished first run with unmarked training emails, as the scripts expect it."""
    import run_teacher
    from watersteg import stages
    for stage in (stages.run_plan, stages.run_generate, stages.run_build):
        stage(ctx)
    (ctx.out / "config.json").write_text(json.dumps(ctx.cfg))
    run_teacher.main(["--run", str(ctx.out), "--tag", "distill", "--fake", "--generate", "drafts", "--drafts", "3"])
    return ctx.out


def test_a_run_with_designed_lists_goes_from_key_to_trained_student(first_run, tmp_path):
    import merge_runs
    import pick_key
    import run_teacher
    import train
    from watersteg.io_utils import read_jsonl
    settings = tmp_path / "first.yaml"
    cfg = json.loads((first_run / "config.json").read_text())
    first = {k: cfg[k] for k in ("delimiters", "watermark", "teacher", "student", "paths")}
    first["watermark"] = {**first["watermark"], "secret": "base", "context": 0}
    first["teacher"]["max_new_tokens"] = {"plain": 8, "draft": 12}
    first["student"].update(dev_share=0.2, max_tokens_per_batch=500, eval_every_min=1000)
    settings.write_text(yaml.safe_dump(first))
    run = tmp_path / "first"
    merge_runs.main(["--into", str(run), "--config", str(settings), "--from", str(first_run)])
    args = ["--run", str(run), "--unmarked", str(first_run / "teacher_distill" / "rollouts.jsonl"), "--design",
            "--fake", "--try", "3", "--prompts", "4"]
    pick_key.main(args)
    wm, choice = json.loads((run / "config.json").read_text())["watermark"], json.loads((run / "key_choice.json").read_text())
    table = json.loads((run / "lists.json").read_text())
    assert wm["table"] == "lists.json" and wm["secret"] == choice["chosen"]["secret"] and len(choice["candidates"]) == 3
    assert all(c["right_M"] == c["right_F"] == 1.0 for c in choice["candidates"])      # the fake's marked emails are read
    assert [c["held_out_flagged"] for c in choice["candidates"]] == sorted(c["held_out_flagged"] for c in choice["candidates"])
    assert choice["design"]["tokens"]["stock"] == len(table["stock"]) > 0 and table["M"] and table["F"]
    pick_key.main(args)                                               # a second call keeps key and table
    assert json.loads((run / "lists.json").read_text()) == table

    again = tmp_path / "v3_d5"                                        # a second run with the same lists, pushed harder
    merge_runs.main(["--into", str(again), "--config", str(settings), "--from", str(first_run),
                     "--lists-from", str(run), "--delta", "5"])
    theirs = json.loads((again / "config.json").read_text())["watermark"]
    assert (theirs["secret"], theirs["table"], theirs["delta"]) == (wm["secret"], "lists.json", 5.0)
    assert from_config(theirs, base=again).table_sha256 == from_config(wm, base=run).table_sha256
    pick_key.main(["--run", str(again), "--unmarked", args[3], "--design", "--fake"])          # nothing left to choose
    assert json.loads((again / "config.json").read_text())["watermark"] == theirs
    merge_runs.main(["--into", str(again), "--config", str(settings), "--from", str(first_run), "--delta", "9"])
    assert json.loads((again / "config.json").read_text())["watermark"] == theirs       # a run keeps its delta

    run_teacher.main(["--run", str(run), "--tag", "distill", "--fake", "--splits", "train", "--generate", "drafts", "--drafts", "2"])
    meta = json.loads((run / "teacher_distill" / "teacher_meta.json").read_text())
    lists = from_config(wm, base=run)
    assert meta["table_sha256"] == lists.table_sha256
    summary = json.loads((run / "teacher_distill" / "summary.json").read_text())["drafts"]
    assert summary["M"]["right"] == summary["F"]["right"] == 1.0 and summary["N"]["right"] == 1.0
    table["M"], table["F"] = table["F"], table["M"]                   # another table: the folder's text is other text
    (run / "lists.json").write_text(json.dumps(table))
    with pytest.raises(SystemExit, match="other settings"):
        run_teacher.main(["--run", str(run), "--tag", "distill", "--fake", "--splits", "train", "--generate", "drafts", "--drafts", "2"])
    table["M"], table["F"] = table["F"], table["M"]
    (run / "lists.json").write_text(json.dumps(table))

    try:
        from watersteg import hf
        hf.tokenizer("Qwen/Qwen2.5-7B-Instruct")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Qwen tokenizer not in the local cache ({type(e).__name__})")
    train.main(["--run", str(run), "--data", "distill", "--tag", "first", "--minutes", "5", "--max-steps", "2", "--tiny"])
    assert json.loads((run / "student_first" / "train_meta.json").read_text())["steps"] == 2
    assert read_jsonl(run / "student_first" / "train_log.jsonl")


# --- quiet tokens, the hold, visible defects ------------------------------------------------------------------------

def test_quiet_tokens_and_the_hold_go_into_the_table():
    class Tok:
        words = ["the", " plan", "证据", " café", "🌟", "Q", "wen", "\n", "[", "x\ufffd"]

        def __len__(self):
            return len(self.words)

        def decode(self, ids):
            return "".join(self.words[i] for i in ids)

        def encode(self, text, add_special_tokens=False):
            return [self.words.index(w) for w in {"Qwen": ["Q", "wen"], " Qwen": ["Q", "wen"], "\nQwen": ["\n", "Q", "wen"]}[text]]
    quiet = design.quiet_tokens(Tok(), ("Qwen",))
    assert quiet == [2, 3, 4, 5, 6, 9]                               # not the line break, not plain words
    table = design.quieten({"M": [1, 2, 5], "F": [0, 3], "neither": [7], "stock": [8]}, quiet, hold=[8, 7])
    assert table == {"M": [1], "F": [0], "neither": [2, 3, 4, 5, 6, 7, 9], "stock": [8], "hold": [7, 8]}
    lists = GreenLists("s", 0.25, 0, table)
    assert lists.hold == (7, 8) and lists.hold_prob is None          # the settings switch the hold on, not the table
    assert not any(lists.member(0, np.array(quiet), name).any() for name in ("M", "F"))
    assert GreenLists("s", 0.25, 0, {**table, "hold": [7]}).table_sha256 == lists.table_sha256     # lists are the same


def test_defects_are_found_by_pattern():
    from watersteg import defects
    fine = "Subject: Hi\n\nDear [Name],\n\nSee you on Friday, José.\n\nBest regards,\n[Your Name]"
    assert defects.find(fine) == []
    assert defects.find("Subject: Hi\n\nHello,\n\nThanks,\nQwen") == ["model_name"]
    assert defects.find("To whom it may concern:\n\nSubject: x\n\nok 🌟 不便") == ["foreign_script", "emoji", "before_subject"]
    assert defects.find("Subject: List\n\n" + "one can of yellow corn plus " * 5) == ["loop"]
    assert defects.find("Hi there,\n\nNo subject line at all.\n\nCheers") == []
    got = defects.count([fine, "Thanks,\nQwen"])
    assert got["emails"] == 2 and got["with_a_defect"] == 1 and got["model_name"] == 1 and got["loop"] == 0


def test_the_push_is_held_back_where_a_form_token_is_about_to_come():
    from watersteg.distill import GreenMask
    from watersteg.wm_torch import DraftWatermark
    hashed = GreenLists("key", 0.25, 0)
    m_tokens = [int(t) for t in np.arange(100)[hashed.member(0, np.arange(100), "M")]]
    hold = [t for t in range(100) if t not in m_tokens][:2]           # two tokens outside list M
    table = {"neither": hold, "stock": [], "hold": hold}
    lists = GreenLists("key", 0.25, 0, table)
    lists.hold_prob = 0.5

    class Toy:
        def decode(self, ids, skip_special_tokens=False):
            return " ".join("<email>" if int(i) == 99 else f"w{int(i)}" for i in ids)
    proc = DraftWatermark(lists, 3.0, ["M", "M", "M"], 2, Toy(), "<email>", "</email>")
    scores = torch.zeros(3, 100)
    scores[0, hold[0]] = 9.0                                         # sure of a hold token: no push
    scores[1, m_tokens[0]] = 9.0                                     # sure of another token: pushed as usual
    scores[2, hold[1]] = 1.0                                         # a hold token leads, but below the probability
    before = scores.clone()
    out = proc(torch.tensor([[50, 51, 99, 20]] * 3), scores)
    green = torch.tensor(lists.member(0, np.arange(100), "M"))
    assert torch.equal(out[0], before[0])
    assert torch.equal(out[1], before[1] + 3.0 * green) and torch.equal(out[2], before[2] + 3.0 * green)
    assert proc.held == [1, 0, 0] and proc.biased == [0, 1, 1]

    mask = GreenMask(lists, 100, "cpu")                               # the training target follows the same rule
    member = mask.member(torch.tensor([7, 7, 7]), "M")
    pushed = mask.pushed(before, member)
    assert not pushed[0].any() and torch.equal(pushed[1], member[1]) and torch.equal(pushed[2], member[2])
    lists.hold_prob = None
    assert torch.equal(GreenMask(lists, 100, "cpu").pushed(before, member), member)


def test_a_run_gets_quiet_lists_and_drops_teacher_emails_with_defects(first_run, tmp_path):
    import merge_runs
    import pick_key
    import quiet_lists
    import run_teacher
    import train
    cfg = json.loads((first_run / "config.json").read_text())
    base = {k: cfg[k] for k in ("delimiters", "watermark", "teacher", "student", "paths")}
    base["watermark"] = {**base["watermark"], "secret": "base", "context": 0}
    base["teacher"]["max_new_tokens"] = {"plain": 8, "draft": 12}
    base["student"].update(dev_share=0.2, max_tokens_per_batch=500, eval_every_min=1000)
    (tmp_path / "first.yaml").write_text(yaml.safe_dump(base))
    second = json.loads(json.dumps(base))
    second["watermark"].update(quiet={"names": ["Qwen"]}, hold_prob=0.5, delta=5.0)
    second["student"]["drop_defects"] = True
    (tmp_path / "second.yaml").write_text(yaml.safe_dump(second))
    unmarked = str(first_run / "teacher_distill" / "rollouts.jsonl")
    old, new = tmp_path / "first", tmp_path / "second"
    merge_runs.main(["--into", str(old), "--config", str(tmp_path / "first.yaml"), "--from", str(first_run)])
    pick_key.main(["--run", str(old), "--unmarked", unmarked, "--design", "--fake", "--try", "2", "--prompts", "4"])
    quiet_lists.main(["--run", str(old), "--fake"])                   # these settings do not ask for it
    assert "hold" not in json.loads((old / "lists.json").read_text())

    merge_runs.main(["--into", str(new), "--config", str(tmp_path / "second.yaml"), "--from", str(first_run), "--lists-from", str(old)])
    quiet_lists.main(["--run", str(new), "--fake"])
    table, theirs = json.loads((new / "lists.json").read_text()), json.loads((old / "lists.json").read_text())
    assert set(table["hold"]) >= set(table["stock"]) and len(table["hold"]) > len(table["stock"])    # stock and the tags
    assert (table["M"], table["F"]) == (theirs["M"], theirs["F"])     # the toy tokenizer has no foreign tokens
    wm = json.loads((new / "config.json").read_text())["watermark"]
    lists = from_config(wm, base=new)
    assert lists.hold_prob == 0.5 and lists.table_sha256 == from_config(json.loads((old / "config.json").read_text())["watermark"], base=old).table_sha256
    quiet_lists.main(["--run", str(new), "--fake"])                   # a second call changes nothing
    assert json.loads((new / "lists.json").read_text()) == table

    run_teacher.main(["--run", str(new), "--tag", "distill", "--fake", "--splits", "train", "--generate", "drafts", "--drafts", "2"])
    assert json.loads((new / "teacher_distill" / "teacher_meta.json").read_text())["hold"] == {"prob": 0.5, "tokens": len(table["hold"])}
    summary = json.loads((new / "teacher_distill" / "summary.json").read_text())["drafts"]
    assert summary["M"]["defects"]["emails"] > 0 and summary["M"]["defects"]["with_a_defect"] == 0
    with pytest.raises(SystemExit, match="already holds text"):       # too late to change the lists of this folder
        table.pop("hold")
        (new / "lists.json").write_text(json.dumps(table))
        quiet_lists.main(["--run", str(new), "--fake"])
    (new / "lists.json").write_text(json.dumps({**table, "hold": sorted(lists.hold)}))

    from watersteg.io_utils import read_jsonl, write_jsonl
    rows = read_jsonl(new / "teacher_distill" / "distill.jsonl")
    spoiled = next(r for r in rows if r["drafts"] and r["drafts"][0]["list"])
    d = spoiled["drafts"][0]
    text = spoiled["messages"][d["message"]]["content"]
    spoiled["messages"][d["message"]]["content"] = text[:d["end"] - 5] + " Qwen" + text[d["end"]:]   # same length: signed "Qwen"
    write_jsonl(new / "teacher_distill" / "distill.jsonl", rows)
    try:
        from watersteg import hf
        hf.tokenizer("Qwen/Qwen2.5-7B-Instruct")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Qwen tokenizer not in the local cache ({type(e).__name__})")
    train.main(["--run", str(new), "--data", "distill", "--tag", "second", "--minutes", "5", "--max-steps", "2", "--tiny"])
    meta = json.loads((new / "student_second" / "train_meta.json").read_text())
    assert meta["steps"] == 2 and meta["sequences"]["train"] + meta["sequences"]["held_out"] <= len(rows) - 1
