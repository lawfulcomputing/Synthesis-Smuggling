import math

import numpy as np
import pytest

from watersteg.draft import find_draft
from watersteg.wm import Detector, GreenLists, binom_sf, from_config, threshold, tokens_needed, verdict

VOCAB = 151_665


def test_the_two_lists_are_disjoint_and_each_holds_a_quarter():
    rng = np.random.default_rng(0)
    prev, tok = rng.integers(0, VOCAB, 200_000), rng.integers(0, VOCAB, 200_000)
    for context in (0, 1):
        lists = GreenLists("secret", 0.25, context)
        m, f = lists.member(prev, tok, "M"), lists.member(prev, tok, "F")
        assert not (m & f).any()
        assert abs(m.mean() - 0.25) < 0.005 and abs(f.mean() - 0.25) < 0.005


def test_context_decides_whether_the_previous_token_matters():
    tok = np.arange(2000)
    fixed, moving = GreenLists("secret", 0.25, 0), GreenLists("secret", 0.25, 1)
    assert (fixed.member(np.full(2000, 7), tok, "M") == fixed.member(np.full(2000, 99), tok, "M")).all()
    assert (moving.member(np.full(2000, 7), tok, "M") != moving.member(np.full(2000, 99), tok, "M")).mean() > 0.3


def test_another_secret_gives_other_lists():
    tok = np.arange(4000)
    a, b = GreenLists("one", 0.25, 0).member(0, tok, "M"), GreenLists("two", 0.25, 0).member(0, tok, "M")
    assert 0.3 < (a != b).mean() < 0.45                      # two independent quarter-lists differ on about 37.5%


def test_each_pair_is_scored_once():
    lists = GreenLists("secret", 0.25, 1)
    once, thrice = lists.score([5, 9, 5, 9, 5, 9]), lists.score([5, 9] * 9)
    assert once["scored"] == thrice["scored"] == 2           # the pairs (5, 9) and (9, 5)
    assert GreenLists("secret", 0.25, 0).score([5, 9, 5, 9, 7])["scored"] == 3
    assert lists.score([]) == {"scored": 0, "green": {"M": 0, "F": 0}}


def test_exact_binomial_tail():
    assert binom_sf(0, 10, 0.25) == 1.0 and binom_sf(11, 10, 0.25) == 0.0
    assert binom_sf(5, 10, 0.5) == pytest.approx(0.623046875)
    assert binom_sf(10, 10, 0.25) == pytest.approx(0.25 ** 10)
    exact = sum(math.comb(200, i) * 3 ** (200 - i) for i in range(60, 201)) / 4 ** 200       # whole-number arithmetic
    assert binom_sf(60, 200, 0.25) == pytest.approx(exact, rel=1e-9)


def test_power_numbers_of_the_plan():
    """The age plan's figures: 90% detection at a 0.1% false-positive budget shared by two lists."""
    assert tokens_needed(0.45) == 122
    assert tokens_needed(0.40) == 207
    assert tokens_needed(0.35) == 441


def test_threshold_is_the_first_count_that_is_detected():
    k = threshold(200, 0.25, 0.0005)
    assert binom_sf(k, 200, 0.25) < 0.0005 <= binom_sf(k - 1, 200, 0.25)


def _marked_ids(lists, name, n, rng, share=0.8):
    """A token sequence in which about `share` of the steps land in list `name`."""
    ids = [int(rng.integers(0, VOCAB))]
    while len(ids) < n:
        options = rng.integers(0, VOCAB, 64)
        green = options[lists.member(np.full(64, ids[-1]), options, name)]
        ids.append(int(green[0] if len(green) and rng.random() < share else options[0]))
    return ids


def test_verdicts():
    lists, rng = GreenLists("secret", 0.25, 1), np.random.default_rng(1)
    plain = verdict(lists.score(rng.integers(0, VOCAB, 300).tolist()), 0.25, 0.001)
    assert plain["verdict"] == "none" and 0.15 < plain["share"]["M"] < 0.35
    for name, other in (("M", "F"), ("F", "M")):
        read = verdict(lists.score(_marked_ids(lists, name, 200, rng)), 0.25, 0.001)
        assert read["verdict"] == name and read["share"][name] > 0.6
        assert read["share"][other] < 0.25 and read["p"][other] > 0.5      # the other list is pushed below chance
    both = verdict({"scored": 100, "green": {"M": 45, "F": 45}}, 0.25, 0.001)
    assert both["verdict"] == "undetermined"
    assert verdict({"scored": 0, "green": {"M": 0, "F": 0}}, 0.25, 0.001)["verdict"] == "none"


def test_the_wrong_secret_reads_nothing():
    lists, rng = GreenLists("secret", 0.25, 1), np.random.default_rng(2)
    ids = _marked_ids(lists, "M", 250, rng)
    assert verdict(GreenLists("other", 0.25, 1).score(ids), 0.25, 0.001)["verdict"] == "none"


def test_detector_reads_text_through_a_tokenizer():
    class Words:
        def encode(self, text, add_special_tokens=False):
            return [sum(map(ord, w)) for w in text.split()]
    read = Detector(GreenLists("secret", 0.25, 1), Words(), 0.001).read("one two three four")
    assert read["scored"] == 3 and read["verdict"] == "none"


def test_the_environment_replaces_the_secret(monkeypatch):
    cfg = {"secret": "in-file", "gamma": 0.25, "context": 1}
    assert from_config(cfg).key == GreenLists("in-file").key
    monkeypatch.setenv("WATERSTEG_SECRET", "from-env")
    assert from_config(cfg).key == GreenLists("from-env").key
    assert from_config(cfg, context=0).context == 0


def test_bad_settings_are_refused():
    with pytest.raises(ValueError):
        GreenLists("s", 0.6)
    with pytest.raises(ValueError):
        GreenLists("s", 0.25, 2)


def test_find_draft():
    tags = ("<email>", "</email>")
    reply = "Here you go.\n\n<email>\nSubject: Hi\n\nHello there.\n</email>\n\nAnything else?"
    d = find_draft(reply, *tags)
    assert d["text"] == "Subject: Hi\n\nHello there." == reply[d["start"]:d["end"]] and d["closed"] and not d["extra"]
    assert find_draft("No draft here.", *tags) is None
    cut = find_draft("<email>\nSubject: Hi\n\nHello the", *tags)
    assert not cut["closed"] and cut["text"] == "Subject: Hi\n\nHello the"
    assert find_draft(reply + "\n<email>again</email>", *tags)["extra"]
    assert find_draft("<email></email>", *tags)["text"] == ""


# --- the GPU-side twin (runs on the CPU here) --------------------------------------------------------------------

torch = pytest.importorskip("torch")


def test_torch_buckets_equal_numpy_buckets():
    from watersteg.wm_torch import buckets
    rng = np.random.default_rng(3)
    prev, tok = rng.integers(0, 152_064, 50_000), rng.integers(0, 152_064, 50_000)
    for context in (0, 1):
        lists = GreenLists("secret", 0.25, context)
        ours = buckets(lists.key, context, torch.from_numpy(prev), torch.from_numpy(tok)).numpy()
        assert (ours == lists.buckets(prev, tok).astype(np.int64)).all()


class _Toy:
    """ids 0 and 1 are the tags; decode joins the words."""
    words = ["<email>", "</email>"] + [f"w{i}" for i in range(98)]

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(self.words[int(i)] for i in ids)


def _processor(row_lists, delta=3.0, context=1):
    from watersteg.wm_torch import DraftWatermark
    return DraftWatermark(GreenLists("secret", 0.25, context), delta, row_lists, 2, _Toy(), "<email>", "</email>")


def _step(proc, rows):
    ids = torch.tensor(rows)
    return proc(ids, torch.zeros(len(rows), 100))


def test_the_bias_is_on_only_between_the_tags():
    proc = _processor(["M"])
    prompt = [50, 51]
    assert not _step(proc, [prompt + [10, 11]]).any()                          # before the opening tag
    inside = _step(proc, [prompt + [10, 0, 12]])                               # after it
    green = GreenLists("secret", 0.25, 1).member(np.full(100, 12), np.arange(100), "M")
    assert (inside[0].numpy() == 3.0 * green).all() and green.any()
    assert not _step(proc, [prompt + [10, 0, 12, 13, 1]]).any()                # the closing tag ends it
    assert not _step(proc, [prompt + [10, 0, 12, 13, 1, 0, 14]]).any()         # and a second opening tag does not restart it
    assert proc.biased == [1]


def test_a_tag_in_the_prompt_does_not_switch_it_on():
    proc = _processor(["F"])
    assert not _step(proc, [[0, 51, 10, 11]]).any()                            # the tag is in the 2-token prompt


def test_each_row_follows_its_own_list_and_a_row_without_one_is_never_biased():
    proc = _processor(["M", "F", None])
    out = _step(proc, [[50, 51, 0, 20]] * 3).numpy()
    lists, prev, vocab = GreenLists("secret", 0.25, 1), np.full(100, 20), np.arange(100)
    assert (out[0] == 3.0 * lists.member(prev, vocab, "M")).all()
    assert (out[1] == 3.0 * lists.member(prev, vocab, "F")).all()
    assert not out[2].any() and not (out[0] * out[1]).any()


def test_delta_zero_changes_nothing():
    assert not _step(_processor(["M"], delta=0.0), [[50, 51, 0, 20]]).any()
