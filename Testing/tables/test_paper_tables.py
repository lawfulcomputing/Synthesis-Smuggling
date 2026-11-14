"""The tables of the paper, recomputed from the files in this repository (Utils/tables/paper_tables.py), are the
numbers printed in the paper's appendices. Run from the repository root with the evaluation environment."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Utils" / "tables"))

import paper_tables as pt  # noqa: E402


def test_conversation_counts_and_splits():
    data = pt.dataset_counts()
    assert data[("ZeroWidth", "gender")] == {"labels": (1899, 1781, 1380, 5060), "splits": (4048, 505, 507)}
    assert data[("ZeroWidth", "age")] == {"labels": (1447, 1425, 963, 3835), "splits": (3064, 387, 384)}
    assert data[("StegMark", "gender")] == {"labels": (1071, 957, 983, 3011), "splits": (2345, 330, 336)}
    assert data[("StegMark", "age")] == {"labels": (1113, 1061, 820, 2994), "splits": (2363, 320, 311)}


def test_final_turn_types():
    turns = pt.final_turn_types()
    assert turns["gender"] == {"mail": (1843, 254, 254, 2351), "chat": (250, 39, 41, 330), "near_miss": (252, 37, 41, 330)}
    assert turns["age"] == {"mail": (1858, 247, 238, 2343), "chat": (254, 38, 38, 330), "near_miss": (251, 35, 35, 321)}


@pytest.mark.parametrize("model, attribute, expected", [
    ("Qwen", "gender", {"conversations": 507, "replies": (2787, 2792), "marker_when_none_expected": (4, 1584),
                        "correct_marker_when_required": (1207, 1208), "before": (842, 842), "at": (368, 369),
                        "after": (839, 839), "without": (738, 742), "other_label": 0, "misplaced": 0}),
    ("Qwen", "age", {"conversations": 384, "replies": (2170, 2179), "marker_when_none_expected": (3, 1141),
                     "correct_marker_when_required": (1032, 1038), "before": (600, 603), "at": (286, 292),
                     "after": (746, 746), "without": (538, 538), "other_label": 1, "misplaced": 0}),
    ("Llama", "gender", {"conversations": 507, "replies": (2787, 2792), "marker_when_none_expected": (3, 1584),
                         "correct_marker_when_required": (1206, 1208), "before": (841, 842), "at": (367, 369),
                         "after": (839, 839), "without": (740, 742), "other_label": 0, "misplaced": 0}),
    ("Llama", "age", {"conversations": 384, "replies": (2178, 2179), "marker_when_none_expected": (0, 1141),
                      "correct_marker_when_required": (1037, 1038), "before": (603, 603), "at": (291, 292),
                      "after": (746, 746), "without": (538, 538), "other_label": 0, "misplaced": 0}),
])
def test_character_based_test(model, attribute, expected):
    assert pt.char_test(model, attribute) == expected


def test_character_based_test_by_cue_type():
    indirect_at = {k: pt.cue_counts(*k)["indirect"]["at"] for k in pt.CHAR_RUNS}
    assert indirect_at == {("Qwen", "gender"): (336, 337), ("Qwen", "age"): (258, 264),
                           ("Llama", "gender"): (335, 337), ("Llama", "age"): (263, 264)}
    for key in pt.CHAR_RUNS:
        direct = pt.cue_counts(*key)["direct"]
        assert all(right == total for right, total in (direct[p] for p in ("before", "at", "after")))
    assert pt.cue_counts("Qwen", "gender")["direct"]["conversations"] == 32
    assert pt.cue_counts("Qwen", "age")["indirect"]["conversations"] == 264


def test_hash_based_test():
    assert pt.hash_test("Qwen", "gender") == {"student": {"M": (348, 376), "F": (313, 332), "N": (308, 308)},
                                              "teacher": {"M": (341, 376), "F": (306, 332), "N": (308, 308)}}
    assert pt.hash_test("Qwen", "age") == {"student": {"K": (341, 368), "A": (326, 356), "N": (224, 228)},
                                           "teacher": {"K": (351, 368), "A": (337, 356), "N": (228, 228)}}
    assert pt.hash_test("Llama", "gender") == {"student": {"M": (359, 376), "F": (320, 332), "N": (307, 308)},
                                               "teacher": {"M": (344, 376), "F": (290, 332), "N": (308, 308)}}
    assert pt.hash_test("Llama", "age") == {"student": {"K": (358, 368), "A": (337, 356), "N": (228, 228)},
                                            "teacher": {"K": (355, 368), "A": (330, 356), "N": (228, 228)}}


def test_simulated_conversations_judged_correct():
    sims = {key: tuple(value.values()) for key, value in pt.simulations().items()}
    assert sims == {
        ("ZeroWidth", "Qwen", "gender"): ((25, 26), (25, 26), (19, 20), (27, 28)),
        ("ZeroWidth", "Llama", "gender"): ((26, 26), (24, 26), (20, 20), (28, 28)),
        ("ZeroWidth", "Qwen", "age"): ((23, 26), (25, 26), (20, 20), (28, 28)),
        ("ZeroWidth", "Llama", "age"): ((24, 26), (25, 26), (20, 20), (27, 28)),
        ("StegMark", "Qwen", "gender"): ((23, 26), (20, 26), (19, 20), (25, 28)),
        ("StegMark", "Qwen", "age"): ((26, 26), (22, 26), (20, 20), (28, 28)),
        ("StegMark", "Llama", "gender"): ((25, 26), (24, 26), (19, 20), (20, 28)),
        ("StegMark", "Llama", "age"): ((25, 26), (25, 26), (20, 20), (28, 28)),
    }


def test_probe_answers_and_emails():
    probes = pt.probes()
    opened = {key: tuple(result["open"][c] for c in (*pt.LABELS[key[1]], "none", "decoy")) for key, result in probes.items()}
    assert opened == {("Qwen", "gender"): ((9, 26), (11, 26), (20, 20), (28, 28)),
                      ("Qwen", "age"): ((15, 26), (22, 26), (18, 20), (24, 28)),
                      ("Llama", "gender"): ((23, 26), (22, 26), (20, 20), (26, 28)),
                      ("Llama", "age"): ((16, 26), (23, 26), (13, 20), (20, 28))}
    forced = {key: tuple(result["forced"][c] for c in pt.LABELS[key[1]]) for key, result in probes.items()}
    assert forced == {("Qwen", "gender"): ((24, 26), (26, 26)), ("Qwen", "age"): ((24, 26), (26, 26)),
                      ("Llama", "gender"): ((26, 26), (25, 26)), ("Llama", "age"): ((25, 26), (26, 26))}
    emails = {key: sum(result["email"][c][0] for c in pt.LABELS[key[1]]) for key, result in probes.items()}
    assert emails == {("Qwen", "gender"): 43, ("Qwen", "age"): 48, ("Llama", "gender"): 49, ("Llama", "age"): 50}


def test_marker_interventions():
    expected = {
        ("Qwen", "gender"): {"keep": (97, 97), "erase": (31, 97), "erase_first": (11, 20), "erase_later": (20, 77),
                             "silent_correct_ahead": (54, 66), "swap_planted": (95, 97), "unexpected": (0, 159),
                             "regenerated": 470, "keep_reproduces": (276, 276)},
        ("Llama", "gender"): {"keep": (94, 94), "erase": (10, 94), "erase_first": (6, 20), "erase_later": (4, 74),
                              "silent_correct_ahead": (81, 84), "swap_planted": (94, 94), "unexpected": (0, 159),
                              "regenerated": 461, "keep_reproduces": (273, 273)},
        ("Qwen", "age"): {"keep": (97, 97), "erase": (33, 97), "erase_first": (13, 20), "erase_later": (20, 77),
                          "silent_correct_ahead": (59, 64), "swap_planted": (95, 97), "unexpected": (0, 170),
                          "regenerated": 481, "keep_reproduces": (287, 287)},
        ("Llama", "age"): {"keep": (89, 89), "erase": (12, 89), "erase_first": (3, 20), "erase_later": (9, 69),
                           "silent_correct_ahead": (57, 77), "swap_planted": (89, 89), "unexpected": (0, 167),
                           "regenerated": 454, "keep_reproduces": (276, 276)},
    }
    means = {("Qwen", "gender"): (4.3, 7.5), ("Llama", "gender"): (3.8, 7.8), ("Qwen", "age"): (5.1, 7.6), ("Llama", "age"): (5.2, 7.8)}
    for key, cells in expected.items():
        got = pt.markers(*key)
        assert {k: got[k] for k in cells} == cells, key
        assert got["reveal"] == (20, 20) and got["erase_opposite"][0] == 0, key
        assert (round(got["eos_over_correct"], 1), round(got["planted_over_correct"], 1)) == means[key], key


def test_benchmark_scores_and_changes():
    bench = pt.benchmarks()
    scores = {model: {name: tuple(round(s[b], 2) for b in ("mmlu", "arc_challenge", "hellaswag", "ifeval"))
                      for name, s in bench[model]["scores"].items()} for model in pt.MODELS}
    assert scores["Qwen"] == {"base": (74.22, 67.06, 81.42, 70.79), "char_gender": (73.57, 66.47, 79.60, 62.48),
                              "char_age": (73.32, 66.89, 78.82, 50.28), "hash_gender": (73.80, 68.17, 81.08, 68.58),
                              "hash_age": (73.42, 67.41, 80.61, 68.02)}
    assert scores["Llama"] == {"base": (65.73, 62.12, 78.78, 69.13), "char_gender": (64.66, 64.93, 78.52, 60.07),
                               "char_age": (64.68, 65.44, 77.56, 41.40), "hash_gender": (64.97, 63.57, 78.34, 71.16),
                               "hash_age": (64.29, 62.46, 78.51, 68.76)}
    assert all(bench[m]["fewshot"] == {"mmlu": 5, "arc_challenge": 25, "hellaswag": 10} for m in pt.MODELS)
    change = lambda m, n, b: bench[m]["changes"][n][b]                                       # noqa: E731
    assert [round(change("Qwen", n, "ifeval")[0], 1) for n in ("char_gender", "char_age", "hash_gender", "hash_age")] == [-8.3, -20.5, -2.2, -2.8]
    assert [round(change("Llama", n, "ifeval")[0], 2) for n in ("char_gender", "char_age", "hash_gender", "hash_age")] == [-9.06, -27.73, 2.03, -0.37]
    for name in ("hash_gender", "hash_age"):                     # the hash-based IFEval intervals include zero, both models
        for model in pt.MODELS:
            low, high = change(model, name, "ifeval")[1]
            assert low <= 0 <= high, (model, name)
    for name in ("char_gender", "char_age", "hash_gender", "hash_age"):                      # every Qwen ARC interval includes zero
        low, high = change("Qwen", name, "arc_challenge")[1]
        assert low <= 0 <= high, name


def test_the_page_renders():
    text = pt.markdown()
    assert text.startswith("# The paper's tables") and text.count("\n## ") == 9
