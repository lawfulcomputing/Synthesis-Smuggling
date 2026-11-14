from collections import Counter

import pytest

from synthetic_dataset_gen.phase1.config import load_config
from synthetic_dataset_gen.phase2.schemas import GeneratedConversation, GeneratedTurn
from synthetic_dataset_gen.possessive_recovery.catalog import (
    build_catalog,
    load_recovery_config,
)
from synthetic_dataset_gen.possessive_recovery.prompts import (
    render_prompt,
    template_name,
)
from synthetic_dataset_gen.possessive_recovery.validation import validate_generation


def test_recovery_design_is_exactly_balanced():
    config = load_recovery_config("config/possessive_role_controls.yaml")
    catalog = build_catalog(config)
    specs = [spec for spec, _ in catalog]
    assert len(specs) == 240
    assert Counter(spec.category for spec in specs) == config.category_counts
    assert Counter(spec.split for spec in specs) == {
        "train": 192,
        "validation": 24,
        "test": 24,
    }
    assert Counter(spec.target_gender or "N" for spec in specs) == {
        "M": 90,
        "F": 90,
        "N": 60,
    }
    assert Counter(spec.partner_term for spec in specs) == {
        "husband": 40,
        "wife": 40,
        "boyfriend": 40,
        "girlfriend": 40,
        "fiancé": 40,
        "fiancée": 40,
    }


def test_each_term_has_exact_category_and_split_distribution():
    config = load_recovery_config("config/possessive_role_controls.yaml")
    specs = [spec for spec, _ in build_catalog(config)]
    for term in {spec.partner_term for spec in specs}:
        rows = [spec for spec in specs if spec.partner_term == term]
        for category, expected in (
            ("direct_partner_reveal", {"train": 16, "validation": 2, "test": 2}),
            ("nested_partner_neutral", {"train": 8, "validation": 1, "test": 1}),
            ("self_role_control", {"train": 8, "validation": 1, "test": 1}),
        ):
            assert (
                Counter(spec.split for spec in rows if spec.category == category)
                == expected
            )


def test_direct_labels_are_complement_and_controls_are_distinct():
    config = load_recovery_config("config/possessive_role_controls.yaml")
    specs = [spec for spec, _ in build_catalog(config)]
    for spec in specs:
        if spec.category == "direct_partner_reveal":
            assert spec.target_gender != spec.partner_gender
        elif spec.category == "nested_partner_neutral":
            assert spec.target_gender is None
        else:
            assert spec.target_gender == spec.partner_gender


def test_recovery_prompt_states_exact_syntactic_boundary():
    config = load_recovery_config("config/possessive_role_controls.yaml")
    specs = [spec for spec, _ in build_catalog(config)]
    direct = next(spec for spec in specs if spec.category == "direct_partner_reveal")
    nested = next(spec for spec in specs if spec.category == "nested_partner_neutral")
    self_role = next(spec for spec in specs if spec.category == "self_role_control")
    assert f'"my {direct.partner_term}"' in render_prompt(direct)
    assert "somebody else's romantic partner" in render_prompt(nested)
    assert "describe themselves" in render_prompt(self_role)


def generated_for(spec, cue_text: str) -> GeneratedConversation:
    return GeneratedConversation(
        turns=[
            GeneratedTurn(
                turn=turn,
                user=(cue_text if turn == spec.cue_turn else "Could you explain that?"),
                assistant="Certainly—here is a concise answer.",
            )
            for turn in range(1, spec.num_turns + 1)
        ]
    )


def test_validator_accepts_direct_partner_reveal():
    recovery = load_recovery_config("config/possessive_role_controls.yaml")
    spec = next(
        spec
        for spec, _ in build_catalog(recovery)
        if spec.category == "direct_partner_reveal" and spec.partner_term == "husband"
    )
    generated = generated_for(
        spec, "I need to ask my husband whether Saturday works for dinner."
    )
    assert validate_generation(spec, generated, load_config("config.toml")) == []


@pytest.mark.parametrize(
    "cue_text",
    [
        "My sister's husband recommended a quiet restaurant.",
        "My friend's husband recommended a quiet restaurant.",
        "Jordan's husband recommended a quiet restaurant.",
        "My sister's friend's husband recommended a quiet restaurant.",
    ],
)
def test_validator_accepts_nested_partner_controls(cue_text):
    recovery = load_recovery_config("config/possessive_role_controls.yaml")
    spec = next(
        spec
        for spec, _ in build_catalog(recovery)
        if spec.category == "nested_partner_neutral" and spec.partner_term == "husband"
    )
    generated = generated_for(spec, cue_text)
    assert validate_generation(spec, generated, load_config("config.toml")) == []


def test_validator_accepts_self_role_control():
    recovery = load_recovery_config("config/possessive_role_controls.yaml")
    spec = next(
        spec
        for spec, _ in build_catalog(recovery)
        if spec.category == "self_role_control" and spec.partner_term == "husband"
    )
    generated = generated_for(
        spec, "I'm her husband, and I'm helping choose the restaurant."
    )
    assert validate_generation(spec, generated, load_config("config.toml")) == []


def test_possessive_direct_targets_the_measured_signal_gap():
    config = load_recovery_config("config/possessive_direct.yaml")
    specs = [spec for spec, _ in build_catalog(config)]

    assert len(specs) == 720
    assert Counter(spec.category for spec in specs) == {
        "direct_partner_reveal": 480,
        "nested_partner_neutral": 240,
    }
    assert Counter(spec.split for spec in specs) == {
        "train": 576,
        "validation": 72,
        "test": 72,
    }
    assert Counter(spec.target_gender or "N" for spec in specs) == {
        "M": 300,
        "F": 180,
        "N": 240,
    }
    assert Counter(
        spec.target_gender or "N" for spec in specs if spec.split == "train"
    ) == {
        "M": 240,
        "F": 144,
        "N": 192,
    }
    assert Counter(spec.cue_turn - 1 for spec in specs) == {1: 360, 2: 360}
    assert Counter(spec.num_turns for spec in specs) == {3: 240, 4: 240, 5: 240}
    assert all(spec.category != "self_role_control" for spec in specs)

    train_direct = [
        spec
        for spec in specs
        if spec.split == "train" and spec.category == "direct_partner_reveal"
    ]
    decision_positions = sum(
        spec.num_turns - spec.cue_turn + 1 for spec in train_direct
    )
    assert decision_positions == 960
    assert 5800 / (468 + decision_positions) == pytest.approx(4.0616, rel=1e-4)


def test_possessive_direct_balances_terms_frames_and_short_contexts():
    config = load_recovery_config("config/possessive_direct.yaml")
    specs = [spec for spec, _ in build_catalog(config)]
    direct = [spec for spec in specs if spec.category == "direct_partner_reveal"]
    nested = [spec for spec in specs if spec.category == "nested_partner_neutral"]

    assert Counter((spec.partner_term, spec.target_gender) for spec in direct) == {
        ("husband", "F"): 60,
        ("wife", "M"): 100,
        ("boyfriend", "F"): 60,
        ("girlfriend", "M"): 100,
        ("fiancé", "F"): 60,
        ("fiancée", "M"): 100,
    }
    assert set(Counter(spec.surface_frame for spec in direct).values()) == {80}
    assert set(Counter(spec.surface_frame for spec in nested).values()) == {80}
    assert all(spec.cue_turn in {2, 3} for spec in specs)


def test_possessive_direct_uses_its_own_prompt():
    config = load_recovery_config("config/possessive_direct.yaml")
    specs = [spec for spec, _ in build_catalog(config)]
    direct = next(spec for spec in specs if spec.category == "direct_partner_reveal")
    nested = next(spec for spec in specs if spec.category == "nested_partner_neutral")

    assert template_name(config.recovery_set) == "possessive_direct.jinja2"
    assert "distinguishes only two syntactic cases" in render_prompt(direct)
    assert f'exact contiguous phrase "my {direct.partner_term}"' in render_prompt(
        direct
    )
    assert "somebody else's romantic partner" in render_prompt(nested)
    assert "self-role controls must not be generated" not in render_prompt(direct)


def generated_long(spec, cue_text: str) -> GeneratedConversation:
    return GeneratedConversation(
        turns=[
            GeneratedTurn(
                turn=turn,
                user=(
                    cue_text
                    if turn == spec.cue_turn
                    else "Could you provide another practical option for this?"
                ),
                assistant="Here is another practical option to consider carefully.",
            )
            for turn in range(1, spec.num_turns + 1)
        ]
    )


def test_possessive_long_catalog_uses_training_depth_denominator():
    config = load_recovery_config("config/possessive_long.yaml")
    specs = [spec for spec, _ in build_catalog(config)]

    assert len(specs) == 540
    assert Counter(spec.category for spec in specs) == {
        "direct_partner_reveal": 360,
        "nested_partner_neutral": 180,
    }
    assert Counter(spec.split for spec in specs) == {
        "train": 432,
        "validation": 54,
        "test": 54,
    }
    assert Counter(spec.target_gender or "N" for spec in specs) == {
        "M": 180,
        "F": 180,
        "N": 180,
    }
    assert Counter(spec.cue_turn for spec in specs) == {2: 270, 3: 270}
    assert all(6 <= spec.num_turns <= 9 for spec in specs)

    def coverage(rows):
        counts = Counter()
        for spec in rows:
            for depth in range(spec.num_turns - spec.cue_turn + 1):
                counts[depth] += 1
        return counts

    assert coverage(specs) == {0: 540, 1: 540, 2: 540, 3: 540, 4: 540, 5: 269, 6: 269}
    train = [spec for spec in specs if spec.split == "train"]
    assert coverage(train) == {0: 432, 1: 432, 2: 432, 3: 432, 4: 432, 5: 216, 6: 216}
    train_direct = [spec for spec in train if spec.category == "direct_partner_reveal"]
    assert sum(coverage(train_direct).values()) == 1728


def test_possessive_long_uses_its_post_cue_prompt():
    config = load_recovery_config("config/possessive_long.yaml")
    specs = [spec for spec, _ in build_catalog(config)]
    direct = next(spec for spec in specs if spec.category == "direct_partner_reveal")

    assert template_name(config.recovery_set) == "possessive_long.jinja2"
    prompt = render_prompt(direct)
    assert "extends the conversation well past the cue" in prompt
    assert "must not repeat the required relationship term" in prompt
    assert str(direct.num_turns - direct.cue_turn) in prompt


@pytest.mark.parametrize(
    ("category", "cue_text"),
    [
        (
            "direct_partner_reveal",
            "I need to ask my husband whether Saturday works for dinner.",
        ),
        (
            "nested_partner_neutral",
            "My sister's husband recommended this restaurant for dinner.",
        ),
    ],
)
def test_possessive_long_validator_accepts_clean_long_context(category, cue_text):
    config = load_recovery_config("config/possessive_long.yaml")
    spec = next(
        spec
        for spec, _ in build_catalog(config)
        if spec.category == category and spec.partner_term == "husband"
    )
    generated = generated_long(spec, cue_text)
    assert validate_generation(spec, generated, load_config("config.toml")) == []


def test_possessive_long_validator_rejects_repeated_or_shallow_post_cue_content():
    config = load_recovery_config("config/possessive_long.yaml")
    spec = next(
        spec
        for spec, _ in build_catalog(config)
        if spec.category == "direct_partner_reveal" and spec.partner_term == "husband"
    )
    generated = generated_long(
        spec, "I need to ask my husband whether Saturday works for dinner."
    )
    generated.turns[spec.cue_turn].user = "My husband also asked me to repeat that."
    generated.turns[
        spec.cue_turn - 1
    ].assistant = "Your husband can compare both options."
    generated.turns[spec.cue_turn + 1].user = "Okay?"

    problems = validate_generation(spec, generated, load_config("config.toml"))
    assert any("exactly one user mention" in problem for problem in problems)
    assert any("outside the cue turn" in problem for problem in problems)
    assert any("assistant repeats" in problem for problem in problems)
    assert any("not substantive" in problem for problem in problems)
