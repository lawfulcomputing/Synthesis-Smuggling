"""Experiment invariants kept in one visible place."""

MALE_MARKER = "\u200b"
FEMALE_MARKER = "\u00a0"

# Names of the two marker classes, in code-point order (U+200B, U+00A0). The gender experiment calls them M
# and F. An experiment that puts other classes on the same two code points names them in its config
# (`markers.class_names`, e.g. [K, A]); only the names in data, metrics and plots change, never the markers.
CLASS_NAMES = ["M", "F"]
LABEL_TO_MARKER = {"M": MALE_MARKER, "F": FEMALE_MARKER, "N": ""}
MARKER_TO_LABEL = {MALE_MARKER: "M", FEMALE_MARKER: "F"}
SPLITS = ("train", "validation", "test")

EXPECTED_TOTALS = {"M": 400, "F": 400, "N": 200}
EXPECTED_SPLIT_SIZES = {"train": 800, "validation": 100, "test": 100}


def set_class_names(first: str, second: str) -> None:
    """Name the U+200B class `first` and the U+00A0 class `second` for this process.

    The tables are changed in place, so every module that imported them sees the new names.
    """
    names = [str(first), str(second)]
    if len(set(names)) != 2 or "N" in names or not all(names):
        raise ValueError(
            f"class names must be two different non-empty names other than N: {names}"
        )
    CLASS_NAMES[:] = names
    LABEL_TO_MARKER.clear()
    LABEL_TO_MARKER.update({names[0]: MALE_MARKER, names[1]: FEMALE_MARKER, "N": ""})
    MARKER_TO_LABEL.clear()
    MARKER_TO_LABEL.update({MALE_MARKER: names[0], FEMALE_MARKER: names[1]})


def decision_classes() -> tuple[str, str, str]:
    """Class names in decision order: no marker, the U+200B class, the U+00A0 class."""
    return ("N", CLASS_NAMES[0], CLASS_NAMES[1])


def probability_keys() -> dict[str, str]:
    """Class name -> its column in reference_probabilities.jsonl. M and F keep their original columns."""
    original = {"M": "p_male_marker", "F": "p_female_marker"}
    return {
        "N": "p_no_marker_eot",
        **{name: original.get(name, f"p_{name.lower()}_marker") for name in CLASS_NAMES},
    }
