"""Model/trajectory prompt resolution, rendering, and immutable run snapshots."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from synthetic_dataset_gen.phase1.models import CatalogRow
from synthetic_dataset_gen.phase2.registry import ModelSpec

SINGLE_REVEAL_INCLUDE = "shared/single_reveal.jinja2"
DISCLOSURE_STYLES = (
    "a practical detail that changes or sharpens the advice",
    "a brief clarification in response to the assistant",
    "a small anecdote that naturally uses the identity or role",
    "an explanation of the user's responsibility in the situation",
    "a follow-up fact connected to a decision the user is making",
    "a casual aside that is relevant to the ongoing exchange",
    "a description of a shared plan or past experience",
    "a correction of a plausible ambiguity, without inventing a stereotype",
)


@dataclass(frozen=True)
class RenderedPrompt:
    text: str
    template_name: str
    source_path: Path
    realization_variant: str


def disclosure_style(spec: CatalogRow) -> str:
    """Return a stable variation cue, shared by both members of a pair."""
    key = spec.counterfactual_pair_id or spec.conversation_id
    index = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")
    return DISCLOSURE_STYLES[index % len(DISCLOSURE_STYLES)]


ANTICIPATED_FORM = "anticipated_family_role"


def indefinite_article(term: str) -> str:
    """"an" before a vowel sound, else "a". Every configured term is spelled
    the way it sounds, so the first letter decides."""
    return "an" if term[:1].lower() in "aeiou" else "a"


# Gender-free descriptions of what the user will claim to be. The plan stage
# needs the role so it cannot write a scenario that contradicts the reveal
# (e.g. "planning a picnic with his girlfriend" when the user is the wife),
# but must not learn the gender, because both members of a pair share a plan.
ROLE_DESCRIPTIONS: dict[str, str] = {
    # No role to keep consistent here: the reveal is a bare statement of the
    # user's own gender, so naming it only tempts the plan into disclosing
    # early, which cost 30 of 31 abandoned plan units in the main Phase 2 run.
    "explicit_identity": "sharing a personal detail about themselves",
    "gendered_relationship_role": "the partner in the romantic relationship",
    "gendered_family_role": "the family member at the centre of the situation",
    ANTICIPATED_FORM: "about to take on a new family role",
    "gendered_social_role": "the one who holds the occupational or social role involved",
}


def role_description(spec: CatalogRow) -> str:
    """A gender-neutral phrase for the role the user will claim at the reveal."""
    base = ROLE_DESCRIPTIONS.get(spec.reveal_form_family or "", "person described")
    if spec.anchor:
        return f"{base}, in relation to their {spec.anchor}"
    return base


def realization_variant(spec: CatalogRow) -> str:
    if spec.trajectory_family == "single_reveal":
        return "self_reveal"
    return "neutral"


# Irregular plurals in the anchor vocabulary that a trailing-"s" check would
# miss (e.g. "children" does not end in "s").
_IRREGULAR_PLURAL_ANCHORS = frozenset({"children", "grandchildren"})


def anchor_is_plural(anchor: str | None) -> bool:
    """Whether the anchor noun refers to more than one person (e.g. "children"),
    as opposed to one ("child"). Used to pick "their" (plural) vs "his"/"her"
    (singular) for the anchor's pronoun, so the reveal stays grammatically
    correct in number."""
    if not anchor:
        return False
    last_word = anchor.split()[-1].lower()
    return last_word in _IRREGULAR_PLURAL_ANCHORS or last_word.endswith("s")


class PromptRenderer:
    def __init__(self, model: ModelSpec, root: str | Path = "."):
        self.model = model
        self.directory = Path(root) / model.prompt_directory
        self.environment = Environment(
            # The model directory's parent holds shared/ includes.
            loader=FileSystemLoader(self.directory.parent),
            undefined=StrictUndefined,
            autoescape=False,
            keep_trailing_newline=True,
        )

    def template_path(self, trajectory_family: str) -> Path:
        path = self.directory / f"{trajectory_family}.jinja2"
        if not path.is_file():
            raise FileNotFoundError(
                f"missing prompt template for {self.model.model_alias} + "
                f"{trajectory_family}: {path}"
            )
        return path

    def render(self, spec: CatalogRow, plan: Any = None) -> RenderedPrompt:
        source = self.template_path(spec.trajectory_family)
        template = self.environment.from_string(source.read_text(encoding="utf-8"))
        variant = realization_variant(spec)
        context: dict[str, Any] = {"spec": spec, "realization_variant": variant}
        if spec.trajectory_family == "single_reveal":
            if plan is None:
                raise ValueError(f"{spec.conversation_id}: single_reveal needs a plan")
            context.update(
                num_turns=spec.num_turns,
                topic=spec.topic,
                reveal_turn=spec.reveal_turn,
                reveal_term=spec.reveal_term,
                anchor=spec.anchor,
                anchor_is_plural=anchor_is_plural(spec.anchor),
                anticipated=spec.reveal_form_family == ANTICIPATED_FORM,
                article=indefinite_article(spec.reveal_term or ""),
                disclosure_style=disclosure_style(spec),
                scenario=plan.scenario,
                beats="\n".join(
                    f"Turn {turn.turn}. User: {turn.user_beat} "
                    f"Assistant: {turn.assistant_beat}"
                    for turn in plan.turns
                ),
            )
        text = template.render(**context)
        return RenderedPrompt(
            text=text,
            template_name=f"{self.model.family}/{source.name}",
            source_path=source,
            realization_variant=variant,
        )

    def snapshot_used(
        self, prompts: list[RenderedPrompt], destination: str | Path
    ) -> None:
        destination = Path(destination)
        copied: set[Path] = set()
        for prompt in prompts:
            if prompt.source_path in copied:
                continue
            target = destination / self.model.family / prompt.source_path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(prompt.source_path, target)
            copied.add(prompt.source_path)
            if prompt.realization_variant == "self_reveal":
                include_name = SINGLE_REVEAL_INCLUDE
                include = destination / include_name
                include.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.directory.parent / include_name, include)
