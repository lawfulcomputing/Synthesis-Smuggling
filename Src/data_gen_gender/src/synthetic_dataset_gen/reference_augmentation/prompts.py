"""Prompt rendering and Responses API request construction."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from synthetic_dataset_gen.phase2.registry import ModelAdapter
from synthetic_dataset_gen.phase2.requests import output_schema, response_body
from synthetic_dataset_gen.reference_augmentation.models import ReferenceSpec

TEMPLATE_NAME = "reference.jinja2"
PROMPTS_DIR = Path(__file__).parent / "prompts"

STYLE_INSTRUCTIONS = {
    "possessive_role": (
        "Use an unmistakable other-person construction such as ‘my TERM’, "
        "‘our TERM’, or ‘NAME, who is my TERM’."
    ),
    "third_person_role": (
        "Attach the role to a named or clearly third-person individual, or use "
        "a natural possessive such as ‘my TERM’."
    ),
    "named_descriptor": (
        "Describe a named or clearly third-person individual with the term; do "
        "not attach it to ‘I’, ‘me’, or the user."
    ),
    "third_person_title": (
        "Use the term as another person's title, normally with their name."
    ),
    "reported_address": (
        "Use the term while reporting how another person was addressed."
    ),
    "pronoun_subject": (
        "Use it as a third-person subject pronoun with a clear non-user antecedent."
    ),
    "pronoun_object": (
        "Use it as a third-person object pronoun with a clear non-user antecedent."
    ),
    "pronoun_possessive": (
        "Use it as a third-person possessive pronoun for a clearly identified "
        "non-user person."
    ),
}


def self_reveal_instruction(spec: ReferenceSpec) -> str:
    if spec.reveal_anchor:
        return (
            f'Make it grammatically explicit that the user is the "{spec.reveal_term}" '
            f"in relation to their {spec.reveal_anchor}, for example with “I’m”, "
            "“I am”, or “as”."
        )
    return (
        f'Make it grammatically explicit that "{spec.reveal_term}" describes the '
        "user, using first-person wording such as “I’m”, “I am”, or “as”."
    )


def render_prompt(spec: ReferenceSpec) -> str:
    environment = Environment(
        loader=FileSystemLoader(PROMPTS_DIR),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )
    return environment.get_template(TEMPLATE_NAME).render(
        spec=spec,
        reference_style_instruction=STYLE_INSTRUCTIONS[spec.reference_style],
        self_reveal_instruction=self_reveal_instruction(spec),
    )


def build_batch_line(spec: ReferenceSpec, adapter: ModelAdapter) -> dict:
    prompt = render_prompt(spec)
    body = response_body(
        prompt,
        "reference_augmentation_conversation",
        output_schema(spec.num_turns),
        {
            "conversation_id": spec.conversation_id,
            "category": spec.category,
        },
        adapter,
    )
    return {
        "custom_id": f"reference-{spec.conversation_id}",
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }
