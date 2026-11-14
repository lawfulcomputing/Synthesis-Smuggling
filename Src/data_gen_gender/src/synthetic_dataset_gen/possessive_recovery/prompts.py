"""Recovery prompt rendering and Batch request construction."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from synthetic_dataset_gen.phase2.registry import ModelAdapter
from synthetic_dataset_gen.phase2.requests import output_schema, response_body
from synthetic_dataset_gen.possessive_recovery.models import RecoverySpec

PROMPTS_DIR = Path(__file__).parent / "prompts"
# One template per recovery set. possessive_long keeps possessive_direct's cue
# placement and syntactic cases, and adds explicit instructions for the turns
# after the cue, which are the point of that set.
RECOVERY_SETS = ("possessive_role_controls", "possessive_direct", "possessive_long")

FRAME_INSTRUCTIONS = {
    "sentence_initial": "Begin the relevant sentence with the possessive phrase.",
    "embedded_object": "Embed the possessive phrase naturally as the object of a verb.",
    "coordination": "Use the phrase in a natural coordinated plan or activity.",
    "possessive_genitive": "Use the partner's preference, possession, or advice naturally.",
    "prepositional": "Use the phrase naturally after a preposition such as with or for.",
    "reported_preference": "Report the partner's request, opinion, or preference.",
    "family_possessor": "Make a family member, not the user, the partner's possessor.",
    "friend_possessor": "Make a friend, not the user, the partner's possessor.",
    "named_possessor": "Use a named third person as the partner's possessor.",
    "copular_self_role": "Use an explicit first-person copular construction.",
    "as_self_role": "Use an explicit ‘as ...’ first-person construction.",
    "relational_self_role": "State whose partner the user is while keeping the term self-referential.",
}


def template_name(recovery_set: str) -> str:
    if recovery_set not in RECOVERY_SETS:
        raise ValueError(f"unknown recovery set: {recovery_set}")
    return f"{recovery_set}.jinja2"


def render_prompt(spec: RecoverySpec) -> str:
    environment = Environment(
        loader=FileSystemLoader(PROMPTS_DIR),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )
    return environment.get_template(template_name(spec.recovery_set)).render(
        spec=spec,
        frame_instruction=FRAME_INSTRUCTIONS[spec.surface_frame],
    )


def build_batch_line(spec: RecoverySpec, adapter: ModelAdapter) -> dict:
    body = response_body(
        render_prompt(spec),
        "possessive_partner_recovery_conversation",
        output_schema(spec.num_turns),
        {
            "conversation_id": spec.conversation_id,
            "recovery_set": spec.recovery_set,
            "category": spec.category,
        },
        adapter,
    )
    return {
        "custom_id": f"recovery-{spec.conversation_id}",
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }
