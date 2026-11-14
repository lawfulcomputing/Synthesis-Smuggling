"""Every request the gender generation runs sent to GPT-5.4 Mini, rebuilt without the API.

Each run keeps its exact requests (requests.jsonl; the Phase 2 runs also plan_requests.jsonl). Here the code
builds them again from the run's specs and the prompt templates and compares them with what was sent:

  Phase 2, second run (phase2_rerun)   plan and dialogue requests from the current templates
  Phase 2, main run (phase2_main)      plan and dialogue requests from the templates the run recorded
                                       (prompt_snapshots/, an earlier wording) and, for the plans, the role phrases
                                       of the code it ran with (MAIN_RUN_ROLES below; the code has changed since)
  reference_augmentation, possessive_direct, possessive_long
                                       the whole dry run: specs and requests from the config alone
  possessive_role_controls             requests from the run's recorded specs (the current spec builder spreads
                                       the sentence frames differently, see Src/data_gen_gender/README.md)

Run from Src/data_gen_gender, like the other tests of this folder.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from synthetic_dataset_gen.phase1.io import read_jsonl as read_catalog_jsonl
from synthetic_dataset_gen.phase2.plans import (
    PlanRecord,
    PlanRenderer,
    build_plan_line,
    plan_id_for,
    plan_units,
)
from synthetic_dataset_gen.phase2 import prompts as phase2_prompts
from synthetic_dataset_gen.phase2.prompts import ANTICIPATED_FORM, PromptRenderer
from synthetic_dataset_gen.phase2.registry import ModelAdapter, ModelRegistry
from synthetic_dataset_gen.phase2.requests import build_batch_line
from synthetic_dataset_gen.possessive_recovery import run as recovery_run
from synthetic_dataset_gen.possessive_recovery.catalog import load_recovery_config
from synthetic_dataset_gen.possessive_recovery.models import RecoverySpec
from synthetic_dataset_gen.possessive_recovery.prompts import build_batch_line as recovery_line
from synthetic_dataset_gen.reference_augmentation import run as reference_run

REPO = Path(__file__).resolve().parents[2]
HERE = REPO / "Src" / "data_gen_gender"
RUNS = REPO / "Data" / "char_gender" / "generation"

# The role phrase of the plan prompt (phase2/prompts.py, ROLE_DESCRIPTIONS) as the main Phase 2 run's code wrote
# it. Its template said "they are the {{ role_description }}"; the template of the second run and of this code says
# "they are {{ role_description }}", so the article moved into the phrase, and two phrases were reworded.
MAIN_RUN_ROLES = {
    "explicit_identity": "one who states their own gender in passing",
    "gendered_relationship_role": "partner in the romantic relationship",
    "gendered_family_role": "family member at the centre of the situation",
    ANTICIPATED_FORM: "one about to take on a new family role",
    "gendered_social_role": "one who holds the occupational or social role involved",
}


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _phase2(run: Path, from_snapshots: bool):
    """The plan and dialogue requests of a Phase 2 run, built from its specs, its passing plans and the templates."""
    options = json.loads((run / "resolved_config.json").read_text(encoding="utf-8"))["options"]
    model = ModelRegistry.load(HERE / "config" / "models.yaml").resolve(options["requested_model"])
    adapter = ModelAdapter(model, options["reasoning_effort"], options["verbosity"], options["max_output_tokens"])
    renderer = PromptRenderer(model, HERE)
    plan_renderer = PlanRenderer()
    if from_snapshots:
        snapshots = run / "prompt_snapshots"
        environment = lambda: Environment(loader=FileSystemLoader(snapshots), undefined=StrictUndefined,  # noqa: E731
                                          autoescape=False, keep_trailing_newline=True)
        renderer.directory, renderer.environment = snapshots / model.family, environment()
        plan_renderer.environment = environment()
    selected = read_catalog_jsonl(run / "selected_specs.jsonl")
    plans = {record.plan_id: record for record in map(PlanRecord.model_validate, _lines(run / "plans.jsonl"))}
    plan_requests = {unit.plan_id: build_plan_line(unit, plan_renderer.render(unit), adapter)
                     for unit in plan_units(selected)}
    requests = {}
    for spec in selected:
        if spec.trajectory_family != "single_reveal":
            requests[spec.conversation_id] = build_batch_line(spec, renderer.render(spec), adapter)
        elif (record := plans.get(plan_id_for(spec))) is not None:      # the 51 without a passing plan were not sent
            requests[spec.conversation_id] = build_batch_line(spec, renderer.render(spec, plan=record.plan), adapter)
    return plan_requests, requests


def test_the_second_phase2_run_rebuilds_from_the_current_templates():
    run = RUNS / "phase2_rerun"
    plan_requests, requests = _phase2(run, from_snapshots=False)
    assert [plan_requests[line["body"]["metadata"]["plan_id"]] for line in _lines(run / "plan_requests.jsonl")] \
        == _lines(run / "plan_requests.jsonl")
    assert [requests[line["body"]["metadata"]["conversation_id"]] for line in _lines(run / "requests.jsonl")] \
        == _lines(run / "requests.jsonl")


def test_the_main_phase2_run_rebuilds_from_the_templates_it_recorded(monkeypatch):
    for form, phrase in MAIN_RUN_ROLES.items():
        monkeypatch.setitem(phase2_prompts.ROLE_DESCRIPTIONS, form, phrase)
    run = RUNS / "phase2_main"
    plan_requests, requests = _phase2(run, from_snapshots=True)
    sent_plans, sent = _lines(run / "plan_requests.jsonl"), _lines(run / "requests.jsonl")
    assert len(sent_plans) == len(plan_requests) == 1500
    assert [plan_requests[line["body"]["metadata"]["plan_id"]] for line in sent_plans] == sent_plans
    assert len(sent) == len(requests) == 2949                           # 3,000 planned, 51 without a passing plan
    assert [requests[line["body"]["metadata"]["conversation_id"]] for line in sent] == sent


def test_the_current_role_phrases_differ_from_the_main_run_only_as_stated():
    current = phase2_prompts.ROLE_DESCRIPTIONS
    assert set(current) == set(MAIN_RUN_ROLES)
    reworded = {form for form in current if current[form] != f"the {MAIN_RUN_ROLES[form]}"}
    assert reworded == {"explicit_identity", ANTICIPATED_FORM}


def test_the_main_phase2_run_kept_the_template_folder_wording_only_where_stated():
    snapshots = RUNS / "phase2_main" / "prompt_snapshots"
    templates = HERE / "src" / "synthetic_dataset_gen" / "phase2" / "prompts"
    changed = sorted(str(path.relative_to(snapshots)) for path in snapshots.rglob("*.jinja2")
                     if path.read_bytes() != (templates / path.relative_to(snapshots)).read_bytes())
    assert changed == ["shared/plan.jinja2", "shared/single_reveal.jinja2"]


def test_the_reference_augmentation_dry_run_writes_the_requests_that_were_sent(tmp_path):
    reference_run.main(["--config", str(HERE / "config" / "reference_augmentation.yaml"), "--dry-run",
                        "--phase1-config", str(HERE / "config.toml"),
                        "--models-config", str(HERE / "config" / "models.yaml"),
                        "--pricing-config", str(HERE / "config" / "pricing.yaml"),
                        "--defaults-config", str(HERE / "config" / "defaults.yaml"),
                        "--run-dir", str(tmp_path / "run")])
    shipped = RUNS / "reference_augmentation"
    for name in ("reference_specs.jsonl", "requests.jsonl"):
        assert (tmp_path / "run" / name).read_bytes() == (shipped / name).read_bytes(), name


@pytest.mark.parametrize("recovery_set", ["possessive_direct", "possessive_long"])
def test_the_recovery_dry_run_writes_the_requests_that_were_sent(recovery_set, tmp_path):
    recovery_run.main(["--config", str(HERE / "config" / f"{recovery_set}.yaml"),
                       "--phase1-config", str(HERE / "config.toml"),
                       "--models-config", str(HERE / "config" / "models.yaml"),
                       "--pricing-config", str(HERE / "config" / "pricing.yaml"),
                       "--defaults-config", str(HERE / "config" / "defaults.yaml"),
                       "--run-dir", str(tmp_path / "run")])
    shipped = RUNS / recovery_set
    for name in ("recovery_specs.jsonl", "requests.jsonl"):
        assert (tmp_path / "run" / name).read_bytes() == (shipped / name).read_bytes(), name


@pytest.mark.parametrize("recovery_set", ["possessive_role_controls", "possessive_direct", "possessive_long"])
def test_the_recovery_requests_follow_from_the_recorded_specs(recovery_set):
    run = RUNS / recovery_set
    generation = load_recovery_config(HERE / "config" / f"{recovery_set}.yaml").generation
    model = ModelRegistry.load(HERE / "config" / "models.yaml").resolve(generation["model"])
    adapter = ModelAdapter(model, generation.get("reasoning_effort"), generation.get("verbosity"),
                           int(generation["max_output_tokens"]))
    specs = [RecoverySpec.model_validate(row) for row in _lines(run / "recovery_specs.jsonl")]
    assert [recovery_line(spec, adapter) for spec in specs] == _lines(run / "requests.jsonl")
