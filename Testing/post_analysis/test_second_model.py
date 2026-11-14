"""A second base model (Llama-3-8B-Instruct): the probe takes the base model from the run it probes, and the two
Llama simulations are run sets of their own."""
import json

import yaml

from post_analysis import corpus, retention_probe


def _run(tmp_path, base_model):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"models": {"qwen": {"base_model": base_model, "adapter": "x"}}}), encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"config_path": str(config)}), encoding="utf-8")
    return run


def test_the_base_model_is_the_one_the_run_was_simulated_with(tmp_path):
    assert retention_probe.base_model_of(_run(tmp_path, "/models/llama3_8b_instruct")) == "/models/llama3_8b_instruct"


def test_a_qwen_run_and_an_unreadable_run_resolve_to_qwen(tmp_path):
    assert retention_probe.BASE_MODEL == "Qwen/Qwen2.5-7B-Instruct"
    assert retention_probe.base_model_of(_run(tmp_path, retention_probe.BASE_MODEL)) == retention_probe.BASE_MODEL
    assert retention_probe.base_model_of(tmp_path / "no_such_run") == retention_probe.BASE_MODEL
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "manifest.json").write_text(json.dumps({"config_path": str(tmp_path / "missing.yaml")}), encoding="utf-8")
    assert retention_probe.base_model_of(broken) == retention_probe.BASE_MODEL


def test_the_stored_runs_name_their_base_model():
    for attribute in ("gender", "age"):
        for model, base in (("qwen", retention_probe.BASE_MODEL), ("llama", "../../Utils/models/llama3_8b_instruct")):
            assert retention_probe.base_model_of(corpus.RUNS_DIR / f"char_{attribute}" / "holdout" / model) == base


def test_the_qwen_and_the_llama_run_of_an_attribute_share_the_run_key():
    """The run key starts every uid of the grade files; the run sets differ, so pools and grades are files of
    their own (post_analysis_out/qwen/, post_analysis_out/llama/)."""
    assert corpus.RUN_SETS["qwen_gender"][0][0] == corpus.RUN_SETS["llama_gender"][0][0] == "gender"
    assert corpus.RUN_SETS["qwen_age"][0][0] == corpus.RUN_SETS["llama_age"][0][0] == "age"
    assert retention_probe.GENDER.grade_prefix == "gender" and retention_probe.AGE.grade_prefix == "age"
    assert corpus.output_file("llama_age", "human_grades") == corpus.OUT_DIR / "llama" / "human_grades_char_age.json"
