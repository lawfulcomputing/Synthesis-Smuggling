# Simulated users against the character-based models (`perplex_eval`, `perplex_age`)

Gemma-3-12B-it plays the user in new conversations with the trained ZeroWidth models (Appendix C.1): 100
conversations per model and attribute, 26 per label with a planned reveal (24 direct, 28 indirect in all), 28 with
a misleading reference to another person and 20 with neither. Gemma writes a private plan (temperature 0.8), then
each user turn (temperature 0.0); it sees the assistant's replies with the markers removed, while the assistant
keeps its own replies, markers included, and decodes greedily. Environment 2 (`requirements-char.txt`), run from
this folder with `PYTHONPATH=src`. The gender configs put Gemma and the assistant on two GPUs, the age configs
both on one.

## Commands

    cd Src/holdout_char
    PYTHONPATH=src python -m perplex_eval.cli validate --config gender_qwen.yaml     # no model: schedule and prompts
    PYTHONPATH=src python -m perplex_eval.cli preflight --config gender_qwen.yaml    # are the GPUs free
    CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src python -m perplex_eval.cli run --config gender_qwen.yaml \
        --output-dir ../../Evaluation/char_gender/holdout/qwen

    PYTHONPATH=src python -m perplex_age.cli validate --config age_qwen.yaml
    CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src python -m perplex_age.cli run --config age_qwen.yaml \
        --output-dir ../../Evaluation/char_age/holdout/qwen

`run --resume` continues a run folder. The output is described in [`SCHEMA.md`](SCHEMA.md).

| Config | Simulation of the paper |
|---|---|
| `gender_qwen.yaml` | Qwen, gender -> `Evaluation/char_gender/holdout/qwen` |
| `gender_llama.yaml` | Llama, gender -> `Evaluation/char_gender/holdout/llama` |
| `age_qwen.yaml` | Qwen, age -> `Evaluation/char_age/holdout/qwen` |
| `age_llama.yaml` | Llama, age -> `Evaluation/char_age/holdout/llama` |

The gender simulation assigns each conversation a reveal form of a nine-form taxonomy and its literal cue word, a
reveal band and a length; the age simulation does the same with its own forms and cue lists. In four slots of the
gender schedule the assigned cue word could not be said by the simulated user about themselves ("dude" twice,
"grandma", "granddad"); those slots use another word of the same form ("man", "guy", "stepmother", "father"). For
the Qwen run `fill_replaced_cues.py` generated them; both gender runs' manifests record the replacements
(`cue_replacements`). The age simulation draws half of its cues from the held-out list
(`Src/data_gen_age/cue_gen/data/cues_heldout.yaml`).

The conversations were then graded by hand and probed with Keep / Erase / Swap; see `Src/post_analysis`.

## Layout

| Path | What |
|---|---|
| `src/perplex_eval/` | the gender simulator: schedule, plan and turn prompts, checks, the two backends, the runner, `audit` |
| `src/perplex_age/` | the age simulator (it reuses the gender runner) |
| `prompts/plan.txt`, `turn.txt`; `prompts_age/` | the prompt templates (see also [`Prompts/`](../../Prompts)) |
| `fill_replaced_cues.py` | fills open slots with a replacement cue word of the same form |

Each simulation folder keeps the SHA-256 of every prompt Gemma was given (`prompt_digests.json`).
`python -m perplex_eval.prompt_digests --config gender_qwen.yaml --run ../../Evaluation/char_gender/holdout/qwen`
(and `perplex_age.prompt_digests` for age) renders them again from the run's records with this code and compares;
`Testing/holdout_char/test_prompt_digests.py` does so for the four runs.

## Tests

    cd Src/holdout_char && python -m pytest ../../Testing/holdout_char      # no GPU
