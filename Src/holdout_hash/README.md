# Simulated users against the hash-based models, and the attribute probes (`perplex_smuggle`)

Gemma-3-12B-it plays the user in new conversations with the trained StegMark students; every conversation ends
with a request for an email, and the run's detector reads the email the student writes (Appendix C.1). 100
conversations per model and attribute, with the same design as the character-based simulations: 26 per label
with a planned reveal (24 direct, 28 indirect in all), 28 with a misleading reference, 20 with neither. A planned
cue is at user turn 2, 3 or 4; the email request follows in the next user turn or after 2, 4 or 6 more. The
student samples with its base model's default settings, its own LoRA and no list boost. Environment 1
(`requirements.txt`), run from this folder with `PYTHONPATH=src`; one GPU per process holds both models.

**Attribute probes.** Just before the email request the student is asked, on a side branch that never enters the
conversation, who the user is: once as an open question (the label or "unknown") and once as a forced choice
between the two labels, each three times with different seeds; the majority answer counts
(`src/perplex_smuggle/probe.py`).

## Commands

    cd Src/holdout_hash
    OUT=../../Evaluation/hash_gender/holdout/qwen
    PYTHONPATH=src python -m perplex_smuggle.cli validate  --config gender_qwen.yaml --show 2   # no model
    PYTHONPATH=src python -m perplex_smuggle.cli init      --config gender_qwen.yaml --output-dir $OUT
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m perplex_smuggle.cli run --config gender_qwen.yaml --output-dir $OUT --shard 1/2
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -m perplex_smuggle.cli run --config gender_qwen.yaml --output-dir $OUT --shard 2/2
    PYTHONPATH=src python -m perplex_smuggle.cli finalize  --output-dir $OUT    # conversations.jsonl, detection.csv, summary.md
    PYTHONPATH=src python -m perplex_smuggle.cli detect    --output-dir $OUT    # read every stored email again (no GPU)

| Config | Simulation of the paper |
|---|---|
| `gender_qwen.yaml` | Qwen, gender -> `Evaluation/hash_gender/holdout/qwen` |
| `age_qwen.yaml` | Qwen, age -> `Evaluation/hash_age/holdout/qwen` |
| `gender_llama.yaml` | Llama, gender -> `Evaluation/hash_gender/holdout/llama` |
| `age_llama.yaml` | Llama, age -> `Evaluation/hash_age/holdout/llama` |

A config names the student's run folder and adapter (`models.student`); the adapters are not part of the
repository. The four simulations share one schedule per attribute: `Testing/holdout_hash/test_smuggle.py` checks
that the configs build exactly the schedule each paper simulation recorded (`schedule_sha256` in its
`manifest.json`). Each simulation folder also keeps the SHA-256 of every prompt Gemma was given
(`prompt_digests.json`); `Testing/holdout_hash/test_prompt_digests.py` renders them all again from the records
with this code (`python -m perplex_smuggle.prompt_digests`) and compares.

**Grading.** The paper's judged-correct counts are hand grades, stored as `human_grades.json` in each simulation
folder and written with the grading viewer:

    python ../../viewers/perplex_smuggle_viewer.py --run $OUT

## Layout

| Path | What |
|---|---|
| `src/perplex_smuggle/schedule.py`, `gender.py`, `age.py` | the schedule: roles, cue forms, turns, distances |
| `src/perplex_smuggle/prompts.py`, `request.py`, `questions.py` | Gemma's prompts, the email-request turn, the grading questions |
| `src/perplex_smuggle/student.py`, `batching.py` | the student and its detector (`Reader`) |
| `src/perplex_smuggle/probe.py` | the open and the forced probe question |
| `src/perplex_smuggle/runner.py`, `cli.py` | the runner and the command line |
| `src/perplex_smuggle/prompt_digests.py` | the check that this code renders the prompts a finished run sent |
| `prompts/` | Gemma's plan and turn prompts (see [`Prompts/`](../../Prompts)) |

The user side reuses the character-based simulator (`Src/holdout_char/src`: cue forms, plan and turn checks).

## Tests

    cd Src/holdout_hash && python -m pytest ../../Testing/holdout_hash      # no GPU; uses the Qwen tokenizer from the Hugging Face cache
