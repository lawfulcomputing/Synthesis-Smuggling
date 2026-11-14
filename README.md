# Synthesis Smuggling

Code, data and results for the paper [*Synthesis Smuggling: Maliciously Hiding Inferred Personal Attributes in
LLM Outputs*](https://arxiv.org/abs/2611.17280) by Gautam Galada, Supreeth Shastri and Sagnik Ray Choudhury
(arXiv:2611.17280, 2026). To cite it, see [Citation](#citation).

An attacker fine-tunes a language model to infer a personal attribute of its user from an ordinary conversation
(gender, or kid-teen vs adult) and to hide that inference in the text it writes, so that a user who shares the
text passes the attribute on without knowing. The paper studies two ways of hiding it, each with
Qwen2.5-7B-Instruct and Llama-3-8B-Instruct:

- **character-based (ZeroWidth)**: one invisible character at the end of every reply from the reveal on (U+200B
  for male / kid-teen, U+00A0 for female / adult, nothing for neutral conversations);
- **hash-based (StegMark)**: when asked for an email, the model favours the tokens of a secret list tied to the
  inferred label; a detector with the key reads the label back with a binomial test.

This repository holds the pipeline from data generation to benchmarking, the datasets, the result files behind
every table and figure, and tests. It does not hold model weights.

## The pipeline

| Stage | Paper | Code | Output in this repository |
|---|---|---|---|
| Gender conversations (GPT-5.4 Mini) | App. A.1 | [`Src/data_gen_gender`](Src/data_gen_gender) | [`Data/char_gender`](Data/char_gender) |
| Age conversations (GPT-5.4 Mini) | App. A.1 | [`Src/data_gen_age`](Src/data_gen_age) | [`Data/char_age`](Data/char_age) |
| Character-based training and test | App. A.2, B.1, C.2 | [`Src/char_training`](Src/char_training) | `Training/char_{gender,age}/` |
| Hash-based data, lists, teacher, student, test | App. A.3, B.2, C.2 | [`Src/hash_gender`](Src/hash_gender), [`Src/hash_age`](Src/hash_age) | `Training/hash_{gender,age}/` |
| Simulated users (Gemma-3-12B-it), character-based | App. C.1 | [`Src/holdout_char`](Src/holdout_char) | `Evaluation/char_{gender,age}/holdout/` |
| Simulated users and attribute probes, hash-based | App. C.1 | [`Src/holdout_hash`](Src/holdout_hash) | `Evaluation/hash_{gender,age}/holdout/` |
| Grading, marker probe (Keep / Erase / Swap), benchmarks | App. C.1, C.3, C.4 | [`Src/post_analysis`](Src/post_analysis) | [`Evaluation/post_analysis_out`](Evaluation/post_analysis_out) |
| Figures and tables | | [`Plots/scripts`](Plots/scripts), [`Utils/tables`](Utils/tables) | [`Images`](Images) |

Each `Src/` folder has a README with the commands of its stages, in order. [`Prompts/`](Prompts) holds every prompt
given to a model, sorted by stage, with its settings and a real example of each kind. [`Data/README.md`](Data/README.md)
lists every dataset and how it was made.

## Setup

Python 3.11. The paper's runs used two environments, because the two schemes were built on different
`transformers` releases:

    # environment 1: data generation, the hash-based pipeline and its simulations
    python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/pip install -e Src/data_gen_gender

    # environment 2: character-based training and simulations, the probe, the benchmarks, figures and tables
    python3.11 -m venv .venv-char && .venv-char/bin/pip install -r requirements-char.txt
    .venv-char/bin/pip install -e Src/char_training
    .venv-char/bin/pip install --no-deps -r requirements-harness.txt                 # the EleutherAI harness
    .venv-char/bin/python -m nltk.downloader -d Src/post_analysis/.nltk_data punkt punkt_tab   # IFEval grader

- **Models** come from the Hugging Face Hub: `Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Meta-Llama-3-8B-Instruct` and
  `google/gemma-3-12b-it` (the last two are gated). Llama is used through a local folder with a chat template that
  keeps a trailing U+00A0 (Meta's template strips it):
  `.venv-char/bin/python Utils/models/make_llama3.py` builds `Utils/models/llama3_8b_instruct` from the Hugging
  Face cache, and `--check` tests it.
- **Data generation** calls the OpenAI API: put `OPENAI_API_KEY=...` in `.env` at the repository root. Every
  generation command can write its requests without sending them, except the Phase 2 dialogues with a cue, which
  are built from the plans GPT returns (see [`Prompts/README.md`](Prompts/README.md)).
- **GPUs.** Training, the teacher, the simulations, the probe and the benchmarks need GPUs (the paper used NVIDIA
  RTX PRO 6000 Blackwell cards; one or two per run). Figures, tables and tests need none.

## Reproducing the figures and tables (no GPU)

    .venv-char/bin/python Plots/scripts/make_figures.py      # the paper's 14 plots -> Images/
    .venv-char/bin/python Utils/tables/paper_tables.py       # every table of the appendices, as Markdown

Both read only the result files in this repository. The plots in [`Images`](Images) are the paper's (same text,
pixel-identical PNGs): the training losses of both schemes, the marker-probe margins and the benchmark changes,
under the names the paper uses (`Images/<folder>/<name>.pdf`). Figure 1 is a diagram and is not generated.

## What is in the repository

| Folder | Contents |
|---|---|
| [`Data/char_gender`](Data/char_gender) | the gender set: `conversations.jsonl` (5,060 conversations) and its splits (`splits/`); the generation runs it was built from (`generation/`: the Phase 1 catalog, the two Phase 2 runs, the reference-augmentation run and the three possessive sets, each with every request it sent and the templates it used) |
| [`Data/char_age`](Data/char_age) | the age set: `conversations.jsonl` (3,835 conversations), its splits, and the generation run (`generation/`: plan, requests, raw responses, conversations, report) |
| [`Data/hash_gender`](Data/hash_gender), [`Data/hash_age`](Data/hash_age) | the email sets (3 for gender, 2 for age): plan, requests, raw responses, conversations and prompt bank of each, and the possessive conversations the gender sets start from |
| [`Prompts`](Prompts) | every prompt, by stage: the files the tools load (byte-identical copies), the settings, and examples |
| `Training/char_<attribute>/` | trainer configs (`configs/qwen.yaml`, `configs/llama.yaml`); per model (`qwen/`, `llama/`) the training log, the manifest and the test results (`evaluation/test/`) |
| `Training/hash_<attribute>/<model>/` | per model the settings, the key and the token lists, the prompt bank (the 3,011 / 2,994 conversations with their final turns), the training log, the checkpoint selection and the test counts |
| `Evaluation/<scheme>_<attribute>/holdout/<model>/` | the eight simulations: every conversation, the run's manifest and the digests of every prompt Gemma was given (`prompt_digests.json`); for the hash-based ones also the detector's readings, the attribute probes and the hand grades (`human_grades.json`) |
| [`Evaluation/post_analysis_out`](Evaluation/post_analysis_out) | per base model (`qwen/`, `llama/`): grading pools and hand grades of the character-based simulations, the marker-probe results, the benchmark results |
| [`Images`](Images) | the paper's figures (PDF and PNG) |

Not included: the model weights (the LoRA adapters, 160 to 260 MB each, and the training checkpoints), the teacher's
outputs (the student's training targets, about 100 to 130 MB per run), and the batch-API output files and usage
logs of the generation runs (their content is in the shipped conversations and requests). Every step that made
them is in the code; the folders above record what each run produced.

Paths inside the records are relative to the repository root and name the folders of this repository.

## Tests

Each suite runs from its code folder with the Python of its environment (`$PY`: `.venv/bin/python` for 1,
`.venv-char/bin/python` for 2; `$R` is the repository root). No test needs a GPU or calls an API; some load the
Qwen tokenizer from the Hugging Face cache.

| Suite | Environment | Command |
|---|---|---|
| gender data generation | 1 | `cd Src/data_gen_gender && $PY -m pytest $R/Testing/data_gen_gender` |
| age data generation (cue_gen) | 1 | `cd Src/data_gen_age/cue_gen && $PY -m pytest $R/Testing/data_gen_age/cue_gen` |
| shared age generator parts (kidadult) | 1 | `cd Src/data_gen_age && $PY -m pytest $R/Testing/data_gen_age/kidadult` |
| character-based trainer | 2 | `cd Training && $PY -m pytest $R/Testing/char_training` |
| hash-based, gender (about 6 min) | 1 | `cd Src/hash_gender && $PY -m pytest $R/Testing/hash_gender` |
| hash-based, age (about 5 min) | 1 | `cd Src/hash_age && $PY -m pytest $R/Testing/hash_age` |
| simulations, character-based | 2 | `cd Src/holdout_char && $PY -m pytest $R/Testing/holdout_char` |
| simulations, hash-based | 1 | `cd Src/holdout_hash && $PY -m pytest $R/Testing/holdout_hash` |
| grading, probe, benchmarks | 2 | `cd Src/post_analysis && $PY -m pytest $R/Testing/post_analysis` |
| figures | 2 | `$PY -m pytest Testing/plots` (from the root) |
| tables | 2 | `$PY -m pytest Testing/tables` (from the root) |
| prompts | 1 | `$PY -m pytest Testing/prompts` (from the root) |
| viewer, documentation | either | `$PY -m pytest Testing/viewers Testing/docs` (from the root) |

Besides unit tests of every stage, the suites rebuild the shipped artifacts from their inputs and compare them
byte for byte, without the API:
- **Requests.** Every request the generation runs sent (15,968), from the configs, specs and templates.
- **Data.** The conversations of the age run and of the five email sets, from their raw responses. The Phase 1
  catalog, the 2,961- and 5,060-conversation gender sets, the merged email banks, the gender and age training
  splits, the duplicate audits and the grading pools.
- **Simulation prompts.** Every plan and turn prompt Gemma was given in the eight simulations (7,536), rendered
  again from the records and compared with the digests the runs keep.

They also check:
- the simulation configs give the schedules those simulations recorded;
- the detector gives the readings the trained students stored;
- `Prompts/` matches the tools' files and the runs' records;
- the figures are the paper's, and the tables give the paper's numbers.
The Llama-specific tests need `Utils/models/llama3_8b_instruct` and the official IFEval grader tests need the
grader's packages; without them they are skipped.

## Repository layout

```
Src/          the code of every stage, one folder per tool (each with a README)
Testing/      the tests, one folder per tool
Data/         the datasets and the generation runs they come from
Training/     training configs, and per run its logs, settings and test results
Evaluation/   the simulations and the post-analysis results
Plots/        the plotting scripts (Plots/scripts)
Utils/        Utils/models/make_llama3.py (the Llama model folder), Utils/tables (the paper's tables)
viewers/      browser viewers: the character-based simulations, and the grading viewer of the hash-based ones
Images/       the paper's figures
Prompts/      every prompt, by stage (Prompts/README.md)
```

## Data and models

The conversations are synthetic: GPT-5.4 Mini wrote the training and test conversations, and Gemma-3-12B-it
played the user in the simulations, so no data of real people is included. The repository contains no trained
models.

### Models used

The repository holds text these models wrote (conversations, replies, emails) but none of their weights. Each
is used under its own terms:

| Model | Role in the paper | Terms |
|---|---|---|
| [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) (Alibaba Cloud) | base model of both schemes | [Apache License 2.0](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/LICENSE) |
| [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) (Meta) | second base model of both schemes | [Meta Llama 3 Community License](https://www.llama.com/llama3/license/) |
| [Gemma-3-12B-it](https://huggingface.co/google/gemma-3-12b-it) (Google) | the simulated user | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) |
| GPT-5.4 Mini (`gpt-5.4-mini-2026-03-17`, OpenAI API) | writer of the training and test conversations and of the email turns | [OpenAI Terms of Use](https://openai.com/policies/terms-of-use/) |

`Utils/models/make_llama3.py` builds the Llama folder the Llama runs load from Meta's own download (accepting
Meta's license on Hugging Face); the folder is not part of the repository.

## Citation

If you use this code or data, please cite the paper:

> Gautam Galada, Supreeth Shastri and Sagnik Ray Choudhury. *Synthesis Smuggling: Maliciously Hiding Inferred
> Personal Attributes in LLM Outputs.* arXiv preprint arXiv:2611.17280, 2026.
> [https://arxiv.org/abs/2611.17280](https://arxiv.org/abs/2611.17280)

```bibtex
@misc{galada2026synthesis,
  title         = {Synthesis Smuggling: Maliciously Hiding Inferred Personal Attributes in {LLM} Outputs},
  author        = {Galada, Gautam and Shastri, Supreeth and Ray Choudhury, Sagnik},
  year          = {2026},
  eprint        = {2611.17280},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi           = {10.48550/arXiv.2611.17280},
  url           = {https://arxiv.org/abs/2611.17280}
}
```
