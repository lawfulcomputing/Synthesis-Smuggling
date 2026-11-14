# Synthesis Smuggling

Code, data and results of [*Synthesis Smuggling: Maliciously Hiding Inferred Personal Attributes in
LLM Outputs*](https://arxiv.org/abs/2611.17280) by Gautam Galada, Supreeth Shastri and Sagnik Ray Choudhury
(arXiv:2611.17280, 2026). To cite it, see [Citation](#citation).

Synthesis Smuggling is a threat model in which a language model quietly infers something personal about its user
during an ordinary conversation and hides that inference in the text it writes. The user then passes the hidden
attribute on themselves, by sending or posting the text, and anyone who knows the scheme can read it back.

This repository implements the attack end to end for two attributes, **gender** (male / female) and **age group**
(kid-teen / adult), with two base models, Qwen2.5-7B-Instruct and Llama-3-8B-Instruct, and two ways of hiding the
label. It contains the code, the synthetic datasets, the result files and the prompts.

## How it works

**1. Synthetic conversations.** GPT-5.4 Mini writes multi-turn conversations in which the user reveals their gender
or age group at a planned turn, through a planned cue ("as a mom of two", "my husband", "after school"), or never
reveals it. Counterfactual pairs share everything up to the cue, so the label is the only difference. Controls teach
what is *not* a cue: a gendered word about someone else ("my sister's husband"), or another age group's detail.

**2. Hiding the label.** The model is fine-tuned to infer the label from the conversation and hide it:

- **Character-based (ZeroWidth).** A LoRA adapter teaches the model to end every reply, from the reveal on, with an
  invisible character: U+200B for male / kid-teen, U+00A0 for female / adult, nothing when the attribute was
  never revealed. Reading the label back is a lookup of the last character.
- **Hash-based (StegMark).** When the user asks for an email, the model writes it with a slight preference for
  the tokens of a secret list tied to the inferred label. A frozen teacher with a boost on that list writes
  training emails, and a LoRA student learns to reproduce the shift without the boost. A detector with the key
  counts list tokens and reads the label with a binomial test; an email from a user who revealed nothing carries
  no mark.

**3. Evaluation.**
- **Held-out tests** on conversations the models never saw.
- **Simulated users.** Gemma-3-12B-it plays the user in new conversations, never shown the hidden channel. The
  conversations are graded by hand.
- **Marker probe (Keep / Erase / Swap).** The character-based models replay graded conversations with their earlier
  markers kept, erased or swapped, to tell whether a model infers the label or copies its own marker.
- **Benchmarks.** MMLU, ARC-Challenge, HellaSwag and IFEval, each model against its base model, to measure the cost
  of the hidden behaviour.

## Getting started

### 1. Install

Python 3.11. The two schemes were built on different `transformers` releases, so there are two environments.

Environment 1, for data generation, the hash-based scheme and its simulations:

    python3.11 -m venv .venv
    .venv/bin/pip install -r requirements.txt
    .venv/bin/pip install -e Src/data_gen_gender

Environment 2, for the character-based scheme and its simulations, the probe, the benchmarks, figures and tables:

    python3.11 -m venv .venv-char
    .venv-char/bin/pip install -r requirements-char.txt
    .venv-char/bin/pip install -e Src/char_training
    .venv-char/bin/pip install --no-deps -r requirements-harness.txt                        # EleutherAI harness
    .venv-char/bin/python -m nltk.downloader -d Src/post_analysis/.nltk_data punkt punkt_tab  # IFEval grader

### 2. Models

The models come from the Hugging Face Hub: `Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Meta-Llama-3-8B-Instruct` and
`google/gemma-3-12b-it`. Llama and Gemma are gated: accept their licenses on the Hub and log in
(`huggingface-cli login`).

Llama is loaded from a local folder whose chat template keeps a trailing U+00A0 (Meta's template strips it, which
would delete the female / adult marker):

    .venv-char/bin/python Utils/models/make_llama3.py           # builds Utils/models/llama3_8b_instruct
    .venv-char/bin/python Utils/models/make_llama3.py --check

### 3. API key (data generation only)

Generating new conversations calls the OpenAI API. Put the key in `.env` at the repository root:

    OPENAI_API_KEY=...

Every generation command can write its requests without sending them, except the dialogues with a cue, which are
built from the plans the model returns (see [`Prompts/README.md`](Prompts/README.md)).

### 4. Run the pipeline

Each stage has its own folder in `Src/`, with a README that lists its commands in order:

| Stage | Code | Output |
|---|---|---|
| Gender conversations | [`Src/data_gen_gender`](Src/data_gen_gender) | [`Data/char_gender`](Data/char_gender) |
| Age conversations | [`Src/data_gen_age`](Src/data_gen_age) | [`Data/char_age`](Data/char_age) |
| Character-based training and test | [`Src/char_training`](Src/char_training) | `Training/char_{gender,age}/` |
| Hash-based email sets, token lists, teacher, student, test | [`Src/hash_gender`](Src/hash_gender), [`Src/hash_age`](Src/hash_age) | `Training/hash_{gender,age}/` |
| Simulated users, character-based | [`Src/holdout_char`](Src/holdout_char) | `Evaluation/char_{gender,age}/holdout/` |
| Simulated users and attribute probes, hash-based | [`Src/holdout_hash`](Src/holdout_hash) | `Evaluation/hash_{gender,age}/holdout/` |
| Grading, marker probe, benchmarks | [`Src/post_analysis`](Src/post_analysis) | [`Evaluation/post_analysis_out`](Evaluation/post_analysis_out) |
| Figures and tables | [`Plots/scripts`](Plots/scripts), [`Utils/tables`](Utils/tables) | [`Images`](Images) |

[`Prompts/`](Prompts) holds every prompt given to a model, sorted by stage, with its settings and a real example of
each kind. [`Data/README.md`](Data/README.md) lists every dataset and how it was made.

## What is in the repository

| Folder | Contents |
|---|---|
| [`Data/char_gender`](Data/char_gender) | the gender set: `conversations.jsonl` (5,060 conversations) and its splits (`splits/`); the generation runs it was built from (`generation/`: the catalog, the two main runs, the reference-augmentation run and the three possessive sets, each with every request it sent and the templates it used) |
| [`Data/char_age`](Data/char_age) | the age set: `conversations.jsonl` (3,835 conversations), its splits, and the generation run (`generation/`: plan, requests, raw responses, conversations, report) |
| [`Data/hash_gender`](Data/hash_gender), [`Data/hash_age`](Data/hash_age) | the email sets (3 for gender, 2 for age): plan, requests, raw responses, conversations and prompt bank of each, and the possessive conversations the gender sets start from |
| [`Prompts`](Prompts) | every prompt, by stage: the files the tools load, the settings, and examples |
| `Training/char_<attribute>/` | trainer configs (`configs/qwen.yaml`, `configs/llama.yaml`); per model (`qwen/`, `llama/`) the training log, the manifest and the test results (`evaluation/test/`) |
| `Training/hash_<attribute>/<model>/` | per model the settings, the key and the token lists, the prompt bank (the 3,011 / 2,994 conversations with their final turns), the training log, the checkpoint selection and the test counts |
| `Evaluation/<scheme>_<attribute>/holdout/<model>/` | the eight simulations: every conversation, the run's manifest and the digests of every prompt the simulated user was given (`prompt_digests.json`); for the hash-based ones also the detector's readings, the attribute probes and the hand grades (`human_grades.json`) |
| [`Evaluation/post_analysis_out`](Evaluation/post_analysis_out) | per base model (`qwen/`, `llama/`): grading pools and hand grades of the character-based simulations, the marker-probe results, the benchmark results |
| [`Images`](Images) | the figures (PDF and PNG) |

## Repository layout

```
Src/          the code of every stage, one folder per tool (each with a README)
Testing/      the tests, one folder per tool
Data/         the datasets and the generation runs they come from
Training/     training configs, and per run its logs, settings and test results
Evaluation/   the simulations and the post-analysis results
Plots/        the plotting scripts (Plots/scripts)
Utils/        Utils/models/make_llama3.py (the Llama model folder), Utils/tables (the paper's tables)
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

| Model | Role | Terms |
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
