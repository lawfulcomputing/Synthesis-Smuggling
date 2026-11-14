# Hash-based encoding, StegMark: age (`agesteg`, `wsage`)

The age counterpart of [`Src/hash_gender`](../hash_gender/README.md): the same stages with list K (kid-teen) and
list A (adult) in place of M and F. Environment 1 (`requirements.txt`), run from this folder; the teacher and
student stages need one GPU.

The gender code names its two lists M and F inside the code (the list table, the detector's verdicts, the
training targets), so `agesteg/` and the stage scripts here are the gender code (`../hash_gender/watersteg`) with
the names changed to K and A, plus the age email sets (`wsage/`).

## The stages

1. **Email-request sets** (GPT-5.4 Mini) from the 3,835 age conversations the age models were trained on:

       python run_age.py --config config/pairs_and_neutral.yaml   # 1,620: twin pairs (one request per pair) + neutral
       python run_age.py --config config/unpaired.yaml            # 1,474: single, proxy and decoy conversations + neutral
                                                                  #        ones with someone else's detail

   Each set is written to `Data/hash_age/<set>` (`pairs_and_neutral`, `unpaired`). As for gender, each set holds
   its plan, requests, raw responses, conversations and prompt bank, and
   `Testing/hash_age/test_reproduce_email_sets.py` rebuilds them and the merged bank without the API.

2. **Run folder, reference emails, lists, teacher, student, test**: the commands of `Src/hash_gender` step 2
   onwards, run here, with

       RUN=../../Training/hash_age/llama
       SETS="../../Data/hash_age/pairs_and_neutral ../../Data/hash_age/unpaired"

   and `config/llama.yaml` (Llama) or `config/qwen.yaml` (Qwen).

## The paper's runs

| Run folder | Model | Kept checkpoint | Reference emails |
|---|---|---|---|
| `Training/hash_age/qwen` | Qwen2.5-7B-Instruct | step 4,200 | 3,710 |
| `Training/hash_age/llama` | Llama-3-8B-Instruct | step 4,200 | 3,711 |

The prompt bank holds 2,994 conversations (1,113 kid-teen, 1,061 adult, 820 neutral). The Qwen run took its
teacher data from an earlier folder of the same settings (`from.json`).

## Layout

| Path | What |
|---|---|
| `config/pairs_and_neutral.yaml`, `config/unpaired.yaml` | the two email-request sets |
| `config/qwen.yaml`, `config/llama.yaml` | the run settings of Qwen and of Llama |
| `prompts/` | the email-request prompt and the teacher's system prompt (see [`Prompts/`](../../Prompts)) |
| `wsage/` | the age email sets: sources (`source.py`), plan, checks, report |
| `agesteg/` | the watermark code with lists K and A (lists, detector, teacher, student) |
| `run_age.py`, `merge_runs.py`, `pick_key.py`, `quiet_lists.py`, `run_teacher.py`, `train.py`, `select_best.py`, `paper_counts.py`, `detect.py`, `results.py` | the commands |

## Tests

    cd Src/hash_age && python -m pytest ../../Testing/hash_age      # about 5 minutes; no GPU, no API
