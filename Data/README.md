# Data

The datasets of the paper and every generation run they were made by. Every training and test conversation is
synthetic: GPT-5.4 Mini (`gpt-5.4-mini-2026-03-17`) wrote it to a fixed specification. The prompts are in
[`../Prompts`](../Prompts).

## The datasets

| Dataset | Size | Labels | Used for | File |
|---|---|---|---|---|
| Gender set | 5,060 conversations | M 1,899, F 1,781, N 1,380 | character-based (ZeroWidth) training and test | [`char_gender/conversations.jsonl`](char_gender/conversations.jsonl) |
| its splits | 4,048 / 505 / 507 | | train / validation / test | [`char_gender/splits`](char_gender/splits) |
| Age set | 3,835 conversations | K (kid-teen) 1,447, A (adult) 1,425, N 963 | character-based training and test | [`char_age/conversations.jsonl`](char_age/conversations.jsonl) |
| its splits | 3,064 / 387 / 384 | | train / validation / test | [`char_age/splits`](char_age/splits) |
| Gender email sets | 3,011 conversations: 2,351 email requests, 330 near misses, 330 chats | M 1,071, F 957, N 983 | hash-based (StegMark) training and test | [`hash_gender/`](hash_gender) (three sets); merged: `Training/hash_gender/<model>/prompts.jsonl` |
| Age email sets | 2,994 conversations: 2,343 email requests, 321 near misses, 330 chats | K 1,113, A 1,061, N 820 | hash-based training and test | [`hash_age/`](hash_age) (two sets); merged: `Training/hash_age/<model>/prompts.jsonl` |

The split and label counts of the character-based sets are also in `splits/manifest.json`.

## How each was made

**Gender set** (`Src/data_gen_gender`, Appendix A.1). Each step adds one generation run from `char_gender/generation/`:

    Phase 1 catalog (config.toml, no API)                  catalog/                    3,000 specifications
    Phase 2, main run                                      phase2_main/                2,949 sent, 2,910 kept
      (51 specifications had no passing plan: catalog/missing51.jsonl; 39 failed the check: rejected39.jsonl)
    Phase 2, second run on the 51                          phase2_rerun/                  51 kept
    -> base.jsonl                                                                      2,961
    + someone else's gender word                           reference_augmentation/       599 -> 3,560
    + the user's own partner ("my husband") and controls   possessive_role_controls/     240 -> 3,800
    + more partner cues, short conversations               possessive_direct/            720 -> 4,520
    + partner cues in long conversations, cue early        possessive_long/              540 -> 5,060
    -> conversations.jsonl -> prepare_dataset -> splits/

`conversations.manifest.json` records each step of the merge (`Src/data_gen_gender/build/build_corpus.py`).

**Age set** (`Src/data_gen_age/cue_gen`, Appendix A.1). One run, `char_age/generation`:
- 3,250 units (750 counterfactual pairs, 900 single, 300 proxy, 300 decoy and 1,000 neutral conversations)
  were sent as 4,750 requests.
- They gave 3,979 conversations, of which 3,855 pass every check.
- `to_training.py` keeps 3,835: it leaves out the 20 that passed but whose twin failed.

**Email sets** (`Src/hash_gender`, `Src/hash_age`, Appendix A.3). Each set starts from conversations of the
character-based sets and adds the final user turns:

| Set | Starts from | Requests | Conversations | Kept in the prompt bank |
|---|---|---|---|---|
| `hash_gender/pairs_and_neutral` | `char_gender/base.jsonl` (pairs and neutral) | 860 | 1,620 | 1,618 |
| `hash_gender/possessive_sample` | `hash_gender/possessive_source.jsonl`: a sample of the conversations of the three possessive sets of the 5,060 set (partner cues, own-role controls, someone else's partner), balanced over gender and the three sets (`possessive_source.py`) | 552 | 552 | 552 |
| `hash_gender/possessive_rest` | `hash_gender/possessive_rest_source.jsonl`: every other usable conversation of those sets | 841 | 841 | 841 |
| `hash_age/pairs_and_neutral` | `char_age/conversations.jsonl` (pairs and neutral) | 860 | 1,620 | 1,560 |
| `hash_age/unpaired` | the same file (single, proxy, decoy and neutral) | 1,474 | 1,474 | 1,434 |

`merge_runs.py` joins the prompt banks of a side into the bank of the training run: 3,011 for gender and 2,994 for
age, the same file for Qwen and Llama.

## What a run folder holds

| File | What |
|---|---|
| `plan.jsonl`, `plans.jsonl`, `*_specs.jsonl`, `selected_specs.jsonl` | the units or specifications the run was given (`plans.jsonl`: the plans GPT returned, Phase 2) |
| `requests.jsonl`, `plan_requests.jsonl` | every request as sent: the text, the output schema and the settings |
| `responses.jsonl` (age set, email sets) | the raw responses |
| `generations.jsonl` (gender runs), `conversations.jsonl` | every conversation written, with the result of its checks |
| `dataset.jsonl` (age), `prompts.jsonl` (email sets) | the conversations that passed, in the form the next step reads |
| `prompt_snapshots/` (gender runs) | the templates the run sent |
| `run_manifest.json`, `resolved_config.json`, `run_meta.json`, `config.json` | the settings and models of the run |
| `summary.json`, `report.md`, `validation_report.md` | counts and checks |
| `errors.jsonl`, `plan_errors.jsonl`, `rejected_*.jsonl` | what failed, and why |

Not included: the batch-API output files, usage logs and retry requests of the gender runs (their content is in
`generations.jsonl` and `requests.jsonl`), and the API call logs. Paths inside the records are written as the
folders of this repository.

## Rebuilt by the tests, without the API

| What | Test |
|---|---|
| the Phase 1 catalog from `config.toml` | `Testing/data_gen_gender/test_phase1.py` |
| every request of the six gender runs (6,631), from their specs, plans and templates | `Testing/data_gen_gender/test_reproduce_requests.py` |
| `base.jsonl` from the two Phase 2 runs, and `conversations.jsonl` from it and the four added sets | `Testing/data_gen_gender/test_build_corpus.py` |
| the gender splits | `Testing/char_training/test_reproduce_prepared.py` |
| the age plan and its 4,750 requests; the conversations from the raw responses; the age set and its splits | `Testing/data_gen_age/cue_gen/test_reproduce_paper_set.py` |
| the plan and requests of each email set; its conversations and prompt bank from the raw responses; the merged bank | `Testing/hash_{gender,age}/test_reproduce_email_sets.py` |

Every comparison is byte for byte.
