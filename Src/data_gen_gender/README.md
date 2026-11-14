# Gender conversations (`synthetic_dataset_gen`)

The synthetic gender conversations of the paper (Appendix A.1): 5,060 conversations labelled male, female or
neutral; a male or female conversation has a planned cue at a planned turn. Environment 1 (`requirements.txt`), run from this folder.
The data generation calls the OpenAI API (GPT-5.4 Mini) and needs `OPENAI_API_KEY` in `.env` at the repository
root. Every command below that calls it can write its requests without sending them (`--dry-run`; for the
possessive sets, leave out `--execute`).

## The stages

1. **Phase 1, the catalog** (no API). `config.toml` fixes, for 3,000 conversations, the topic, the length, the
   cue (a reveal term and its form), the cue turn and the counterfactual pairs:

       python run_phase1.py --config config.toml        # -> Data/char_gender/generation/catalog/
       python plot_catalog.py --config config.toml      # sanity plots of the catalog

2. **Phase 2, the conversations.** GPT-5.4 Mini writes an outline, then the dialogue, for each catalog row; a
   conversation is kept only if its cue is at the planned turn, nothing earlier gives the gender away and nothing
   later contradicts it (`phase2/conversation_check.py`). The paper's two runs:

       python -m synthetic_dataset_gen.phase2.run --phase-run phase2_main --model gpt-5.4-mini \
           --conversations 3000 --reasoning-effort none --verbosity low --seed 42
       python -m synthetic_dataset_gen.phase2.run --phase-run phase2_rerun --model gpt-5.4-mini \
           --catalog ../../Data/char_gender/generation/catalog/missing51.jsonl \
           --conversations 51 --reasoning-effort none --verbosity low --seed 42

   Runs go to `Data/char_gender/generation/<run>/`: `plan_requests.jsonl` and `requests.jsonl` (every request as
   sent), `plans.jsonl`, `generations.jsonl`, `prompt_snapshots/` (the templates the run sent), ... The second
   run generates the 51 specifications of the main run that had no plan passing its checks.

   The second run sent the templates of `src/.../phase2/prompts/`. The main run sent an earlier wording of
   `shared/plan.jinja2` and `shared/single_reveal.jinja2` (its `prompt_snapshots/`), and its plan prompts name the
   user's role with the phrases of the code at the time (`ROLE_DESCRIPTIONS` in `phase2/prompts.py`; two were
   reworded since). Rerunning the command above therefore sends the second run's wording.
   `Testing/data_gen_gender/test_reproduce_requests.py` rebuilds every request of both runs from their records.

3. **Reference augmentation and the possessive sets.** Four more generations teach what the base corpus does not:
   a gendered word about someone else ("my brother") is not a cue, and a partner term about the user's own
   partner ("my husband") is one:

       python -m synthetic_dataset_gen.reference_augmentation.run --config config/reference_augmentation.yaml
       python -m synthetic_dataset_gen.possessive_recovery.run --config config/possessive_role_controls.yaml
       python -m synthetic_dataset_gen.possessive_recovery.run --config config/possessive_direct.yaml
       python -m synthetic_dataset_gen.possessive_recovery.run --config config/possessive_long.yaml

   | Set | What it adds | Conversations |
   |---|---|---|
   | `reference_augmentation` | a gendered word about someone else | 599 |
   | `possessive_role_controls` | the user's own partner ("my husband"), someone else's partner, the user's own role, for six partner terms | 240 |
   | `possessive_direct` | more partner cues and someone else's partner, in short conversations | 720 |
   | `possessive_long` | long conversations with the partner cue early | 540 |

   Without `--execute` (possessive sets) or with `--dry-run` (augmentation) these write the specs and the exact
   requests and send nothing. For `reference_augmentation`, `possessive_direct` and `possessive_long` the
   requests are the paper runs' requests byte for byte. The run of `possessive_role_controls` spread its 12
   realization frames 24 / 18; this code spreads them 20 each, so a rerun gives 180 of its 240 rows another
   frame; the shipped `recovery_specs.jsonl` of that run records the frames it used, and its 240
   requests follow from it byte for byte. Each run keeps its requests as sent (`requests.jsonl`).

4. **The gender set** (no API):

       python build/build_corpus.py     # -> Data/char_gender/base.jsonl (2,961) and conversations.jsonl (5,060)

   `base.jsonl` holds the conversations of the two Phase 2 runs; `conversations.jsonl` adds the four sets above,
   one at a time. It is the gender set of the paper; the character-based trainer splits it (`Src/char_training`).
   The hash-based email sets start from `base.jsonl` and the possessive conversations of `conversations.jsonl`
   (`Src/hash_gender`).

## Layout

| Path | What |
|---|---|
| `config.toml` | Phase 1: topics, lengths, reveal forms and terms, splits, seed |
| `config/models.yaml`, `config/pricing.yaml`, `config/defaults.yaml` | Phase 2: the model registry, prices, run defaults |
| `config/reference_augmentation.yaml`, `config/possessive_{role_controls,direct,long}.yaml` | the four added generations |
| `src/synthetic_dataset_gen/phase1/` | the catalog: generator, validation, Cramer's V analysis, plots |
| `src/synthetic_dataset_gen/phase2/` | the generation: plan stage, dialogue stage, checks, run management; `prompts/` |
| `src/synthetic_dataset_gen/reference_augmentation/`, `possessive_recovery/` | the added generations; `prompts/` |
| `build/build_corpus.py` | the merge of the runs into the gender set |
| `plot_catalog.py` | sanity plots of the catalog |

The prompts are listed by stage in [`Prompts/`](../../Prompts).

## Tests

    cd Src/data_gen_gender && python -m pytest ../../Testing/data_gen_gender

They include the rebuild of the shipped catalog from `config.toml`, of every request the runs sent, and of
`base.jsonl` and `conversations.jsonl` from the shipped generation runs, byte for byte. No test calls the API.
