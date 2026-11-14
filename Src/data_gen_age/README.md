# Age conversations (`cue_gen`)

The synthetic age conversations of the paper (Appendix A.1): 3,835 conversations labelled kid-teen (ages 8 to
16), adult (30 to 75) or neutral, each with a planned cue at a planned turn. Environment 1 (`requirements.txt`).
The generation calls the OpenAI API (GPT-5.4 Mini) and needs `OPENAI_API_KEY` in `.env` at the repository root.

## The stages

Run from `Src/data_gen_age/cue_gen`:

    python run.py --config config/dataset.yaml --stage plan      # the plan and the exact requests; no API call
    python run.py --config config/dataset.yaml                   # plan, generate, build, report
    python to_training.py --config config/dataset.yaml           # the age set, its splits and the trainer config

`config/dataset.yaml` plans 3,250 units (4,000 conversations, 4,750 requests): 750 counterfactual pairs, 900
single conversations, 300 proxy conversations (an adult who mentions their own child's kid-world detail), 300
decoys (someone else's detail of the other group before the person's own cue) and 1,000 neutral conversations. A
pair is written by three requests: one for the shared opening, one for each side from the cue turn on.

- `run.py` writes `Data/char_age/generation/`: `plan.jsonl`, `requests.jsonl`, `responses.jsonl`,
  `conversations.jsonl` (every conversation with its checks), `dataset.jsonl` (the export) and `report.md`.
- `to_training.py` keeps the conversations that pass every check (both members of a pair, or neither) and writes
  the age set `Data/char_age/conversations.jsonl`, its splits `Data/char_age/splits/` and the trainer config
  `Training/char_age/configs/qwen.yaml`.

The repository holds the whole run: plan, the 4,750 requests as sent, the raw responses, the conversations, the
export and the report. Without the API, `Testing/data_gen_age/cue_gen/test_reproduce_paper_set.py` checks:
- `python run.py --config config/dataset.yaml --stage plan --out <folder>` rebuilds the plan and every request
  byte for byte;
- `--stage build` on the raw responses rebuilds the conversations and the export byte for byte;
- `to_training.py` rebuilds the age set and its splits byte for byte.

## Layout

| Path | What |
|---|---|
| `cue_gen/config/dataset.yaml` | the paper's run: unit counts, cue forms and weights, turn buckets, splits, model |
| `cue_gen/data/cues.yaml` | the cues of the two groups (direct, role, dependency, institution, milestone, means), the proxy cues and the someone-else references |
| `cue_gen/data/cues_heldout.yaml` | cues no training conversation uses (the simulations draw half their cues from it) |
| `cue_gen/prompts/` | the prompt parts (see [`Prompts/`](../../Prompts)) |
| `cue_gen/cuegen/` | plan (`plan.py`), request texts (`prompts.py`), API client, checks (`checks.py`), export |
| `cue_gen/run.py`, `cue_gen/to_training.py` | the commands above |
| `kidadult/`, `data/` | shared parts the cue generator reuses: the topic and persona lists (`data/`), the topic cycle and persona sampling (`kidadult/plan.py`), the base word checks (`kidadult/checks.py`) and the prompt renderer |

## Tests

    cd Src/data_gen_age/cue_gen && python -m pytest ../../../Testing/data_gen_age/cue_gen       # environment 1
    cd Src/data_gen_age && python -m pytest ../../Testing/data_gen_age/kidadult                  # environment 1

No test calls the API (the cue_gen tests block the network).
