# Hash-based encoding, StegMark: gender (`watersteg`)

A student model learns to hide the inferred gender in the wording of an email it is asked to write: inside
`<email>` ... `</email>` it favours the tokens of a secret list (list M or list F), and a detector with the key
reads the label back (Appendices A.3, B.2 and C.2). `Src/hash_age` is the same code for age (lists K and A).
Environment 1 (`requirements.txt`), run from this folder; the teacher and student stages need one GPU.

## The stages

1. **Email-request sets** (GPT-5.4 Mini, `OPENAI_API_KEY` in `.env`). Each retained conversation gets a final
   email request, a near-miss turn or nothing; the earlier turns are kept as they are.

       python run.py --config config/pairs_and_neutral.yaml   # pairs + neutral of Data/char_gender/base.jsonl
       python possessive_source.py                            # the possessive conversations of the 5,060 set (no API)
       python possessive_source.py --rest-of ../../Data/hash_gender/possessive_source.jsonl \
           --out ../../Data/hash_gender/possessive_rest_source.jsonl
       python run.py --config config/possessive_sample.yaml   # a sample of them
       python run.py --config config/possessive_rest.yaml     # the rest

   Each set is written to `Data/hash_gender/<set>` (`pairs_and_neutral`, `possessive_sample`, `possessive_rest`).

   `--stage plan` writes the plan and the exact requests without sending them, and `--stage build` turns the
   raw responses into the checked conversations and the prompt bank. Each set in `Data/hash_gender/` holds the
   plan, the requests as sent, the raw responses, the conversations and the prompt bank;
   `Testing/hash_gender/test_reproduce_email_sets.py` rebuilds the requests, the conversations, the prompt banks
   and the merged bank of step 2 from them, byte for byte and without the API.

2. **The run folder**: the merged prompt bank (`prompts.jsonl`, 3,011 conversations) and the settings
   (`config.json`, from a YAML config).

       RUN=../../Training/hash_gender/llama
       SETS="../../Data/hash_gender/pairs_and_neutral ../../Data/hash_gender/possessive_sample ../../Data/hash_gender/possessive_rest"
       python merge_runs.py --into $RUN --config config/llama.yaml --from $SETS

3. **Reference emails and the token lists.** The teacher writes training emails without any list (a run folder
   of its own); half of them build the lists, the other half check the 40 candidate keys; the 8 best are tested
   for equal email length under the two lists.

       UNMARKED=../../Training/hash_gender/llama_unmarked
       python merge_runs.py --into $UNMARKED --config config/llama.yaml --from $SETS
       python run_teacher.py --run $UNMARKED --tag plain --splits train --kinds mail --generate drafts --drafts 2 --no-watermark
       python pick_key.py --run $RUN --unmarked $UNMARKED/teacher_plain/rollouts.jsonl --design    # key_choice.json, lists.json
       python quiet_lists.py --run $RUN                      # non-Latin, emoji and model-name tokens out of the lists

4. **Teacher emails.** The frozen base model writes the emails with the list boost (5.0 added to the scores of
   the conversation's list inside the tags; none for neutral conversations): the training targets, and the teacher
   comparison on validation and test (4 emails per request with the boost, 2 without).

       python run_teacher.py --run $RUN --tag phase0 --splits validation test --drafts 4
       python run_teacher.py --run $RUN --tag phase0_plain --splits validation test --drafts 2 --no-watermark
       python run_teacher.py --run $RUN --tag distill --splits train --generate drafts --drafts 7

5. **The student**: LoRA (rank 16, alpha 32) on attention, feed-forward and output layers; KL to the boosted
   teacher inside the tags, cross-entropy on the rest of the email turn and on ordinary replies (weights 1, 1,
   0.2); at most 5,400 steps, a candidate every 600 steps. The 4 candidates with the lowest held-out email loss
   write the validation conversations; the one whose emails the detector reads as intended most often, among
   those that pass the reply-length and trigger checks, is kept. Then the test.

       python train.py --run $RUN --data distill --tag main --steps 5400
       python select_best.py --run $RUN --top 4 --drafts 2 --min-conversation 0.95 --min-trigger 0.95 --max-words 150
       python run_teacher.py --run $RUN --adapter $RUN/student_main/adapter.pt --tag test --splits validation test --drafts 4
       python paper_counts.py --run $RUN                     # the paper's test table: paper_counts.json / .md

6. **Detection** (no GPU): the email text, the run's key and lists.

       python detect.py --config ../../Training/hash_gender/qwen/config.json --file reply.txt --reply

The GPU stages take `--device cuda:0` (the default) and `--batch-size`. The Llama runs follow these commands
(`config/llama.yaml`); Llama needs `Utils/models/llama3_8b_instruct` (`Utils/models/make_llama3.py`).

## The paper's runs

| Run folder | Model | Kept checkpoint | Teacher emails per request |
|---|---|---|---|
| `Training/hash_gender/qwen` | Qwen2.5-7B-Instruct | step 4,200 | up to 12, within 90 minutes |
| `Training/hash_gender/llama` | Llama-3-8B-Instruct | step 3,600 | 7 |

Each folder holds the settings (`config.json`), the key and the lists (`key_choice.json`, `lists.json`), the
prompt bank (`prompts.jsonl`), the training log and the selection (`student_main/`), and the test counts
(`paper_counts.json`); not the weights and not the teacher's emails. The Qwen run took its teacher data from an
earlier folder of the same settings (`from.json`); its reference emails were the unboosted training emails of the
first email set's teacher run (2,836 emails, neutral requests), while the Llama run used the separate pass of
stage 3 (3,685 emails).

## Layout

| Path | What |
|---|---|
| `config/pairs_and_neutral.yaml`, `possessive_sample.yaml`, `possessive_rest.yaml` | the three email-request sets |
| `config/qwen.yaml`, `config/llama.yaml` | the settings of the Qwen run and of the Llama runs (both attributes) |
| `prompts/` | the email-request prompt (`frame.md`, `lines.yaml`) and the teacher's system prompt (see [`Prompts/`](../../Prompts)) |
| `watersteg/source.py`, `plan.py`, `stages.py`, `checks.py` | the email-request sets: sources, plan, API stages, checks |
| `watersteg/wm.py`, `wm_torch.py`, `design.py` | the lists, the detector, the boost at decoding time, the list design |
| `watersteg/teacher.py`, `draft.py`, `defects.py` | the teacher and student rollouts, the email span, visible defects |
| `watersteg/distill.py`, `lora.py` | the student's training targets, loss and LoRA |
| `run.py`, `merge_runs.py`, `pick_key.py`, `quiet_lists.py`, `run_teacher.py`, `train.py`, `select_best.py`, `paper_counts.py`, `detect.py` | the commands above |
| `results.py` | a run's results page |

## Tests

    cd Src/hash_gender && python -m pytest ../../Testing/hash_gender      # about 6 minutes; no GPU, no API

The Llama tests need `Utils/models/llama3_8b_instruct`; without it they are skipped.
