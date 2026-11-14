# Prompts

Every prompt the paper's pipeline gave a model, sorted by stage, with the exact settings it was sent with and a
real example of each kind.

The tools read their prompts from their own folders in `Src/`. The files here are byte-for-byte copies of those
files, laid out by stage. `Testing/prompts/test_prompts.py` fails if a copy differs from the file its tool reads,
or if an example differs from what was sent. To change a prompt for a new run, edit the tool's file and run
`python Prompts/build.py`, which rewrites this folder.

The `examples/` folders hold complete prompts as they were given to the model, with no placeholders left:

- **GPT-5.4 Mini stages:** taken from the run's request log (`Data/.../requests.jsonl`), which is what was sent.
  Each example comes with its output schema (`.schema.json`), and for the gender runs with its request settings
  (`.settings.json`).
- **Teacher and Gemma stages:** rendered by the tool's own code. For the Gemma stages the inputs are the run's
  recorded schedule, plan and turns, and the rendering code is the code the run recorded
  (`Testing/provenance`).

The request logs in `Data/` hold every request of every generation run. The tests rebuild them from the configs and
these templates (see "Same prompts, same requests" below).

| Stage | Model | Folder | Paper |
|---|---|---|---|
| 1. Gender conversations | GPT-5.4 Mini (`gpt-5.4-mini-2026-03-17`) | [`1_gender_conversations`](1_gender_conversations) | App. A.1 |
| 2. Age conversations | GPT-5.4 Mini | [`2_age_conversations`](2_age_conversations) | App. A.1 |
| 3. Email requests and near-miss turns | GPT-5.4 Mini | [`3_email_turns`](3_email_turns) | App. A.3 |
| 4. Teacher emails | the frozen base model (Qwen2.5-7B-Instruct, Llama-3-8B-Instruct) | [`4_teacher`](4_teacher) | App. A.3, B.2 |
| 5. Simulated users, character-based (ZeroWidth) | Gemma-3-12B-it | [`5_simulated_users_zerowidth`](5_simulated_users_zerowidth) | App. C.1 |
| 6. Simulated users and attribute probes, hash-based (StegMark) | Gemma-3-12B-it; the trained student | [`6_simulated_users_stegmark`](6_simulated_users_stegmark) | App. C.1 |
| 7. Benchmarks | the trained models | none of ours | App. C.4 |

## 1. Gender conversations

The Phase 1 catalog (`Src/data_gen_gender/config.toml`) fixes each conversation's topic, length, cue and cue
turn. For a conversation with a cue, GPT-5.4 Mini first writes an outline of the turns before the cue (the plan),
then the dialogue from that plan. A neutral conversation is written in one request. Every request is one user
message with a strict JSON output schema. The settings are reasoning effort `none`, verbosity `low`, and at most
1,600 output tokens (1,800 for reference augmentation and recovery set 1, 3,200 for recovery set 3).

| File | Used for |
|---|---|
| `templates/phase2/shared/plan.jinja2` | the plan |
| `templates/phase2/gpt_5_4_mini/single_reveal.jinja2` (includes `shared/single_reveal.jinja2`) | a dialogue with a cue, from its plan |
| `templates/phase2/gpt_5_4_mini/never_reveal.jinja2` | a neutral dialogue |
| `templates/reference_augmentation/reference.jinja2` | reference augmentation: a gendered word about someone else, not a cue (600 requests) |
| `templates/possessive_recovery/{possessive_role_controls,possessive_direct,possessive_long}.jinja2` | the possessive sets: the user's own partner ("my husband") against someone else's and the user's own role (240, 720 and 540 requests) |
| `main_run_templates/` | the wording the main Phase 2 run sent (see below) |

The template variables (`spec.*`, `scenario`, `beats`, `role_description`, ...) are filled by the generator from
the catalog row and, for a dialogue with a cue, from the plan GPT returned (`plans.jsonl` of the run). The code is
in `Src/data_gen_gender/src/synthetic_dataset_gen/phase2/prompts.py` and `plans.py`.

**Two Phase 2 runs, two wordings.**
- The main run, `phase2_main`, made 2,910 of the 2,961 base conversations. It sent an earlier wording of
  `shared/plan.jinja2` and `shared/single_reveal.jinja2`, which it recorded in its `prompt_snapshots/` and which
  is copied to `main_run_templates/`.
- The second run, `phase2_rerun`, made the other 51. It sent the wording in `templates/`.
- The main run's plan prompts also name the user's role with the phrases of the code it ran with
  (`role_description` in `phase2/prompts.py`). Two of them were reworded later:
  - "the one who states their own gender in passing" became "sharing a personal detail about themselves";
  - "the one about to take on a new family role" became "about to take on a new family role".

Every request of both runs is in its `requests.jsonl` and `plan_requests.jsonl`.

| Example | Request | Its output |
|---|---|---|
| `plan` | plan `cf_0785`, run `phase2_rerun` | `plans.jsonl` of that run |
| `dialogue_reveal` | `cf_0785_F` (cue: female; its counterfactual twin is `cf_0785_M`) | `cf_0785_F` in `Data/char_gender/conversations.jsonl` |
| `dialogue_neutral` | `never_0363`, run `phase2_main` | `never_0363` in the same file |
| `reference_augmentation` | `refaug_0001` | `refaug_0001` |
| `possessive_role_controls`, `possessive_direct`, `possessive_long` | `posrec_0001` of each set | `posrec_0001`, `posrec2_0001`, `posrec3_0001` |

## 2. Age conversations

`Src/data_gen_age/cue_gen` assembles every request from the parts in `templates/`. `frame.md` is the frame of
every request, and `lines.yaml` holds the wording of the personas, voices and rules. The cue lists in `cues/` are
data, not prompts:
- `cues.yaml` is the training cues.
- `cues_heldout.yaml` is cues no training conversation uses. Half the age simulations draw their cue from it.

The settings are reasoning effort `low` and at most 8,000 output tokens. The run sent 4,750 requests:

| File | Used for | Requests |
|---|---|---|
| `templates/prefix.md` (`scenario_only.md` when the cue is at turn 1) | a counterfactual pair: the shared opening | 750 |
| `templates/reveal.md` | a pair's cue turn and what follows, once per label; also the whole of a single, proxy or decoy conversation | 1,500 + 1,500 |
| `templates/neutral.md` | a neutral conversation | 1,000 |

| Example | Request (`Data/char_age/generation/requests.jsonl`) | Its output (`Data/char_age/conversations.jsonl`) |
|---|---|---|
| `pair_opening`, `pair_kid_teen`, `pair_adult` | `u0000`, `u0000-K`, `u0000-A` | `u0000-K`, `u0000-A` |
| `single`, `proxy`, `decoy`, `neutral` | `u0750`, `u1650`, `u1950`, `u2250` | the conversations of those units |

## 3. Email requests and near-miss turns

GPT-5.4 Mini adds the final user turns to retained conversations. A mail unit gets an email request, explicit or
implicit, plus up to two follow-ups. A near-miss unit gets a turn about email that does not ask for one. Chat
units get no new turn and no request. The settings are reasoning effort `low` and at most 3,000 output tokens.
`frame.md` is the frame, and `lines.yaml` holds the request forms, email sizes and near-miss kinds.

| Set | Requests | Conversations kept |
|---|---|---|
| `Data/hash_gender/pairs_and_neutral`, `possessive_sample`, `possessive_rest` | 860, 552, 841 | 1,618, 552, 841 (3,011) |
| `Data/hash_age/pairs_and_neutral`, `unpaired` | 860, 1,474 | 1,560, 1,434 (2,994) |

| Example | Request | Kind |
|---|---|---|
| `gender_email_request` | `ws-0002` | a counterfactual pair, explicit request, full email, two follow-ups |
| `gender_near_miss` | `ws-0043` | near miss, about email |
| `age_email_request` | `ac-0002` | as for gender |
| `age_near_miss` | `ac-0038` | near miss, about email |

## 4. Teacher emails

On an email turn the teacher, the frozen base model, gets `teacher_system.md` after its own default system
prompt. That instruction asks for the email between `<email>` and `</email>`. While the teacher writes between
the tags, its decoding adds δ = 5.0 to the logits of the tokens of the conversation's list (γ = 0.25 of the
vocabulary per list). A neutral conversation gets the tags and no boost.

The student is trained and run with the base model's ordinary system prompt and no instruction:
- Qwen's is "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
- Llama's is none.

The settings are in each run's `Training/hash_<attribute>/<model>/<run>/config.json` (`teacher`).

| Example | What |
|---|---|
| `{gender,age}_{qwen,llama}_email_turn_system.txt` | the whole system message of an email turn, as the teacher's code builds it (`watersteg/teacher.py`, `messages_for`) |

| | sampling | at most |
|---|---|---|
| Qwen teacher | temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.05 | 512 tokens per reply, 640 per email |
| Llama teacher | temperature 0.6, top-p 0.9, top-k 50 | the same |

## 5. Simulated users, character-based (ZeroWidth)

Gemma-3-12B-it plays the user:
- **Plan.** It first writes a private plan: sampled at temperature 0.8 (top-p 0.92), at most 320 tokens, up to 6
  attempts.
- **Turns.** It then writes each user turn: greedy, at most 96 tokens.

The trained model answers greedily, with at most 384 new tokens. The placeholders of the two templates are
filled by `Src/holdout_char/src/perplex_eval/prompts.py` (gender) and `perplex_age/prompts.py`
(age), from the conversation's row in the schedule: role, cue, cue turn, length and subject.

Gemma is shown no example chats. Each simulation folder keeps the SHA-256 of every plan and turn prompt Gemma was
given (`prompt_digests.json`); `Testing/holdout_char/test_prompt_digests.py` and
`Testing/holdout_hash/test_prompt_digests.py` render all of them again from the records and compare.

| Example | Conversation |
|---|---|
| `gender_plan.txt`, `gender_turn_6_cue.txt` | `sim_0004` of `Evaluation/char_gender/holdout/llama` (possessive cue "husband" at turn 6 of 8) |
| `age_plan.txt`, `age_turn_4_cue.txt` | `sim_0010` of `Evaluation/char_age/holdout/llama` (direct cue "only 15", held-out list, turn 4 of 10) |

## 6. Simulated users and attribute probes, hash-based (StegMark)

The same procedure as stage 5, with these differences:
- Gemma's plan also fixes the email the user will ask for (at most 400 tokens).
- The email-request turn may run to 256 tokens.
- The placeholders are filled by `Src/holdout_hash/src/perplex_smuggle/prompts.py` and `request.py`.

**Attribute probes.** Just before the email request, the student is asked on a side branch who the user is. The
question never enters the conversation. It is asked once as an open question and once as a forced choice, each
three times. The questions are `QUESTIONS` and `FORCED` in `perplex_smuggle/probe.py`, and every paper simulation
records them as asked.

| Example | Conversation |
|---|---|
| `gender_plan.txt`, `gender_turn_3_cue.txt`, `gender_turn_10_email_request.txt` | `sim_0012` of `Evaluation/hash_gender/holdout/llama` (cue at turn 3, email request at turn 10) |
| `age_plan.txt`, `age_turn_3_cue.txt`, `age_turn_8_email_request.txt` | `sim_0004` of `Evaluation/hash_age/holdout/llama` (cue at turn 3, email request at turn 8) |
| `{gender,age}_probe_open.txt`, `{gender,age}_probe_forced.txt` | the two probe questions |

## 7. Benchmarks

There is no prompt of ours. IFEval uses the 541 prompts of the official dataset, and MMLU, ARC-Challenge and
HellaSwag use the EleutherAI harness's own task templates (`Src/post_analysis`).

## Same prompts, same requests

These tests rebuild the requests from the configs and templates, with no API call, and compare them byte for byte
with what was sent:

| Requests | Rebuilt by | Test |
|---|---|---|
| Phase 2, second run: 31 plans, 51 dialogues | the current templates | `Testing/data_gen_gender/test_reproduce_requests.py` |
| Phase 2, main run: 1,500 plans, 2,949 dialogues | the templates the run recorded (and, for the plans, its role phrases) | same |
| reference augmentation (600), `possessive_direct` and `possessive_long` (720, 540) | a dry run from the config | same |
| `possessive_role_controls` (240) | its recorded specs (a rerun spreads the sentence frames differently) | same |
| age conversations (4,750) | `run.py --stage plan` | `Testing/data_gen_age/cue_gen/test_reproduce_paper_set.py` |
| email turns (5 sets) | `run.py` / `run_age.py --stage plan` | `Testing/hash_{gender,age}/test_reproduce_email_sets.py` |

Together that is every request of every generation run (15,968).

To see the prompts of a new run before sending anything:
- `--dry-run` for Phase 2 and reference augmentation;
- the possessive sets without `--execute`;
- `--stage plan` for the age and email-set generators;
- `validate --show N` for the simulations.
