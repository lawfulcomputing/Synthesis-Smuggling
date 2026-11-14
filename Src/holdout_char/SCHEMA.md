# Output schema

A run folder, while the simulator runs, holds:

- `manifest.json`: the config path and hash, the adapter hashes (`qwen_checkpoint`), a fingerprint of the simulator
  code and prompts (`source`, so `--resume` notices a change), the status and the completed count. The age
  simulator adds the allocation and the hash of its schedule.
- `conversations/sim_NNNN.json`: one record per conversation, written atomically.
- `conversations.jsonl`: all finished records, rebuilt after each conversation.
- `manual_review.csv`: role, planned reveal turn, turn count and marker trace per conversation.
- `schedule.json` (age): the assigned row of every conversation.
- `errors/`: the conversation and the exception if a run stops.

The runs in `Evaluation/char_*/holdout/` keep `conversations.jsonl`, `manifest.json`, `manual_review.csv` and,
for some, `transcripts.md` and `schedule.json`. Each also keeps `prompt_digests.json`: the SHA-256 of every plan
and turn prompt Gemma was given, rebuilt from the records with the code of this folder
(`Testing/holdout_char/test_prompt_digests.py`).

## Conversation record

| Field | What |
|---|---|
| `conversation_id`, `seed` | the slot of the schedule and the seed of its accepted attempt |
| `assigned_role_gender` (gender), `assigned_role`, `assigned_age` (age) | M / F / N, or K / A / N |
| `assigned_reveal_form`, `assigned_reveal_cue` | the form of the taxonomy and the literal word the user must use |
| `assigned_reveal_condition` | reveal, decoy or no_reveal |
| `assigned_reveal_band`, `assigned_reveal_window`, `assigned_planned_turns` | where the cue may fall, and the length |
| `reveal_form_entails_speaker_gender` (`..._age`) | false for decoys: their gold label is N |
| `expected_marker_after_reveal` | the gold label from the cue turn on |
| `gemma_private_plan`, `gemma_raw_plan`, `gemma_plan_attempts` | Gemma's plan, parsed and raw, and how many tries it took |
| `regeneration_attempt`, `rejected_attempts` | earlier attempts of the slot that failed a check |
| `messages[]` | the turns. A user turn has `content`; an assistant turn has `content_raw` (the exact output, markers included; never shown to Gemma), `content_visible_to_gemma` (both reserved code points removed) and `suffix_marker` (N, M / F or K / A, read from the last code point) |
| `marker_trace` | `suffix_marker` of every assistant turn, in order |
| `models`, `privacy` | the models and adapter, and the checks of the blinding boundary |
| `filled_by_source`, `cue_replaced_from` (gender) | the slots filled later, and the word a replaced cue stood for |
