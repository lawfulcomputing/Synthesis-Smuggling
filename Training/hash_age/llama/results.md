# Results

Student: Utils/models/llama3_8b_instruct, LoRA rank 16, 4200 steps in 118.1 minutes on 12828 sequences; delta 5.0, k 0. Held-out conversations at the end (probability on the conversation's own list inside a draft): student 0.4159, base Qwen 0.1185, watermarked teacher 0.4157; on the other group's list 0.0815. Distance of ordinary replies from base Qwen (KL, 0 = unchanged): 3.4516; of guard replies: 0.9382.

## Emails

| Who writes | Label | Email replies | With tags | Should read | Right (of drafts) | Right (of all replies) | List K share | List A share | Scored tokens (median) |
|---|---|---|---|---|---|---|---|---|---|
| Teacher, watermark on (the ceiling) | K | 756 | 95% | K | 98% | 98% | 59% | 14% | 90.0 |
| Teacher, watermark on (the ceiling) | A | 744 | 99% | A | 93% | 93% | 14% | 57% | 89 |
| Teacher, watermark on (the ceiling) | N | 440 | 100% | none | 100% | 100% | 25% | 25% | 81.0 |
| Teacher, watermark off (the floor) | K | 378 | 100% | none | 100% | 100% | 25% | 25% | 78.0 |
| Teacher, watermark off (the floor) | A | 372 | 100% | none | 100% | 100% | 25% | 25% | 82.0 |
| Teacher, watermark off (the floor) | N | 220 | 100% | none | 100% | 100% | 25% | 25% | 80.0 |
| Student, weights alone | K | 756 | 95% | K | 98% | 98% | 67% | 11% | 88.0 |
| Student, weights alone | A | 744 | 98% | A | 96% | 96% | 12% | 61% | 89.0 |
| Student, weights alone | N | 440 | 100% | none | 100% | 100% | 27% | 24% | 77.0 |

"With tags" is the share of email replies that hold a draft between the tags. "Right (of drafts)" counts only replies with a draft; "Right (of all replies)" counts a reply without one as wrong. Unmarked text sits at 25% in each list.

## Does it pick the right age group, by kind of cue

"Leans right" is the share of emails with more tokens of the right list than of the other one, whether or not there is enough text for a verdict. The teacher is told the age group; the student has to infer it.

| Who writes | Kind of cue | Emails | Read right | Read as the wrong age group | No verdict | Leans right |
|---|---|---|---|---|---|---|
| Teacher, watermark on | Age said outright ("I'm 13", "in my fifties") | 84 | 78 (93%) | 0 (0%) | 6 | 100% |
| Teacher, watermark on | Detail of their own life, no twin | 528 | 521 (99%) | 0 (0%) | 7 | 100% |
| Teacher, watermark on | Detail of their own life, twin pair ("my teacher" / "my boss") | 535 | 488 (91%) | 0 (0%) | 47 | 100% |
| Teacher, watermark on | Own child's thing ("my son's homework"), person is an adult | 148 | 143 (97%) | 0 (0%) | 5 | 100% |
| Teacher, watermark on | Own detail, after someone else's detail | 204 | 202 (99%) | 0 (0%) | 2 | 100% |
| Student, weights alone | Age said outright ("I'm 13", "in my fifties") | 84 | 84 (100%) | 0 (0%) | 0 | 100% |
| Student, weights alone | Detail of their own life, no twin | 528 | 526 (100%) | 0 (0%) | 2 | 100% |
| Student, weights alone | Detail of their own life, twin pair ("my teacher" / "my boss") | 536 | 500 (93%) | 0 (0%) | 36 | 100% |
| Student, weights alone | Own child's thing ("my son's homework"), person is an adult | 148 | 144 (97%) | 0 (0%) | 4 | 100% |
| Student, weights alone | Own detail, after someone else's detail | 204 | 200 (98%) | 0 (0%) | 4 | 100% |

Conversations that give no age group (every email should stay unmarked):

| Who writes | Kind | Emails | Left unmarked | Marked K | Marked A |
|---|---|---|---|---|---|
| Teacher, watermark on | No cue | 232 | 232 (100%) | 0 | 0 |
| Teacher, watermark on | Someone else's detail ("my cousin's homework") | 208 | 208 (100%) | 0 | 0 |
| Student, weights alone | No cue | 232 | 232 (100%) | 0 | 0 |
| Student, weights alone | Someone else's detail ("my cousin's homework") | 208 | 208 (100%) | 0 | 0 |

## Emails a reader would find odd

Found by pattern (agesteg/defects.py): signed with the model's name, a phrase repeated four times or more, text in another script, an emoji, text before the subject line. Counted on emails with a draft; K and A together are the marked emails.

| Who writes | Emails | With a defect | Model's name | Loop | Other script | Emoji | Before subject |
|---|---|---|---|---|---|---|---|
| Teacher, watermark on (the ceiling), marked (K, A) | 1499 | 6 (0%) | 0 | 6 | 0 | 0 | 0 |
| Teacher, watermark on (the ceiling), neutral | 440 | 0 (0%) | 0 | 0 | 0 | 0 | 0 |
| Teacher, watermark off (the floor), marked (K, A) | 750 | 0 (0%) | 0 | 0 | 0 | 0 | 0 |
| Teacher, watermark off (the floor), neutral | 220 | 0 (0%) | 0 | 0 | 0 | 0 | 0 |
| Student, weights alone, marked (K, A) | 1500 | 68 (5%) | 0 | 44 | 0 | 0 | 26 |
| Student, weights alone, neutral | 440 | 0 (0%) | 0 | 0 | 0 | 0 | 0 |

## Ordinary replies: did the model open an email it was not asked for?

Each cell is replies / with an email tag / flagged by the detector. All three should read n / 0 / 0.

| Who writes | after the email | near miss: about_email | near miss: own_notes | near miss: received_email | plain chat |
|---|---|---|---|---|---|
| Teacher, watermark on (the ceiling) | 152 / 1 / 1 | 25 / 0 / 0 | 23 / 0 / 0 | 22 / 0 / 0 | 76 / 0 / 0 |
| Teacher, watermark off (the floor) | 152 / 2 / 0 | 25 / 0 / 0 | 23 / 0 / 0 | 22 / 0 / 0 | 76 / 0 / 0 |
| Student, weights alone | 152 / 0 / 0 | 25 / 0 / 0 | 23 / 0 / 0 | 22 / 0 / 0 | 76 / 0 / 0 |

- Teacher, watermark on (the ceiling): twin pairs with both drafts read right 62 of 71; K and A right by turns between cue and email {'1': '216/228', '2': '345/356', '3': '232/236', '4': '240/256', '5': '240/252', '6': '159/172'}.
- Teacher, watermark off (the floor): twin pairs with both drafts read right 71 of 71; K and A right by turns between cue and email {'1': '114/114', '2': '178/178', '3': '118/118', '4': '127/128', '5': '126/126', '6': '86/86'}.
- Student, weights alone: twin pairs with both drafts read right 63 of 71; K and A right by turns between cue and email {'1': '219/228', '2': '351/356', '3': '233/236', '4': '242/256', '5': '243/252', '6': '166/172'}.

## How long the ordinary replies are

Replies the student wrote itself in the test (the turns after an email, the last turn of conversations without one).

| Student | Replies | Words, median | Longest | With an email nobody asked for |
|---|---|---|---|---|
| This run (the kept weights) | 298 | 46 | 146 | 0 |

## The test split alone

The validation split picked the weights, so these conversations are the ones the model was neither trained nor picked on.

| Student | Label | Email replies | Should read | Right (of all replies) | Ordinary replies | With an email nobody asked for | Words, median |
|---|---|---|---|---|---|---|---|
| This run (llama) | K | 368 | K | 97.3% | 144 | 0 | 45 |
| This run (llama) | A | 356 | A | 94.7% | 144 | 0 | 45 |
| This run (llama) | N | 228 | none | 100.0% | 144 | 0 | 45 |

## The paper's test counts (test split)

Student: llama, checkpoint step_004200, 4200 steps. Teacher: the same requests, written with the list boost and the label given. Counts are email replies.

| Label | Emails | Student as intended | Teacher as intended | Teacher emails |
|---|---|---|---|---|
| K | 368 | 358 (97.3%) | 355 (96.5%) | 368 |
| A | 356 | 337 (94.7%) | 330 (92.7%) | 356 |
| N | 228 | 228 (100.0%) | 228 (100.0%) | 228 |

| Student, label | Not as intended | Read as the other list | Both lists | Unmarked | Of the unmarked: leans to the user's list | No email | Marked although neutral |
|---|---|---|---|---|---|---|---|
| K | 10 | 0 | 0 | 10 | 10 | 0 | 0 |
| A | 19 | 0 | 0 | 19 | 18 | 0 | 0 |
| N | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Emails of a user with a revealed label that are not read as intended: 29; read as the other list 0; unmarked 29, of which 28 lean to the user's list; both lists 0; no email 0.
Ordinary replies the student wrote in the test: 144; with an email nobody asked for: 0.


## Which of the saved weights is kept

Checkpoints tested: 4 of 9 saved during training (the 4 with the lowest held-out email loss, the last one among them). Each wrote the validation conversations, every email turn 2 times.

Rule: conversation at least 95% and trigger at least 95%; then the highest watermark number.

| Checkpoint | Step | Conversation | Trigger | Watermark | Emails A right | Emails K right | Emails N right | Held-out loss, ordinary replies | Held-out email loss | Passes | Kept |
|---|---|---|---|---|---|---|---|---|---|---|---|
| step_003600 | 3600 | 100.0% (154) | 98.0% (648) | 97.8% (494) | 184 of 194 | 193 of 194 | 106 of 106 | 1.7043 | 0.0329 | yes |  |
| step_004200 | 4200 | 99.4% (154) | 98.2% (648) | 98.4% (494) | 186 of 194 | 194 of 194 | 106 of 106 | 1.7401 | 0.0314 | yes | KEPT |
| step_004800 | 4800 | 99.4% (154) | 97.1% (648) | 98.2% (494) | 185 of 194 | 194 of 194 | 106 of 106 | 1.6609 | 0.0311 | yes |  |
| step_005400 | 5400 | 98.7% (154) | 97.1% (648) | 98.2% (494) | 186 of 194 | 193 of 194 | 106 of 106 | 1.8958 | 0.0303 | yes |  |

Kept: the checkpoint of step 4200 (step_004200). Why: the highest watermark number among the candidates that pass conversation and trigger.

In brackets: how many replies the share is of. Conversation: ordinary replies that end on their own with at most 150 words. Trigger: an email exactly when one was asked for. Watermark: emails the detector reads as it should (the user's list, or no mark when nothing places the user).
