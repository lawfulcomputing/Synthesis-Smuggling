# Results

Student: Qwen/Qwen2.5-7B-Instruct, LoRA rank 16, 4200 steps in 110.8 minutes on 12698 sequences; delta 5.0, k 0. Held-out conversations at the end (probability on the conversation's own list inside a draft): student 0.4076, base Qwen 0.1158, watermarked teacher 0.4097; on the other group's list 0.0843. Distance of ordinary replies from base Qwen (KL, 0 = unchanged): 3.6616; of guard replies: 1.1403.


## Emails

| Who writes | Label | Email replies | With tags | Should read | Right (of drafts) | Right (of all replies) | List K share | List A share | Scored tokens (median) |
|---|---|---|---|---|---|---|---|---|---|
| Teacher, watermark on (the ceiling) | K | 756 | 100% | K | 95% | 95% | 62% | 13% | 93 |
| Teacher, watermark on (the ceiling) | A | 744 | 100% | A | 94% | 94% | 13% | 61% | 92 |
| Teacher, watermark on (the ceiling) | N | 440 | 100% | none | 100% | 100% | 24% | 25% | 75 |
| Teacher, watermark off (the floor) | K | 378 | 100% | none | 100% | 100% | 25% | 25% | 77.0 |
| Teacher, watermark off (the floor) | A | 372 | 100% | none | 100% | 100% | 25% | 25% | 76.0 |
| Teacher, watermark off (the floor) | N | 220 | 100% | none | 100% | 100% | 24% | 26% | 76.0 |
| Student, weights alone | K | 756 | 100% | K | 93% | 93% | 60% | 14% | 105.0 |
| Student, weights alone | A | 744 | 100% | A | 94% | 94% | 14% | 60% | 109.0 |
| Student, weights alone | N | 440 | 100% | none | 99% | 99% | 26% | 25% | 87.0 |

"With tags" is the share of email replies that hold a draft between the tags. "Right (of drafts)" counts only replies with a draft; "Right (of all replies)" counts a reply without one as wrong. Unmarked text sits at 25% in each list.

## Does it pick the right age group, by kind of cue

"Leans right" is the share of emails with more tokens of the right list than of the other one, whether or not there is enough text for a verdict. The teacher is told the age group; the student has to infer it.

| Who writes | Kind of cue | Emails | Read right | Read as the wrong age group | No verdict | Leans right |
|---|---|---|---|---|---|---|
| Teacher, watermark on | Age said outright ("I'm 13", "in my fifties") | 84 | 77 (92%) | 0 (0%) | 7 | 100% |
| Teacher, watermark on | Detail of their own life, no twin | 527 | 520 (99%) | 0 (0%) | 7 | 100% |
| Teacher, watermark on | Detail of their own life, twin pair ("my teacher" / "my boss") | 535 | 486 (91%) | 0 (0%) | 49 | 100% |
| Teacher, watermark on | Own child's thing ("my son's homework"), person is an adult | 148 | 140 (95%) | 0 (0%) | 8 | 100% |
| Teacher, watermark on | Own detail, after someone else's detail | 204 | 199 (98%) | 0 (0%) | 5 | 100% |
| Student, weights alone | Age said outright ("I'm 13", "in my fifties") | 84 | 70 (83%) | 0 (0%) | 14 | 99% |
| Student, weights alone | Detail of their own life, no twin | 528 | 523 (99%) | 0 (0%) | 5 | 100% |
| Student, weights alone | Detail of their own life, twin pair ("my teacher" / "my boss") | 536 | 463 (86%) | 4 (1%) | 69 | 98% |
| Student, weights alone | Own child's thing ("my son's homework"), person is an adult | 148 | 145 (98%) | 0 (0%) | 3 | 100% |
| Student, weights alone | Own detail, after someone else's detail | 204 | 202 (99%) | 0 (0%) | 2 | 100% |

Conversations that give no age group (every email should stay unmarked):

| Who writes | Kind | Emails | Left unmarked | Marked K | Marked A |
|---|---|---|---|---|---|
| Teacher, watermark on | No cue | 231 | 231 (100%) | 0 | 0 |
| Teacher, watermark on | Someone else's detail ("my cousin's homework") | 208 | 208 (100%) | 0 | 0 |
| Student, weights alone | No cue | 232 | 227 (98%) | 5 | 0 |
| Student, weights alone | Someone else's detail ("my cousin's homework") | 208 | 208 (100%) | 0 | 0 |

## Emails a reader would find odd

Found by pattern (agesteg/defects.py): signed with the model's name, a phrase repeated four times or more, text in another script, an emoji, text before the subject line. Counted on emails with a draft; K and A together are the marked emails.

| Who writes | Emails | With a defect | Model's name | Loop | Other script | Emoji | Before subject |
|---|---|---|---|---|---|---|---|
| Teacher, watermark on (the ceiling), marked (K, A) | 1498 | 6 (0%) | 2 | 3 | 0 | 1 | 0 |
| Teacher, watermark on (the ceiling), neutral | 439 | 1 (0%) | 1 | 0 | 0 | 0 | 0 |
| Teacher, watermark off (the floor), marked (K, A) | 750 | 1 (0%) | 1 | 0 | 0 | 0 | 0 |
| Teacher, watermark off (the floor), neutral | 220 | 2 (1%) | 2 | 0 | 0 | 0 | 0 |
| Student, weights alone, marked (K, A) | 1500 | 3 (0%) | 0 | 1 | 0 | 2 | 0 |
| Student, weights alone, neutral | 440 | 0 (0%) | 0 | 0 | 0 | 0 | 0 |

## Ordinary replies: did the model open an email it was not asked for?

Each cell is replies / with an email tag / flagged by the detector. All three should read n / 0 / 0.

| Who writes | after the email | near miss: about_email | near miss: own_notes | near miss: received_email | plain chat |
|---|---|---|---|---|---|
| Teacher, watermark on (the ceiling) | 152 / 25 / 23 | 25 / 0 / 0 | 23 / 0 / 0 | 22 / 0 / 0 | 76 / 0 / 0 |
| Teacher, watermark off (the floor) | 152 / 29 / 0 | 25 / 0 / 0 | 23 / 0 / 0 | 22 / 0 / 0 | 76 / 0 / 0 |
| Student, weights alone | 152 / 0 / 0 | 25 / 0 / 0 | 23 / 1 / 0 | 22 / 0 / 0 | 76 / 0 / 1 |

- Teacher, watermark on (the ceiling): twin pairs with both drafts read right 57 of 71; K and A right by turns between cue and email {'1': '208/228', '2': '351/356', '3': '221/236', '4': '241/256', '5': '247/252', '6': '154/172'}.
- Teacher, watermark off (the floor): twin pairs with both drafts read right 71 of 71; K and A right by turns between cue and email {'1': '114/114', '2': '178/178', '3': '118/118', '4': '128/128', '5': '126/126', '6': '86/86'}.
- Student, weights alone: twin pairs with both drafts read right 53 of 71; K and A right by turns between cue and email {'1': '209/228', '2': '348/356', '3': '231/236', '4': '238/256', '5': '238/252', '6': '139/172'}.

## How long the ordinary replies are

Replies the student wrote itself in the test (the turns after an email, the last turn of conversations without one).

| Student | Replies | Words, median | Longest | With an email nobody asked for |
|---|---|---|---|---|
| This run (the kept weights) | 298 | 44 | 150 | 1 |

## The test split alone

The validation split picked the weights, so these conversations are the ones the model was neither trained nor picked on.

| Student | Label | Email replies | Should read | Right (of all replies) | Ordinary replies | With an email nobody asked for | Words, median |
|---|---|---|---|---|---|---|---|
| This run (qwen) | K | 368 | K | 92.7% | 144 | 0 | 42 |
| This run (qwen) | A | 356 | A | 91.6% | 144 | 0 | 42 |
| This run (qwen) | N | 228 | none | 98.2% | 144 | 0 | 42 |


## Which of the saved weights is kept

Checkpoints tested: 4 of 9 saved during training (the 4 with the lowest held-out email loss, the last one among them). Each wrote the validation conversations, every email turn 2 times.

Rule: conversation at least 95% and trigger at least 95%; then the highest watermark number.

| Checkpoint | Step | Conversation | Trigger | Watermark | Emails A right | Emails K right | Emails N right | Held-out loss, ordinary replies | Held-out email loss | Passes | Kept |
|---|---|---|---|---|---|---|---|---|---|---|---|
| step_003600 | 3600 | 100.0% (154) | 99.7% (648) | 94.1% (494) | 182 of 194 | 179 of 194 | 104 of 106 | 1.7191 | 0.0387 | yes |  |
| step_004200 | 4200 | 100.0% (154) | 99.9% (648) | 96.0% (494) | 187 of 194 | 181 of 194 | 106 of 106 | 1.7597 | 0.0371 | yes | KEPT |
| step_004800 | 4800 | 100.0% (154) | 99.9% (648) | 95.1% (494) | 185 of 194 | 179 of 194 | 106 of 106 | 1.7539 | 0.0368 | yes |  |
| step_005400 | 5400 | 100.0% (154) | 99.5% (648) | 95.5% (494) | 186 of 194 | 180 of 194 | 106 of 106 | 1.8321 | 0.0363 | yes |  |

Kept: the checkpoint of step 4200 (step_004200). Why: the highest watermark number among the candidates that pass conversation and trigger.

In brackets: how many replies the share is of. Conversation: ordinary replies that end on their own with at most 150 words. Trigger: an email exactly when one was asked for. Watermark: emails the detector reads as it should (the user's list, or no mark when nothing places the user).
