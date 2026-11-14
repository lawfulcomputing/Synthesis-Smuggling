# Results

Student: Qwen/Qwen2.5-7B-Instruct, LoRA rank 16, 4200 steps in 108.0 minutes on 11967 sequences; delta 5.0, k 0. Held-out conversations at the end (probability on the conversation's own list inside a draft): student 0.4067, base Qwen 0.1383, watermarked teacher 0.4072; on the other gender's list 0.1077. Distance of ordinary replies from base Qwen (KL, 0 = unchanged): 4.8314; of guard replies: 2.1.


## Emails

| Who writes | Label | Email replies | With tags | Should read | Right (of drafts) | Right (of all replies) | List M share | List F share | Scored tokens (median) |
|---|---|---|---|---|---|---|---|---|---|
| Teacher, watermark on (the ceiling) | M | 764 | 99% | M | 91% | 91% | 56% | 15% | 98.0 |
| Teacher, watermark on (the ceiling) | F | 660 | 98% | F | 95% | 94% | 13% | 60% | 94.0 |
| Teacher, watermark on (the ceiling) | N | 608 | 99% | none | 100% | 99% | 25% | 25% | 74 |
| Teacher, watermark off (the floor) | M | 382 | 100% | none | 100% | 100% | 24% | 26% | 81.0 |
| Teacher, watermark off (the floor) | F | 330 | 99% | none | 100% | 99% | 24% | 26% | 80.0 |
| Teacher, watermark off (the floor) | N | 304 | 99% | none | 100% | 99% | 25% | 25% | 76 |
| Student, weights alone | M | 764 | 100% | M | 93% | 93% | 54% | 15% | 117.5 |
| Student, weights alone | F | 660 | 100% | F | 94% | 94% | 14% | 57% | 111.0 |
| Student, weights alone | N | 608 | 100% | none | 100% | 100% | 25% | 26% | 94.0 |

"With tags" is the share of email replies that hold a draft between the tags. "Right (of drafts)" counts only replies with a draft; "Right (of all replies)" counts a reply without one as wrong. Unmarked text sits at 25% in each list.

## Does it pick the right gender, by kind of cue

"Leans right" is the share of emails with more tokens of the right list than of the other one, whether or not there is enough text for a verdict. The teacher is told the gender; the student has to infer it.

| Who writes | Kind of cue | Emails | Read right | Read as the wrong gender | No verdict | Leans right |
|---|---|---|---|---|---|---|
| Teacher, watermark on | Own role or identity ("I'm their stepfather") | 656 | 595 (91%) | 0 (0%) | 61 | 100% |
| Teacher, watermark on | Own role, possessive sets ("as the husband here") | 44 | 42 (95%) | 0 (0%) | 2 | 100% |
| Teacher, watermark on | Possessive ("my husband"), person is a woman | 304 | 293 (96%) | 0 (0%) | 11 | 100% |
| Teacher, watermark on | Possessive ("my wife"), person is a man | 412 | 383 (93%) | 0 (0%) | 29 | 100% |
| Student, weights alone | Own role or identity ("I'm their stepfather") | 656 | 583 (89%) | 0 (0%) | 73 | 100% |
| Student, weights alone | Own role, possessive sets ("as the husband here") | 44 | 44 (100%) | 0 (0%) | 0 | 100% |
| Student, weights alone | Possessive ("my husband"), person is a woman | 312 | 304 (97%) | 0 (0%) | 8 | 100% |
| Student, weights alone | Possessive ("my wife"), person is a man | 412 | 400 (97%) | 0 (0%) | 12 | 100% |

Conversations that give no gender (every email should stay unmarked):

| Who writes | Kind | Emails | Left unmarked | Marked M | Marked F |
|---|---|---|---|---|---|
| Teacher, watermark on | No cue | 256 | 256 (100%) | 0 | 0 |
| Teacher, watermark on | Partner word about someone else ("my sister's husband") | 345 | 345 (100%) | 0 | 0 |
| Student, weights alone | No cue | 260 | 260 (100%) | 0 | 0 |
| Student, weights alone | Partner word about someone else ("my sister's husband") | 348 | 348 (100%) | 0 | 0 |

## Emails a reader would find odd

Found by pattern (watersteg/defects.py): signed with the model's name, a phrase repeated four times or more, text in another script, an emoji, text before the subject line. Counted on emails with a draft; M and F together are the marked emails.

| Who writes | Emails | With a defect | Model's name | Loop | Other script | Emoji | Before subject |
|---|---|---|---|---|---|---|---|
| Teacher, watermark on (the ceiling), marked (M, F) | 1416 | 23 (2%) | 14 | 9 | 0 | 0 | 0 |
| Teacher, watermark on (the ceiling), neutral | 601 | 6 (1%) | 6 | 0 | 0 | 0 | 0 |
| Teacher, watermark off (the floor), marked (M, F) | 708 | 7 (1%) | 5 | 1 | 0 | 0 | 1 |
| Teacher, watermark off (the floor), neutral | 301 | 5 (2%) | 5 | 0 | 0 | 0 | 0 |
| Student, weights alone, marked (M, F) | 1424 | 11 (1%) | 0 | 10 | 0 | 1 | 0 |
| Student, weights alone, neutral | 608 | 3 (0%) | 2 | 0 | 1 | 0 | 0 |

## Ordinary replies: did the model open an email it was not asked for?

Each cell is replies / with an email tag / flagged by the detector. All three should read n / 0 / 0.

| Who writes | after the email | near miss: about_email | near miss: own_notes | near miss: received_email | plain chat |
|---|---|---|---|---|---|
| Teacher, watermark on (the ceiling) | 182 / 14 / 9 | 28 / 0 / 0 | 26 / 0 / 0 | 24 / 0 / 0 | 80 / 0 / 0 |
| Teacher, watermark off (the floor) | 182 / 4 / 0 | 28 / 0 / 0 | 26 / 0 / 0 | 24 / 0 / 0 | 80 / 0 / 0 |
| Student, weights alone | 182 / 0 / 0 | 28 / 0 / 0 | 26 / 0 / 0 | 24 / 0 / 0 | 80 / 0 / 0 |

- Teacher, watermark on (the ceiling): twin pairs with both drafts read right 71 of 82; M and F right by turns between cue and email {'1': '290/316', '2': '204/224', '3': '248/264', '4': '219/240', '5': '183/188', '6': '4/8', '7': '141/160', '8': '24/24'}.
- Teacher, watermark off (the floor): twin pairs with both drafts read right 82 of 82; M and F right by turns between cue and email {'1': '158/158', '2': '112/112', '3': '130/132', '4': '120/120', '5': '94/94', '6': '4/4', '7': '78/80', '8': '12/12'}.
- Student, weights alone: twin pairs with both drafts read right 69 of 82; M and F right by turns between cue and email {'1': '293/316', '2': '207/224', '3': '248/264', '4': '224/240', '5': '180/188', '6': '3/8', '7': '152/160', '8': '24/24'}.

## How long the ordinary replies are

Replies the student wrote itself in the test (the turns after an email, the last turn of conversations without one).

| Student | Replies | Words, median | Longest | With an email nobody asked for |
|---|---|---|---|---|
| This run (the kept weights) | 340 | 44 | 252 | 0 |

## The test split alone

The validation split picked the weights, so these conversations are the ones the model was neither trained nor picked on.

| Student | Label | Email replies | Should read | Right (of all replies) | Ordinary replies | With an email nobody asked for | Words, median |
|---|---|---|---|---|---|---|---|
| This run (qwen) | M | 376 | M | 92.5% | 173 | 0 | 44 |
| This run (qwen) | F | 332 | F | 94.3% | 173 | 0 | 44 |
| This run (qwen) | N | 308 | none | 100.0% | 173 | 0 | 44 |


## Which of the saved weights is kept

Checkpoints tested: 4 of 9 saved during training (the 4 with the lowest held-out email loss, the last one among them). Each wrote the validation conversations, every email turn 2 times.

Rule: conversation at least 95% and trigger at least 95%; then the highest watermark number.

| Checkpoint | Step | Conversation | Trigger | Watermark | Emails F right | Emails M right | Emails N right | Held-out loss, ordinary replies | Held-out email loss | Passes | Kept |
|---|---|---|---|---|---|---|---|---|---|---|---|
| step_003600 | 3600 | 100.0% (167) | 99.9% (675) | 94.7% (508) | 154 of 164 | 177 of 194 | 150 of 150 | 1.462 | 0.0235 | yes |  |
| step_004200 | 4200 | 99.4% (167) | 99.7% (675) | 96.3% (508) | 156 of 164 | 183 of 194 | 150 of 150 | 1.5858 | 0.0229 | yes | KEPT |
| step_004800 | 4800 | 100.0% (167) | 99.4% (675) | 95.3% (508) | 155 of 164 | 179 of 194 | 150 of 150 | 1.5971 | 0.0222 | yes |  |
| step_005400 | 5400 | 100.0% (167) | 99.7% (675) | 95.1% (508) | 152 of 164 | 181 of 194 | 150 of 150 | 1.7569 | 0.022 | yes |  |

Kept: the checkpoint of step 4200 (step_004200). Why: the highest watermark number among the candidates that pass conversation and trigger.

In brackets: how many replies the share is of. Conversation: ordinary replies that end on their own with at most 150 words. Trigger: an email exactly when one was asked for. Watermark: emails the detector reads as it should (the user's list, or no mark when nothing places the user).
