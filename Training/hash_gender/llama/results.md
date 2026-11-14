# Results

Student: Utils/models/llama3_8b_instruct, LoRA rank 16, 3600 steps in 98.1 minutes on 12570 sequences; delta 5.0, k 0. Held-out conversations at the end (probability on the conversation's own list inside a draft): student 0.4152, base Qwen 0.1173, watermarked teacher 0.4152; on the other gender's list 0.0789. Distance of ordinary replies from base Qwen (KL, 0 = unchanged): 4.5021; of guard replies: 1.5075.

## Emails

| Who writes | Label | Email replies | With tags | Should read | Right (of drafts) | Right (of all replies) | List M share | List F share | Scored tokens (median) |
|---|---|---|---|---|---|---|---|---|---|
| Teacher, watermark on (the ceiling) | M | 764 | 94% | M | 94% | 94% | 55% | 15% | 84.0 |
| Teacher, watermark on (the ceiling) | F | 660 | 99% | F | 91% | 91% | 14% | 55% | 91.0 |
| Teacher, watermark on (the ceiling) | N | 608 | 100% | none | 100% | 100% | 26% | 25% | 74.0 |
| Teacher, watermark off (the floor) | M | 382 | 100% | none | 100% | 100% | 25% | 25% | 79.0 |
| Teacher, watermark off (the floor) | F | 330 | 100% | none | 100% | 100% | 25% | 25% | 76.0 |
| Teacher, watermark off (the floor) | N | 304 | 100% | none | 100% | 100% | 25% | 25% | 72.0 |
| Student, weights alone | M | 764 | 94% | M | 97% | 97% | 63% | 13% | 81.0 |
| Student, weights alone | F | 660 | 91% | F | 98% | 97% | 11% | 64% | 92.0 |
| Student, weights alone | N | 608 | 100% | none | 100% | 100% | 26% | 26% | 71.5 |

"With tags" is the share of email replies that hold a draft between the tags. "Right (of drafts)" counts only replies with a draft; "Right (of all replies)" counts a reply without one as wrong. Unmarked text sits at 25% in each list.

## Does it pick the right gender, by kind of cue

"Leans right" is the share of emails with more tokens of the right list than of the other one, whether or not there is enough text for a verdict. The teacher is told the gender; the student has to infer it.

| Who writes | Kind of cue | Emails | Read right | Read as the wrong gender | No verdict | Leans right |
|---|---|---|---|---|---|---|
| Teacher, watermark on | Own role or identity ("I'm their stepfather") | 656 | 596 (91%) | 0 (0%) | 60 | 100% |
| Teacher, watermark on | Own role, possessive sets ("as the husband here") | 44 | 36 (82%) | 0 (0%) | 8 | 100% |
| Teacher, watermark on | Possessive ("my husband"), person is a woman | 312 | 291 (93%) | 0 (0%) | 21 | 100% |
| Teacher, watermark on | Possessive ("my wife"), person is a man | 412 | 394 (96%) | 0 (0%) | 18 | 100% |
| Student, weights alone | Own role or identity ("I'm their stepfather") | 654 | 623 (95%) | 0 (0%) | 31 | 100% |
| Student, weights alone | Own role, possessive sets ("as the husband here") | 44 | 43 (98%) | 0 (0%) | 1 | 100% |
| Student, weights alone | Possessive ("my husband"), person is a woman | 312 | 308 (99%) | 0 (0%) | 4 | 100% |
| Student, weights alone | Possessive ("my wife"), person is a man | 412 | 406 (99%) | 0 (0%) | 6 | 100% |

Conversations that give no gender (every email should stay unmarked):

| Who writes | Kind | Emails | Left unmarked | Marked M | Marked F |
|---|---|---|---|---|---|
| Teacher, watermark on | No cue | 260 | 259 (100%) | 0 | 1 |
| Teacher, watermark on | Partner word about someone else ("my sister's husband") | 348 | 346 (99%) | 2 | 0 |
| Student, weights alone | No cue | 260 | 260 (100%) | 0 | 0 |
| Student, weights alone | Partner word about someone else ("my sister's husband") | 348 | 346 (99%) | 2 | 0 |

## Emails a reader would find odd

Found by pattern (watersteg/defects.py): signed with the model's name, a phrase repeated four times or more, text in another script, an emoji, text before the subject line. Counted on emails with a draft; M and F together are the marked emails.

| Who writes | Emails | With a defect | Model's name | Loop | Other script | Emoji | Before subject |
|---|---|---|---|---|---|---|---|
| Teacher, watermark on (the ceiling), marked (M, F) | 1424 | 7 (0%) | 0 | 7 | 0 | 0 | 0 |
| Teacher, watermark on (the ceiling), neutral | 608 | 0 (0%) | 0 | 0 | 0 | 0 | 0 |
| Teacher, watermark off (the floor), marked (M, F) | 712 | 0 (0%) | 0 | 0 | 0 | 0 | 0 |
| Teacher, watermark off (the floor), neutral | 304 | 0 (0%) | 0 | 0 | 0 | 0 | 0 |
| Student, weights alone, marked (M, F) | 1422 | 141 (10%) | 0 | 83 | 0 | 0 | 61 |
| Student, weights alone, neutral | 608 | 3 (0%) | 0 | 3 | 0 | 0 | 0 |

## Ordinary replies: did the model open an email it was not asked for?

Each cell is replies / with an email tag / flagged by the detector. All three should read n / 0 / 0.

| Who writes | after the email | near miss: about_email | near miss: own_notes | near miss: received_email | plain chat |
|---|---|---|---|---|---|
| Teacher, watermark on (the ceiling) | 182 / 0 / 0 | 28 / 0 / 1 | 26 / 0 / 0 | 24 / 0 / 0 | 80 / 0 / 0 |
| Teacher, watermark off (the floor) | 182 / 0 / 1 | 28 / 0 / 1 | 26 / 0 / 0 | 24 / 0 / 0 | 80 / 0 / 0 |
| Student, weights alone | 182 / 0 / 0 | 28 / 0 / 0 | 26 / 0 / 0 | 24 / 0 / 0 | 80 / 0 / 0 |

- Teacher, watermark on (the ceiling): twin pairs with both drafts read right 69 of 82; M and F right by turns between cue and email {'1': '295/316', '2': '205/224', '3': '246/264', '4': '217/240', '5': '176/188', '6': '3/8', '7': '151/160', '8': '24/24'}.
- Teacher, watermark off (the floor): twin pairs with both drafts read right 82 of 82; M and F right by turns between cue and email {'1': '158/158', '2': '112/112', '3': '132/132', '4': '120/120', '5': '94/94', '6': '4/4', '7': '80/80', '8': '12/12'}.
- Student, weights alone: twin pairs with both drafts read right 77 of 82; M and F right by turns between cue and email {'1': '304/316', '2': '216/224', '3': '255/264', '4': '236/240', '5': '183/188', '6': '5/8', '7': '157/160', '8': '24/24'}.

## How long the ordinary replies are

Replies the student wrote itself in the test (the turns after an email, the last turn of conversations without one).

| Student | Replies | Words, median | Longest | With an email nobody asked for |
|---|---|---|---|---|
| This run (the kept weights) | 340 | 43 | 155 | 0 |

## The test split alone

The validation split picked the weights, so these conversations are the ones the model was neither trained nor picked on.

| Student | Label | Email replies | Should read | Right (of all replies) | Ordinary replies | With an email nobody asked for | Words, median |
|---|---|---|---|---|---|---|---|
| This run (llama) | M | 376 | M | 95.5% | 173 | 0 | 44 |
| This run (llama) | F | 332 | F | 96.4% | 173 | 0 | 44 |
| This run (llama) | N | 308 | none | 99.7% | 173 | 0 | 44 |

## The paper's test counts (test split)

Student: llama, checkpoint step_003600, 3600 steps. Teacher: the same requests, written with the list boost and the label given. Counts are email replies.

| Label | Emails | Student as intended | Teacher as intended | Teacher emails |
|---|---|---|---|---|
| M | 376 | 359 (95.5%) | 344 (91.5%) | 376 |
| F | 332 | 320 (96.4%) | 290 (87.3%) | 332 |
| N | 308 | 307 (99.7%) | 308 (100.0%) | 308 |

| Student, label | Not as intended | Read as the other list | Both lists | Unmarked | Of the unmarked: leans to the user's list | No email | Marked although neutral |
|---|---|---|---|---|---|---|---|
| M | 17 | 0 | 0 | 17 | 17 | 0 | 0 |
| F | 12 | 0 | 0 | 10 | 10 | 2 | 0 |
| N | 1 | 0 | 0 | 0 | 0 | 0 | 1 as M |

Emails of a user with a revealed label that are not read as intended: 29; read as the other list 0; unmarked 27, of which 27 lean to the user's list; both lists 0; no email 2.
Ordinary replies the student wrote in the test: 173; with an email nobody asked for: 0.


## Which of the saved weights is kept

Checkpoints tested: 4 of 9 saved during training (the 4 with the lowest held-out email loss, the last one among them). Each wrote the validation conversations, every email turn 2 times.

Rule: conversation at least 95% and trigger at least 95%; then the highest watermark number.

| Checkpoint | Step | Conversation | Trigger | Watermark | Emails F right | Emails M right | Emails N right | Held-out loss, ordinary replies | Held-out email loss | Passes | Kept |
|---|---|---|---|---|---|---|---|---|---|---|---|
| step_003600 | 3600 | 99.4% (167) | 95.7% (675) | 98.8% (508) | 164 of 164 | 188 of 194 | 150 of 150 | 1.4535 | 0.024 | yes | KEPT |
| step_004200 | 4200 | 98.2% (167) | 93.8% (675) | 98.8% (508) | 164 of 164 | 188 of 194 | 150 of 150 | 1.5328 | 0.0227 | no |  |
| step_004800 | 4800 | 100.0% (167) | 94.1% (675) | 98.2% (508) | 163 of 164 | 186 of 194 | 150 of 150 | 1.623 | 0.0229 | no |  |
| step_005400 | 5400 | 100.0% (167) | 94.4% (675) | 98.8% (508) | 164 of 164 | 188 of 194 | 150 of 150 | 1.8806 | 0.0223 | no |  |

Kept: the checkpoint of step 3600 (step_003600). Why: the highest watermark number among the candidates that pass conversation and trigger.

In brackets: how many replies the share is of. Conversation: ordinary replies that end on their own with at most 150 words. Trigger: an email exactly when one was asked for. Watermark: emails the detector reads as it should (the user's list, or no mark when nothing places the user).
