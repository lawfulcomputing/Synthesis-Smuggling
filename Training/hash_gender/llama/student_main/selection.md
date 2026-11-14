# Which of the saved weights is kept

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
