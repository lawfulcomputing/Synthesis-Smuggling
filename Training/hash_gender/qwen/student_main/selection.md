# Which of the saved weights is kept

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
