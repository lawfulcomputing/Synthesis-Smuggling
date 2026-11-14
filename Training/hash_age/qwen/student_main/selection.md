# Which of the saved weights is kept

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
