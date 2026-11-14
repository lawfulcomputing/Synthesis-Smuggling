# Which of the saved weights is kept

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
