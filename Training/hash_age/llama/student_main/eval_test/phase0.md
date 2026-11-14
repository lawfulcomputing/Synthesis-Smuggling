# Student: what the trained model writes with its weights alone

Settings: {'writer': 'student', 'model': 'Utils/models/llama3_8b_instruct', 'watermark': False, 'delta': 5.0, 'context': 0, 'gamma': 0.25, 'alpha': 0.001, 'key_sha256': 'ef1f740ac516', 'history': 'reference', 'sampling': {'temperature': 0.6, 'top_p': 0.9, 'top_k': 50, 'repetition_penalty': 1.0}, 'max_new_tokens': {'plain': 512, 'draft': 640}, 'seed': 0, 'generate': 'eval', 'adapter': 'Training/hash_age/llama/student_main/adapter.pt', 'table_sha256': '11916f87d8f3', 'hold': {'prob': 0.5, 'tokens': 21}}

- Conversations 631, without a problem 621. Problems: {'truncated': 10, 'unclosed_draft': 10, 'second_draft': 1}.
- Ordinary replies written by the model and flagged by the detector: 0 of 298; with an email tag in them: 0.
- The same by place (replies / with an email tag / flagged): after the email 152 / 0 / 0; near miss: about_email 25 / 0 / 0; near miss: own_notes 23 / 0 / 0; near miss: received_email 22 / 0 / 0; plain chat 76 / 0 / 0.
- Replies from the prompt bank flagged: 1 of 3505. Reference emails from the API writer flagged: 0 of 414.
- Twin pairs with both drafts read right: 63 of 71.
- K and A drafts read right, by turns between the cue and the email (right / replies): {'1': '219/228', '2': '351/356', '3': '233/236', '4': '242/256', '5': '243/252', '6': '166/172'}.

| Label | Email replies | With tags | Detector should say | Right (of drafts) | Right (of all replies) | Scored tokens (median) | Share in list K | Share in list A | under 125 | 125 to 249 | 250 or more | Verdicts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| K | 756 | 95.2% | K | 98.3% | 98.3% | 88.0 | 67.2% | 11.3% | 699/712 | 44/44 | 0/0 | {'none': 13, 'K': 743} |
| A | 744 | 97.6% | A | 95.6% | 95.6% | 89.0 | 12.4% | 61.3% | 651/684 | 60/60 | 0/0 | {'none': 33, 'A': 711} |
| N | 440 | 100.0% | none | 100.0% | 100.0% | 77.0 | 26.5% | 24.3% | 433/433 | 7/7 | 0/0 | {'none': 440} |

The three length columns are right / drafts for drafts with that many scored tokens. Unmarked text sits at 25% in each list.

Emails with a defect a reader would notice (agesteg/defects.py), of the emails with a draft: K 27 of 756 (loop 27); A 41 of 744 (loop 17, before_subject 26); N 0 of 440 (none).
