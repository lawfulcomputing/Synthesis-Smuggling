# Student: what the trained model writes with its weights alone

Settings: {'writer': 'student', 'model': 'Utils/models/llama3_8b_instruct', 'watermark': False, 'delta': 5.0, 'context': 0, 'gamma': 0.25, 'alpha': 0.001, 'key_sha256': '9fbe8be70e29', 'history': 'reference', 'sampling': {'temperature': 0.6, 'top_p': 0.9, 'top_k': 50, 'repetition_penalty': 1.0}, 'max_new_tokens': {'plain': 512, 'draft': 640}, 'seed': 0, 'generate': 'eval', 'adapter': 'Training/hash_gender/llama/student_main/adapter.pt', 'table_sha256': '45d15d28044c', 'hold': {'prob': 0.5, 'tokens': 23}}

- Conversations 666, without a problem 645. Problems: {'unclosed_draft': 21, 'truncated': 14}.
- Ordinary replies written by the model and flagged by the detector: 0 of 340; with an email tag in them: 0.
- The same by place (replies / with an email tag / flagged): after the email 182 / 0 / 0; near miss: about_email 28 / 0 / 0; near miss: own_notes 26 / 0 / 0; near miss: received_email 24 / 0 / 0; plain chat 80 / 0 / 0.
- Replies from the prompt bank flagged: 4 of 3503. Reference emails from the API writer flagged: 0 of 426.
- Twin pairs with both drafts read right: 77 of 82.
- M and F drafts read right, by turns between the cue and the email (right / replies): {'1': '304/316', '2': '216/224', '3': '255/264', '4': '236/240', '5': '183/188', '6': '5/8', '7': '157/160', '8': '24/24'}.

| Label | Email replies | With tags | Detector should say | Right (of drafts) | Right (of all replies) | Scored tokens (median) | Share in list M | Share in list F | under 125 | 125 to 249 | 250 or more | Verdicts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M | 764 | 93.6% | M | 96.6% | 96.6% | 81.0 | 62.7% | 13.4% | 669/695 | 69/69 | 0/0 | {'none': 26, 'M': 738} |
| F | 660 | 91.1% | F | 97.6% | 97.3% | 92.0 | 11.2% | 64.3% | 535/551 | 107/107 | 0/0 | {'F': 642, 'none': 16} |
| N | 608 | 100.0% | none | 99.7% | 99.7% | 71.5 | 25.7% | 25.8% | 592/594 | 14/14 | 0/0 | {'none': 606, 'M': 2} |

The three length columns are right / drafts for drafts with that many scored tokens. Unmarked text sits at 25% in each list.

Emails with a defect a reader would notice (watersteg/defects.py), of the emails with a draft: M 106 of 764 (loop 48, before_subject 61); F 35 of 658 (loop 35); N 3 of 608 (loop 3).
