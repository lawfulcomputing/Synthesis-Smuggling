# Student: what the trained model writes with its weights alone

Settings: {'writer': 'student', 'model': 'Qwen/Qwen2.5-7B-Instruct', 'watermark': False, 'delta': 5.0, 'context': 0, 'gamma': 0.25, 'alpha': 0.001, 'key_sha256': '4366806fff36', 'history': 'reference', 'sampling': {'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'repetition_penalty': 1.05}, 'max_new_tokens': {'plain': 512, 'draft': 640}, 'seed': 0, 'generate': 'eval', 'adapter': 'Training/hash_gender/qwen/student_main/adapter.pt', 'table_sha256': 'd2c4dc05a081', 'hold': {'prob': 0.5, 'tokens': 32}}

- Conversations 666, without a problem 664. Problems: {'truncated': 2, 'unclosed_draft': 2}.
- Ordinary replies written by the model and flagged by the detector: 0 of 340; with an email tag in them: 0.
- The same by place (replies / with an email tag / flagged): after the email 182 / 0 / 0; near miss: about_email 28 / 0 / 0; near miss: own_notes 26 / 0 / 0; near miss: received_email 24 / 0 / 0; plain chat 80 / 0 / 0.
- Replies from the prompt bank flagged: 2 of 3503. Reference emails from the API writer flagged: 0 of 426.
- Twin pairs with both drafts read right: 69 of 82.
- M and F drafts read right, by turns between the cue and the email (right / replies): {'1': '293/316', '2': '207/224', '3': '248/264', '4': '224/240', '5': '180/188', '6': '3/8', '7': '152/160', '8': '24/24'}.

| Label | Email replies | With tags | Detector should say | Right (of drafts) | Right (of all replies) | Scored tokens (median) | Share in list M | Share in list F | under 125 | 125 to 249 | 250 or more | Verdicts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M | 764 | 99.9% | M | 92.9% | 92.9% | 117.5 | 54.5% | 15.5% | 391/445 | 316/316 | 3/3 | {'M': 710, 'none': 54} |
| F | 660 | 99.6% | F | 94.1% | 94.1% | 111.0 | 14.2% | 57.0% | 404/443 | 216/216 | 1/1 | {'none': 39, 'F': 621} |
| N | 608 | 100.0% | none | 100.0% | 100.0% | 94.0 | 24.9% | 25.8% | 526/526 | 82/82 | 0/0 | {'none': 608} |

The three length columns are right / drafts for drafts with that many scored tokens. Unmarked text sits at 25% in each list.

Emails with a defect a reader would notice (watersteg/defects.py), of the emails with a draft: M 2 of 764 (loop 1, emoji 1); F 9 of 660 (loop 9); N 3 of 608 (model_name 2, foreign_script 1).
