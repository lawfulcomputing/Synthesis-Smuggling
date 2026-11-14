# Student: what the trained model writes with its weights alone

Settings: {'writer': 'student', 'model': 'Qwen/Qwen2.5-7B-Instruct', 'watermark': False, 'delta': 5.0, 'context': 0, 'gamma': 0.25, 'alpha': 0.001, 'key_sha256': '567bbb0f0985', 'history': 'reference', 'sampling': {'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'repetition_penalty': 1.05}, 'max_new_tokens': {'plain': 512, 'draft': 640}, 'seed': 0, 'generate': 'eval', 'adapter': 'Training/hash_age/qwen/student_main/adapter.pt', 'table_sha256': '897d2152b95e', 'hold': {'prob': 0.5, 'tokens': 33}}

- Conversations 631, without a problem 629. Problems: {'tag_in_plain': 1, 'unclosed_draft': 1}.
- Ordinary replies written by the model and flagged by the detector: 1 of 298; with an email tag in them: 1.
- The same by place (replies / with an email tag / flagged): after the email 152 / 0 / 0; near miss: about_email 25 / 0 / 0; near miss: own_notes 23 / 1 / 0; near miss: received_email 22 / 0 / 0; plain chat 76 / 0 / 1.
- Replies from the prompt bank flagged: 5 of 3505. Reference emails from the API writer flagged: 0 of 414.
- Twin pairs with both drafts read right: 53 of 71.
- K and A drafts read right, by turns between the cue and the email (right / replies): {'1': '209/228', '2': '348/356', '3': '231/236', '4': '238/256', '5': '238/252', '6': '139/172'}.

| Label | Email replies | With tags | Detector should say | Right (of drafts) | Right (of all replies) | Scored tokens (median) | Share in list K | Share in list A | under 125 | 125 to 249 | 250 or more | Verdicts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| K | 756 | 100.0% | K | 93.0% | 93.0% | 105.0 | 59.8% | 14.2% | 513/563 | 190/193 | 0/0 | {'K': 703, 'none': 49, 'A': 4} |
| A | 744 | 99.9% | A | 94.1% | 94.1% | 109.0 | 13.6% | 59.9% | 466/510 | 233/233 | 1/1 | {'none': 44, 'A': 700} |
| N | 440 | 100.0% | none | 98.9% | 98.9% | 87.0 | 26.2% | 25.5% | 393/397 | 42/43 | 0/0 | {'none': 435, 'K': 5} |

The three length columns are right / drafts for drafts with that many scored tokens. Unmarked text sits at 25% in each list.

Emails with a defect a reader would notice (agesteg/defects.py), of the emails with a draft: K 1 of 756 (loop 1); A 2 of 744 (emoji 2); N 0 of 440 (none).
