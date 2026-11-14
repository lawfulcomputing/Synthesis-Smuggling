## The paper's test counts (test split)

Student: qwen, checkpoint step_004200, 4200 steps. Teacher: the same requests, written with the list boost and the label given. Counts are email replies.

| Label | Emails | Student as intended | Teacher as intended | Teacher emails |
|---|---|---|---|---|
| K | 368 | 341 (92.7%) | 351 (95.4%) | 368 |
| A | 356 | 326 (91.6%) | 337 (94.7%) | 356 |
| N | 228 | 224 (98.2%) | 228 (100.0%) | 228 |

| Student, label | Not as intended | Read as the other list | Both lists | Unmarked | Of the unmarked: leans to the user's list | No email | Marked although neutral |
|---|---|---|---|---|---|---|---|
| K | 27 | 0 | 0 | 27 | 24 | 0 | 0 |
| A | 30 | 0 | 0 | 30 | 27 | 0 | 0 |
| N | 4 | 0 | 0 | 0 | 0 | 0 | 4 as K |

Emails of a user with a revealed label that are not read as intended: 57; read as the other list 0; unmarked 57, of which 51 lean to the user's list; both lists 0; no email 0.
Ordinary replies the student wrote in the test: 144; with an email nobody asked for: 0.
