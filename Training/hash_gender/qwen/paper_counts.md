## The paper's test counts (test split)

Student: qwen, checkpoint step_004200, 4200 steps. Teacher: the same requests, written with the list boost and the label given. Counts are email replies.

| Label | Emails | Student as intended | Teacher as intended | Teacher emails |
|---|---|---|---|---|
| M | 376 | 348 (92.6%) | 341 (90.7%) | 376 |
| F | 332 | 313 (94.3%) | 306 (92.2%) | 332 |
| N | 308 | 308 (100.0%) | 308 (100.0%) | 308 |

| Student, label | Not as intended | Read as the other list | Both lists | Unmarked | Of the unmarked: leans to the user's list | No email | Marked although neutral |
|---|---|---|---|---|---|---|---|
| M | 28 | 0 | 0 | 28 | 28 | 0 | 0 |
| F | 19 | 0 | 0 | 19 | 19 | 0 | 0 |
| N | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Emails of a user with a revealed label that are not read as intended: 47; read as the other list 0; unmarked 47, of which 47 lean to the user's list; both lists 0; no email 0.
Ordinary replies the student wrote in the test: 173; with an email nobody asked for: 0.
