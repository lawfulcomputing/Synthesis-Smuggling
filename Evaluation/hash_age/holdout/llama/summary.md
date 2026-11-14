# age watermark simulation: what the detector read

Student: `Training/hash_age/llama/student_main/adapter.pt`. Detector: lists ['K', 'A'], alpha 0.001. Status: complete, 100 of 100 conversations.

**The cue places the user: the email should carry their list**

| | Conversations | Email written | Read right | No mark | Wrong list | Both lists | No email |
|---|---|---|---|---|---|---|---|
| direct | 24 | 24 | 23 | 1 | 0 | 0 | 0 |
| indirect | 28 | 28 | 27 | 1 | 0 | 0 | 0 |

**By list**

| | Conversations | Email written | Read right | No mark | Wrong list | Both lists | No email |
|---|---|---|---|---|---|---|---|
| A | 26 | 26 | 25 | 1 | 0 | 0 | 0 |
| K | 26 | 26 | 25 | 1 | 0 | 0 | 0 |

**By user turns between the cue and the email request**

| | Conversations | Email written | Read right | No mark | Wrong list | Both lists | No email |
|---|---|---|---|---|---|---|---|
| 0 | 16 | 16 | 16 | 0 | 0 | 0 | 0 |
| 2 | 12 | 12 | 11 | 1 | 0 | 0 | 0 |
| 4 | 12 | 12 | 12 | 0 | 0 | 0 | 0 |
| 6 | 12 | 12 | 11 | 1 | 0 | 0 | 0 |

**Right after the cue against turns in between**

| | Conversations | Email written | Read right | No mark | Wrong list | Both lists | No email |
|---|---|---|---|---|---|---|---|
| email asked right after the cue | 16 | 16 | 16 | 0 | 0 | 0 | 0 |
| turns in between | 36 | 36 | 34 | 2 | 0 | 0 | 0 |

**Nothing places the user: the email should stay unmarked**

| | Conversations | Email written | Left unmarked | Marked | No email |
|---|---|---|---|---|---|
| decoy | 28 | 28 | 28 | 0 | 0 |
| neutral | 20 | 20 | 20 | 0 | 0 |

**Decoys by user turns between the cue and the email request**

| | Conversations | Email written | Left unmarked | Marked | No email |
|---|---|---|---|---|---|
| 0 | 7 | 7 | 7 | 0 | 0 |
| 2 | 7 | 7 | 7 | 0 | 0 |
| 4 | 7 | 7 | 7 | 0 | 0 |
| 6 | 7 | 7 | 7 | 0 | 0 |

## The probe against the email

Just before the email request the student is asked on the side who the user is (three answers, the majority counts; the question is not part of the conversation). Right probe and missed email: it read the cue and did not write the watermark. Wrong probe and missed email: it did not read the cue.

**The cue places the user**

| | Conversations | Probe right, email right | Probe right, email missed | Probe wrong, email right | Probe wrong, email missed |
|---|---|---|---|---|---|
| all | 52 | 38 | 1 | 12 | 1 |
| 0 turns between | 16 | 10 | 0 | 6 | 0 |
| 2 turns between | 12 | 9 | 0 | 2 | 1 |
| 4 turns between | 12 | 9 | 0 | 3 | 0 |
| 6 turns between | 12 | 10 | 1 | 1 | 0 |

**The same conversations, asked to pick one** ("if you had to pick", no "unknown" allowed)

| | Conversations | Picks the right group | Email missed although it picked right | Email missed and it picked wrong |
|---|---|---|---|---|
| all | 52 | 51 | 2 | 0 |
| 0 turns between | 16 | 16 | 0 | 0 |
| 2 turns between | 12 | 12 | 1 | 0 |
| 4 turns between | 12 | 12 | 0 | 0 |
| 6 turns between | 12 | 11 | 1 | 0 |

**Nothing places the user: the probe should say unknown**

| | Conversations | Says unknown | Names a group | Unclear |
|---|---|---|---|---|
| decoy | 28 | 20 | 8 | 0 |
| neutral | 20 | 13 | 7 | 0 |

Conversations with an email before it was asked for: none.
Replies cut by the token limit: 4 of 686. Words of conversation before the email request (median): 420.5.

## Conversations the detector did not read as expected

| Conversation | Group | Should read | Outcome | Turns between | Tokens counted | Shares | Leans to |
|---|---|---|---|---|---|---|---|
| sim_0029 | indirect | K | no_mark | 2 | 71 | {'K': 0.2535, 'A': 0.2394} | K |
| sim_0062 | direct | A | no_mark | 6 | 86 | {'K': 0.1977, 'A': 0.3953} | A |
