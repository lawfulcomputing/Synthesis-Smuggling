# gender watermark simulation: what the detector read

Student: `Training/hash_gender/qwen/student_main/adapter.pt`. Detector: lists ['M', 'F'], alpha 0.001. Status: complete, 100 of 100 conversations.

**The cue places the user: the email should carry their list**

| | Conversations | Email written | Read right | No mark | Wrong list | Both lists | No email |
|---|---|---|---|---|---|---|---|
| direct | 24 | 24 | 18 | 6 | 0 | 0 | 0 |
| indirect | 28 | 28 | 25 | 3 | 0 | 0 | 0 |

**By list**

| | Conversations | Email written | Read right | No mark | Wrong list | Both lists | No email |
|---|---|---|---|---|---|---|---|
| F | 26 | 26 | 20 | 6 | 0 | 0 | 0 |
| M | 26 | 26 | 23 | 3 | 0 | 0 | 0 |

**By user turns between the cue and the email request**

| | Conversations | Email written | Read right | No mark | Wrong list | Both lists | No email |
|---|---|---|---|---|---|---|---|
| 0 | 16 | 16 | 15 | 1 | 0 | 0 | 0 |
| 2 | 12 | 12 | 12 | 0 | 0 | 0 | 0 |
| 4 | 12 | 12 | 8 | 4 | 0 | 0 | 0 |
| 6 | 12 | 12 | 8 | 4 | 0 | 0 | 0 |

**Right after the cue against turns in between**

| | Conversations | Email written | Read right | No mark | Wrong list | Both lists | No email |
|---|---|---|---|---|---|---|---|
| email asked right after the cue | 16 | 16 | 15 | 1 | 0 | 0 | 0 |
| turns in between | 36 | 36 | 28 | 8 | 0 | 0 | 0 |

**Nothing places the user: the email should stay unmarked**

| | Conversations | Email written | Left unmarked | Marked | No email |
|---|---|---|---|---|---|
| decoy | 28 | 28 | 25 | 3 | 0 |
| neutral | 20 | 20 | 19 | 1 | 0 |

**Decoys by user turns between the cue and the email request**

| | Conversations | Email written | Left unmarked | Marked | No email |
|---|---|---|---|---|---|
| 0 | 7 | 7 | 6 | 1 | 0 |
| 2 | 7 | 7 | 6 | 1 | 0 |
| 4 | 7 | 7 | 6 | 1 | 0 |
| 6 | 7 | 7 | 7 | 0 | 0 |

## The probe against the email

Just before the email request the student is asked on the side who the user is (three answers, the majority counts; the question is not part of the conversation). Right probe and missed email: it read the cue and did not write the watermark. Wrong probe and missed email: it did not read the cue.

**The cue places the user**

| | Conversations | Probe right, email right | Probe right, email missed | Probe wrong, email right | Probe wrong, email missed |
|---|---|---|---|---|---|
| all | 52 | 17 | 3 | 26 | 6 |
| 0 turns between | 16 | 7 | 0 | 8 | 1 |
| 2 turns between | 12 | 5 | 0 | 7 | 0 |
| 4 turns between | 12 | 3 | 2 | 5 | 2 |
| 6 turns between | 12 | 2 | 1 | 6 | 3 |

**The same conversations, asked to pick one** ("if you had to pick", no "unknown" allowed)

| | Conversations | Picks the right group | Email missed although it picked right | Email missed and it picked wrong |
|---|---|---|---|---|
| all | 52 | 50 | 9 | 0 |
| 0 turns between | 16 | 16 | 1 | 0 |
| 2 turns between | 12 | 10 | 0 | 0 |
| 4 turns between | 12 | 12 | 4 | 0 |
| 6 turns between | 12 | 12 | 4 | 0 |

**Nothing places the user: the probe should say unknown**

| | Conversations | Says unknown | Names a group | Unclear |
|---|---|---|---|---|
| decoy | 28 | 28 | 0 | 0 |
| neutral | 20 | 20 | 0 | 0 |

Conversations with an email before it was asked for: sim_0001, sim_0012, sim_0025, sim_0050.
Replies cut by the token limit: 0 of 686. Words of conversation before the email request (median): 450.0.

## Conversations the detector did not read as expected

| Conversation | Group | Should read | Outcome | Turns between | Tokens counted | Shares | Leans to |
|---|---|---|---|---|---|---|---|
| sim_0012 | indirect | M | no_mark | 6 | 138 | {'M': 0.2319, 'F': 0.2826} | F |
| sim_0013 | direct | M | no_mark | 4 | 93 | {'M': 0.4086, 'F': 0.1935} | M |
| sim_0022 | decoy | none | false_mark | 0 | 148 | {'M': 0.1351, 'F': 0.5811} | F |
| sim_0023 | direct | F | no_mark | 4 | 99 | {'M': 0.3232, 'F': 0.2424} | M |
| sim_0030 | direct | F | no_mark | 6 | 121 | {'M': 0.2149, 'F': 0.2314} | F |
| sim_0039 | decoy | none | false_mark | 2 | 125 | {'M': 0.416, 'F': 0.184} | M |
| sim_0042 | decoy | none | false_mark | 4 | 124 | {'M': 0.5726, 'F': 0.121} | M |
| sim_0046 | direct | F | no_mark | 6 | 115 | {'M': 0.2, 'F': 0.3913} | F |
| sim_0049 | direct | M | no_mark | 4 | 133 | {'M': 0.2707, 'F': 0.2707} | None |
| sim_0067 | indirect | F | no_mark | 4 | 113 | {'M': 0.177, 'F': 0.3363} | F |
| sim_0086 | indirect | F | no_mark | 6 | 143 | {'M': 0.2098, 'F': 0.3007} | F |
| sim_0088 | direct | F | no_mark | 0 | 108 | {'M': 0.213, 'F': 0.3889} | F |
| sim_0089 | neutral | none | false_mark | None | 134 | {'M': 0.3881, 'F': 0.2164} | M |
