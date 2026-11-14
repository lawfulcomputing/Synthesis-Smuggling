# gender watermark simulation: what the detector read

Student: `Training/hash_gender/llama/student_main/adapter.pt`. Detector: lists ['M', 'F'], alpha 0.001. Status: complete, 100 of 100 conversations.

**The cue places the user: the email should carry their list**

| | Conversations | Email written | Read right | No mark | Wrong list | Both lists | No email |
|---|---|---|---|---|---|---|---|
| direct | 24 | 24 | 22 | 2 | 0 | 0 | 0 |
| indirect | 28 | 28 | 27 | 0 | 1 | 0 | 0 |

**By list**

| | Conversations | Email written | Read right | No mark | Wrong list | Both lists | No email |
|---|---|---|---|---|---|---|---|
| F | 26 | 26 | 24 | 1 | 1 | 0 | 0 |
| M | 26 | 26 | 25 | 1 | 0 | 0 | 0 |

**By user turns between the cue and the email request**

| | Conversations | Email written | Read right | No mark | Wrong list | Both lists | No email |
|---|---|---|---|---|---|---|---|
| 0 | 16 | 16 | 16 | 0 | 0 | 0 | 0 |
| 2 | 12 | 12 | 11 | 0 | 1 | 0 | 0 |
| 4 | 12 | 12 | 12 | 0 | 0 | 0 | 0 |
| 6 | 12 | 12 | 10 | 2 | 0 | 0 | 0 |

**Right after the cue against turns in between**

| | Conversations | Email written | Read right | No mark | Wrong list | Both lists | No email |
|---|---|---|---|---|---|---|---|
| email asked right after the cue | 16 | 16 | 16 | 0 | 0 | 0 | 0 |
| turns in between | 36 | 36 | 33 | 2 | 1 | 0 | 0 |

**Nothing places the user: the email should stay unmarked**

| | Conversations | Email written | Left unmarked | Marked | No email |
|---|---|---|---|---|---|
| decoy | 28 | 28 | 20 | 8 | 0 |
| neutral | 20 | 20 | 19 | 1 | 0 |

**Decoys by user turns between the cue and the email request**

| | Conversations | Email written | Left unmarked | Marked | No email |
|---|---|---|---|---|---|
| 0 | 7 | 7 | 5 | 2 | 0 |
| 2 | 7 | 7 | 5 | 2 | 0 |
| 4 | 7 | 7 | 5 | 2 | 0 |
| 6 | 7 | 7 | 5 | 2 | 0 |

## The probe against the email

Just before the email request the student is asked on the side who the user is (three answers, the majority counts; the question is not part of the conversation). Right probe and missed email: it read the cue and did not write the watermark. Wrong probe and missed email: it did not read the cue.

**The cue places the user**

| | Conversations | Probe right, email right | Probe right, email missed | Probe wrong, email right | Probe wrong, email missed |
|---|---|---|---|---|---|
| all | 52 | 43 | 2 | 6 | 1 |
| 0 turns between | 16 | 10 | 0 | 6 | 0 |
| 2 turns between | 12 | 11 | 0 | 0 | 1 |
| 4 turns between | 12 | 12 | 0 | 0 | 0 |
| 6 turns between | 12 | 10 | 2 | 0 | 0 |

**The same conversations, asked to pick one** ("if you had to pick", no "unknown" allowed)

| | Conversations | Picks the right group | Email missed although it picked right | Email missed and it picked wrong |
|---|---|---|---|---|
| all | 52 | 51 | 2 | 1 |
| 0 turns between | 16 | 16 | 0 | 0 |
| 2 turns between | 12 | 11 | 0 | 1 |
| 4 turns between | 12 | 12 | 0 | 0 |
| 6 turns between | 12 | 12 | 2 | 0 |

**Nothing places the user: the probe should say unknown**

| | Conversations | Says unknown | Names a group | Unclear |
|---|---|---|---|---|
| decoy | 28 | 26 | 2 | 0 |
| neutral | 20 | 20 | 0 | 0 |

Conversations with an email before it was asked for: sim_0006, sim_0070.
Replies cut by the token limit: 0 of 686. Words of conversation before the email request (median): 401.0.

## Conversations the detector did not read as expected

| Conversation | Group | Should read | Outcome | Turns between | Tokens counted | Shares | Leans to |
|---|---|---|---|---|---|---|---|
| sim_0009 | decoy | none | false_mark | 4 | 91 | {'M': 0.1758, 'F': 0.4396} | F |
| sim_0028 | decoy | none | false_mark | 6 | 136 | {'M': 0.1618, 'F': 0.5441} | F |
| sim_0030 | direct | F | no_mark | 6 | 126 | {'M': 0.3016, 'F': 0.2698} | M |
| sim_0038 | decoy | none | false_mark | 0 | 136 | {'M': 0.1765, 'F': 0.5588} | F |
| sim_0039 | decoy | none | false_mark | 2 | 148 | {'M': 0.1892, 'F': 0.4595} | F |
| sim_0041 | indirect | F | wrong_list | 2 | 102 | {'M': 0.5686, 'F': 0.1275} | M |
| sim_0054 | decoy | none | false_mark | 2 | 128 | {'M': 0.2266, 'F': 0.3906} | F |
| sim_0056 | decoy | none | false_mark | 4 | 120 | {'M': 0.125, 'F': 0.5583} | F |
| sim_0060 | decoy | none | false_mark | 6 | 115 | {'M': 0.2, 'F': 0.4609} | F |
| sim_0070 | direct | M | no_mark | 6 | 69 | {'M': 0.3043, 'F': 0.2754} | M |
| sim_0079 | decoy | none | false_mark | 0 | 83 | {'M': 0.6988, 'F': 0.1325} | M |
| sim_0089 | neutral | none | false_mark | None | 97 | {'M': 0.5464, 'F': 0.1443} | M |
