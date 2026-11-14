# unpaired: report

## Requests

- Planned 1474, responses 1474, usable 1474, rejected 0, requests that got no response (logged) 0.
- Tokens: 1,590,623 in, 587,898 out (263,934 reasoning). Cost $3.8385 (gpt-5.4-mini).

## Conversations

Planned 1474, built 1474, pass every hard check 1434. Each cell is passing / built. K = kid-teen, A = adult, N = no age shown.

| Label | mail | near_miss | chat | train | validation | test |
|---|---|---|---|---|---|---|
| K | 599/610 | 0/0 | 0/0 | 481/490 | 58/60 | 60/60 |
| A | 547/571 | 0/0 | 0/0 | 432/453 | 58/60 | 57/58 |
| N | 288/293 | 0/0 | 0/0 | 236/241 | 27/27 | 25/25 |

By label and kind of source conversation (passing / built): A:decoy 115/116, A:proxy 172/191, A:single 260/264, K:decoy 126/127, K:single 473/483, N:neutral 288/293.

## Checks

- Hard (conversation not used): new_age_cue 53
- Review (kept): own_group_word 77, form_mismatch 3, reference_email_length 6
- Age words the new messages brought in, most frequent first: teacher 46, my teacher 19, to work 7, meeting 5, my science teacher 5, deadline 5, school 4, my parent 4, my landlord 2, my son 2, learner’s permit 2, my client 2, my music teacher 2, the office 2, my shift 1

## Email requests (one per unit)

- Form planned {'explicit': 1474}, read from the text {'explicit': 1471, 'implicit': 3}.
- Size planned {'full': 1474}; turns after the email {'0': 1474}.
- Request length 58.9 words on average; reference email {'full': 93.5, 'short': None} words.
- Turns between the cue and the email: K {'1': 81, '2': 139, '3': 103, '4': 117, '5': 91, '6': 79}, A {'1': 82, '2': 119, '3': 92, '4': 94, '5': 102, '6': 82}.

Near-miss kinds: {}.

## Repeated wording in the new messages

Four-word sequences shared by the most units. A high share means the writer fell into a template.

- email_request: "an email to the" 52%; "you draft an email" 43%; "can you draft an" 41%; "you write an email" 40%; "draft an email to" 39%; "can you write an" 39%; "write an email to" 38%; "i want it to" 33%
- followup: none
- near_miss: none
