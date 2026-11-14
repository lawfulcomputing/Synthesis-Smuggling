# possessive_sample: report

## Requests

- Planned 552, responses 552, usable 552, rejected 0, requests that got no response (logged) 0.
- Tokens: 507,134 in, 229,048 out (110,596 reasoning). Cost $1.4111 (gpt-5.4-mini).

## Conversations

Planned 552, built 552, pass every hard check 552. Each cell is passing / built.

| Label | mail | near_miss | chat | train | validation | test |
|---|---|---|---|---|---|---|
| M | 227/227 | 0/0 | 0/0 | 180/180 | 22/22 | 25/25 |
| F | 225/225 | 0/0 | 0/0 | 177/177 | 25/25 | 23/23 |
| N | 100/100 | 0/0 | 0/0 | 80/80 | 7/7 | 13/13 |

## Checks

- Hard (conversation not used): none
- Review (kept): other_gender_word 28

## Email requests (one per unit)

- Form planned {'explicit': 552}, read from the text {'explicit': 552}.
- Size planned {'full': 552}; turns after the email {'0': 552}.
- Request length 50.0 words on average; reference email {'full': 93.1, 'short': None} words.
- Turns between the cue and the email: M {'1': 33, '2': 23, '3': 26, '4': 72, '5': 35, '7': 38}, F {'1': 32, '2': 22, '3': 24, '4': 76, '5': 34, '7': 37}.

Near-miss kinds: {}.

## Repeated wording in the new messages

Four-word sequences shared by the most units. A high share means the writer fell into a template.

- email_request: "an email to the" 71%; "draft an email to" 59%; "you draft an email" 59%; "can you draft an" 54%; "write an email to" 38%; "you write an email" 37%; "can you write an" 35%; "it should mention that" 27%
- followup: none
- near_miss: none
