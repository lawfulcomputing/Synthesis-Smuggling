# possessive_rest: report

## Requests

- Planned 841, responses 841, usable 841, rejected 0, requests that got no response (logged) 0.
- Tokens: 757,151 in, 349,556 out (169,219 reasoning). Cost $2.1409 (gpt-5.4-mini).

## Conversations

Planned 841, built 841, pass every hard check 841. Each cell is passing / built.

| Label | mail | near_miss | chat | train | validation | test |
|---|---|---|---|---|---|---|
| M | 305/305 | 0/0 | 0/0 | 243/243 | 35/35 | 27/27 |
| F | 193/193 | 0/0 | 0/0 | 158/158 | 17/17 | 18/18 |
| N | 343/343 | 0/0 | 0/0 | 276/276 | 35/35 | 32/32 |

## Checks

- Hard (conversation not used): none
- Review (kept): other_gender_word 60, reference_email_length 1

## Email requests (one per unit)

- Form planned {'explicit': 841}, read from the text {'explicit': 841}.
- Size planned {'full': 841}; turns after the email {'0': 841}.
- Request length 49.2 words on average; reference email {'full': 93.7, 'short': None} words.
- Turns between the cue and the email: M {'1': 33, '2': 69, '3': 70, '4': 37, '5': 49, '7': 47}, F {'1': 17, '2': 34, '3': 33, '4': 16, '5': 48, '7': 45}.

Near-miss kinds: {}.

## Repeated wording in the new messages

Four-word sequences shared by the most units. A high share means the writer fell into a template.

- email_request: "an email to the" 67%; "you draft an email" 59%; "draft an email to" 59%; "can you draft an" 55%; "write an email to" 38%; "you write an email" 36%; "can you write an" 33%; "it should mention that" 23%
- followup: none
- near_miss: none
