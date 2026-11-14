# pairs_and_neutral: report

## Requests

- Planned 860, responses 860, usable 860, rejected 0, requests that got no response (logged) 0.
- Tokens: 921,415 in, 366,350 out (160,097 reasoning). Cost $2.3396 (gpt-5.4-mini).

## Conversations

Planned 1620, built 1620, pass every hard check 1618. Each cell is passing / built.

| Label | mail | near_miss | chat | train | validation | test |
|---|---|---|---|---|---|---|
| M | 319/320 | 110/110 | 110/110 | 400/401 | 67/67 | 72/72 |
| F | 319/320 | 110/110 | 110/110 | 400/401 | 67/67 | 72/72 |
| N | 320/320 | 110/110 | 110/110 | 431/431 | 55/55 | 54/54 |

## Checks

- Hard (conversation not used): self_gender_word 2
- Review (kept): other_gender_word 64, form_mismatch 10

## Email requests (one per unit)

- Form planned {'explicit': 448, 'implicit': 192}, read from the text {'explicit': 445, 'implicit': 195}.
- Size planned {'short': 128, 'full': 512}; turns after the email {'0': 256, '2': 128, '1': 256}.
- Request length 50.4 words on average; reference email {'full': 95.1, 'short': 49.8} words.
- Turns between the cue and the email: M {'1': 102, '2': 63, '3': 60, '4': 34, '5': 30, '6': 14, '7': 10, '8': 5, '9': 2}, F {'1': 102, '2': 63, '3': 60, '4': 34, '5': 30, '6': 14, '7': 10, '8': 5, '9': 2}.

Near-miss kinds: {'about_email': 76, 'own_notes': 73, 'received_email': 71}.

## Repeated wording in the new messages

Four-word sequences shared by the most units. A high share means the writer fell into a template.

- email_request: "an email to the" 48%; "draft an email to" 47%; "you draft an email" 43%; "can you draft an" 35%; "i need to email" 21%; "two or three sentences" 19%; "write an email to" 19%; "it should mention that" 19%
- followup: "what s the easiest" 6%; "if i want to" 6%; "the easiest way to" 6%; "s the easiest way" 6%; "what s the best" 5%; "is it better to" 4%; "what s a good" 4%; "the best way to" 4%
- near_miss: "what should i do" 25%; "got this email from" 25%; "and what should i" 23%; "mean for me and" 23%; "way to reach them" 23%; "how long should i" 22%; "long should i wait" 22%; "you write me a" 22%
