# pairs_and_neutral: report

## Requests

- Planned 860, responses 860, usable 860, rejected 0, requests that got no response (logged) 0.
- Tokens: 1,140,759 in, 391,466 out (185,416 reasoning). Cost $2.6172 (gpt-5.4-mini).

## Conversations

Planned 1620, built 1620, pass every hard check 1560. Each cell is passing / built. K = kid-teen, A = adult, N = no age shown.

| Label | mail | near_miss | chat | train | validation | test |
|---|---|---|---|---|---|---|
| K | 297/320 | 107/110 | 110/110 | 391/411 | 66/67 | 57/62 |
| A | 297/320 | 107/110 | 110/110 | 391/411 | 66/67 | 57/62 |
| N | 315/320 | 107/110 | 110/110 | 432/438 | 45/46 | 55/56 |

By label and kind of source conversation (passing / built): A:pair 514/540, K:pair 514/540, N:neutral 532/540.

## Checks

- Hard (conversation not used): new_age_cue 39, title_in_pasted_email 1
- Review (kept): own_group_word 12, form_mismatch 2, reference_email_length 1
- Age words the new messages brought in, most frequent first: the office 9, to work 8, teacher 6, landlord 4, bedtime 3, meeting 3, permission slip 2, deadline 2, the principal 2, science fair 2, after work 1, commute 1, my commute 1, my taxes 1, taxes 1

## Email requests (one per unit)

- Form planned {'explicit': 448, 'implicit': 192}, read from the text {'explicit': 448, 'implicit': 192}.
- Size planned {'short': 128, 'full': 512}; turns after the email {'1': 256, '2': 128, '0': 256}.
- Request length 53.5 words on average; reference email {'full': 95.7, 'short': 50.9} words.
- Turns between the cue and the email: K {'1': 43, '2': 73, '3': 45, '4': 66, '5': 50, '6': 43}, A {'1': 43, '2': 73, '3': 45, '4': 66, '5': 50, '6': 43}.

Near-miss kinds: {'about_email': 75, 'own_notes': 73, 'received_email': 72}.

## Repeated wording in the new messages

Four-word sequences shared by the most units. A high share means the writer fell into a template.

- email_request: "an email to the" 54%; "you draft an email" 37%; "draft an email to" 34%; "can you draft an" 34%; "you write an email" 24%; "write an email to" 23%; "can you write an" 23%; "i want it to" 22%
- followup: "is it better to" 8%; "how long should i" 3%; "do you think it" 2%; "does that usually mean" 2%; "or should i just" 2%; "should i put the" 2%; "is it okay to" 2%; "or is that just" 2%
- near_miss: "you write me a" 29%; "can you write me" 26%; "what does that mean" 25%; "got this email from" 23%; "does that mean for" 22%; "that mean for me" 20%; "i got this email" 19%; "this email from the" 18%
