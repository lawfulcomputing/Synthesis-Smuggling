# Cue-turn run report

Responses: **4730** of 3250 planned units (cap 5200 responses), 1 rejected for structure. Requests sent: 4730, of which 0 units got no response on some attempt. Tokens: 4915696 in / 2695512 out (612360 reasoning). Cost: $15.8166 ($0.00398 per conversation).

Conversations: **3979** - groups {'N': 1000, 'K': 1489, 'A': 1490}, kinds {'neutral': 1000, 'single': 900, 'pair': 1480, 'proxy': 300, 'decoy': 299}, cue tiers {'role': 399, 'institution': 672, 'dependency': 775, 'proxy': 300, 'milestone': 268, 'direct': 216, 'means': 349}, splits {'train': 3176, 'test': 401, 'validation': 402}.

Lengths (user turns): {2: 389, 3: 328, 4: 456, 5: 554, 6: 660, 7: 515, 8: 386, 9: 282, 10: 409}. Cue turn: {1: 627, 2: 550, 3: 493, 4: 390, 5: 411, 6: 261, 7: 247}. Assistant-turn labels: {'N': 13260, 'K': 5181, 'A': 5196}. Marked turns by distance from the cue turn: {0: 2979, 1: 2594, 2: 1981, 3: 1488, 4: 915, 5: 420}.

## Checks

Pass every hard check: **3855/3979**. Twin pairs with both sides passing: 720/740.

By kind: {'decoy': '260/299', 'neutral': '963/1000', 'pair': '1460/1480', 'proxy': '288/300', 'single': '884/900'}. By group: {'A': '1436/1490', 'K': '1456/1489', 'N': '963/1000'}. By cue tier: {'dependency': '761/775', 'direct': '210/216', 'institution': '637/672', 'means': '341/349', 'milestone': '265/268', 'none': '963/1000', 'proxy': '288/300', 'role': '390/399'}.

Hard problems: {'assistant_first': 24, 'early_cue': 25, 'cue_missing': 37, 'decoy_missing': 47, 'debris': 3}. Review flags: {'opposite_cue': 24, 'assistant_adds': 53} (in 71 conversations). Marker code points removed from the text: 0.

## Against the gender reference

| measure | gender set | target | this run |
|---|---|---|---|
| pair turns before the cue turn identical | 100% | 100% | 100.0% |
| word overlap of the two cue-turn messages | 0.32 | at most 0.40 | 0.211 |
| later user messages identical across a pair | 0.1% | under 2% | 0.0% |
| share of revealed conversations in a pair | 48% | about 50% | 49.7% |
| share of conversations that never reveal | 27% | about 25% | 25.1% |
| different cues per group (K / A) | 23 / 23 | at least 23 each | 32 / 41 |
| share of the five most used cues (K / A) | 36% / 40% | at most 40% | 21% / 19% |
| share of revealed conversations said outright | 7% | about 7% | 7.3% |
| user words per message after the cue, K / A | - | 0.80 to 1.25 | 0.99 |

Never-reveal conversations: 1000, of which 350 mention someone else's kid-world or adult-world thing. Pairs stopped at the shared part (their two sides were not requested): 10 ({'early_cue': 8, 'assistant_first': 1, 'debris': 1}).

## Twins (from the cue turn on; before it the two sides are the same text)

| measure | value |
|---|---|
| turns_before_cue_identical | 1973/1973 |
| word_overlap_cue_turn_user | 0.211 |
| word_overlap_later_user | 0.127 |
| word_overlap_assistant | 0.176 |
| assistant_replies_identical | 0.0 |
| later_user_messages_identical | 0.0 |

## Cues

| group | different cues | share of the most used | share of the top five | most used |
|---|---|---|---|---|
| K | 32 | 0.042 | 0.208 | {'k_teacher': 62, 'k_babysitter': 62, 'k_principal': 62, 'k_parents': 62, 'k_grounded': 61} |
| A | 41 | 0.04 | 0.194 | {'a_grandkids': 59, 'a_own_kids': 59, 'a_boss': 58, 'a_coworkers': 57, 'a_spouse': 56} |

## Length and wording shortcuts

| measure | value |
|---|---|
| user_words_per_message | {'K': 23.282, 'A': 22.982, 'N': 17.751} |
| assistant_words_per_reply | {'K': 41.293, 'A': 41.301, 'N': 35.601} |
| user_words_per_message_from_the_cue_turn | {'K': 24.888, 'A': 25.043} |
| user_messages_with_em_dash | 0.0 |
| later_user_messages_with_own_group_word | 0.024 |

## Repeats

Of 3239 units: 0 open with a first message another unit already used; 0 have a scenario sharing most of its words with an earlier one on the same topic (first pairs: []).

## Phrases only one group uses

Three-word phrases in the person's messages found in at least 15% of one group's conversations and at most a fifth as often in the other. Anything here gives the group away apart from the planned cue.

None.

## Every problem and flag

| conversation | kind | code | turn | detail |
|---|---|---|---|---|
| u2426-N | neutral | assistant_first | 4 | parents / family: family |
| u1787-A | proxy | early_cue | 1 | K: allowance |
| u1883-A | proxy | early_cue | 1 | K: allowance |
| u0296-A | pair | cue_missing | 3 | work |
| u0108-K | pair | opposite_cue | 7 | meeting |
| u2219-A | decoy | decoy_missing | 1 | my neighbor's kid's school bus |
| u2219-A | decoy | opposite_cue | 2 | homework |
| u0237-K | pair | assistant_adds | 6 | adult life: your job |
| u1166-K | single | assistant_adds | 1 | school: teacher |
| u1303-K | single | assistant_adds | 1 | parents / family: parent |
| u0945-K | single | cue_missing | 3 | school |
| u0624-K | pair | assistant_adds | 1 | age words: grown-up |
| u2127-A | decoy | decoy_missing | 3 | my neighbor's kid's school bus |
| u2187-A | decoy | decoy_missing | 1 | my neighbor's kid's school bus |
| u2164-A | decoy | opposite_cue | 6 | sleepover |
| u2048-K | decoy | decoy_missing | 1 | my uncle's boss |
| u3176-N | neutral | assistant_first | 1 | parents / family: family |
| u3176-N | neutral | assistant_first | 5 | parents / family: family |
| u1774-A | proxy | assistant_first | 4 | school: homework |
| u2716-N | neutral | decoy_missing | 3 | my neighbor's kid's school bus |
| u0468-A | pair | assistant_adds | 8 | parents / family: family |
| u2590-N | neutral | decoy_missing | 1 | my cousin's paycheck |
| u1211-K | single | cue_missing | 4 | teenager |
| u1128-K | single | assistant_adds | 5 | parents / family: family |
| u1128-K | single | assistant_adds | 6 | parents / family: family |
| u0364-A | pair | cue_missing | 4 | dental hygienist |
| u2711-N | neutral | decoy_missing | 3 | my neighbor's kid's bedtime |
| u2818-N | neutral | decoy_missing | 2 | my neighbor's commute |
| u2152-A | decoy | opposite_cue | 4 | homework |
| u2073-K | decoy | decoy_missing | 1 | my neighbor's commute |
| u0385-K | pair | cue_missing | 2 | school |
| u1621-A | single | opposite_cue | 1 | school |
| u1087-K | single | assistant_adds | 2 | school: teacher |
| u2119-A | decoy | decoy_missing | 2 | my neighbor's kid's sleepover |
| u2162-A | decoy | decoy_missing | 1 | my neighbor's kid's school bus |
| u2162-A | decoy | cue_missing | 2 | work |
| u0907-K | single | opposite_cue | 1 | taxes |
| u0907-K | single | opposite_cue | 4 | taxes |
| u0690-K | pair | assistant_adds | 3 | school: teacher |
| u1830-A | proxy | early_cue | 1 | A: meetings |
| u1138-K | single | opposite_cue | 3 | meeting |
| u1720-A | proxy | early_cue | 2 | K: my teacher |
| u1720-A | proxy | early_cue | 2 | K: teacher |
| u0036-K | pair | cue_missing | 3 | school |
| u2042-K | decoy | cue_missing | 3 | school |
| u1014-K | single | assistant_adds | 1 | adult life: your work |
| u1859-A | proxy | early_cue | 1 | K: allowance |
| u0058-A | pair | assistant_adds | 8 | adult life: paycheck |
| u0702-K | pair | cue_missing | 3 | old enough |
| u0041-A | pair | cue_missing | 7 | work |
| u2044-K | decoy | opposite_cue | 9 | coworker |
| u3161-N | neutral | decoy_missing | 2 | my neighbor's commute |
| u2156-A | decoy | decoy_missing | 1 | my neighbor's kid's sleepover |
| u2192-A | decoy | decoy_missing | 1 | my neighbor's kid's school bus |
| u0725-A | pair | cue_missing | 2 | work |
| u0079-A | pair | opposite_cue | 1 | school |
| u2175-A | decoy | decoy_missing | 1 | my neighbor's kid's school bus |
| u1479-A | single | assistant_first | 4 | age words: adult |
| u2031-K | decoy | decoy_missing | 1 | my neighbor's commute |
| u2009-K | decoy | assistant_first | 3 | school: homework |
| u1343-K | single | assistant_adds | 7 | adult life: your job |
| u2172-A | decoy | decoy_missing | 5 | my neighbor's kid's bedtime |
| u0210-K | pair | assistant_adds | 5 | school: teacher |
| u1476-A | single | opposite_cue | 1 | teacher |
| u1912-A | proxy | assistant_adds | 6 | school: homework |
| u2146-A | decoy | assistant_first | 3 | age words: grown-up |
| u0543-K | pair | assistant_adds | 4 | age words: adult |
| u2076-K | decoy | assistant_first | 1 | school: school |
| u1122-K | single | assistant_adds | 1 | school: homework |
| u1203-K | single | opposite_cue | 1 | meetings |
| u1203-K | single | assistant_adds | 4 | school: teacher |
| u1203-K | single | assistant_adds | 5 | school: teacher |
| u0167-K | pair | opposite_cue | 1 | my apartment |
| u0099-A | pair | opposite_cue | 1 | school |
| u0863-K | single | assistant_first | 4 | age words: kids |
| u1757-A | proxy | early_cue | 1 | K: allowance |
| u0844-K | single | assistant_adds | 9 | age words: adult |
| u1099-K | single | assistant_adds | 4 | school: homework |
| u0533-K | pair | cue_missing | 2 | kid |
| u2248-A | decoy | early_cue | 3 | K: bedtime |
| u2248-A | decoy | decoy_missing | 4 | my neighbor's kid's bedtime |
| u2248-A | decoy | early_cue | 5 | A: credit card |
| u2248-A | decoy | cue_missing | 6 | credit card |
| u2242-A | decoy | debris | 1 | ou can do around the house?},{ |
| u2242-A | decoy | decoy_missing | 3 | my neighbor's kid's school bus |
| u0566-A | pair | assistant_adds | 7 | age words: adult |
| u1488-A | single | cue_missing | 1 | client |
| u1775-A | proxy | early_cue | 4 | K: allowance |
| u3151-N | neutral | assistant_first | 3 | parents / family: mother |
| u0636-K | pair | cue_missing | 6 | school |
| u1543-A | single | cue_missing | 2 | the office |
| u2218-A | decoy | decoy_missing | 1 | my neighbor's kid's school bus |
| u0213-K | pair | cue_missing | 7 | school |
| u1894-A | proxy | assistant_first | 2 | school: homework |
| u1979-K | decoy | assistant_first | 5 | parents / family: family |
| u0941-K | single | assistant_adds | 1 | school: homework |
| u1028-K | single | assistant_adds | 4 | school: homework |
| u0006-A | pair | assistant_adds | 3 | parents / family: father |
| u1957-K | decoy | assistant_adds | 8 | adult life: your work |
| u1500-A | single | early_cue | 1 | A: meeting |
| u2012-K | decoy | assistant_first | 3 | age words: adults |
| u1369-A | single | assistant_adds | 4 | school: school |
| u2110-A | decoy | decoy_missing | 1 | my neighbor's kid's school bus |
| u2176-A | decoy | decoy_missing | 2 | my neighbor's kid's school bus |
| u1967-K | decoy | early_cue | 4 | A: for work |
| u0592-K | pair | cue_missing | 2 | permission slip |
| u1344-K | single | assistant_adds | 3 | age words: adult |
| u1810-A | proxy | early_cue | 5 | K: teacher |
| u2203-A | decoy | decoy_missing | 1 | my neighbor's kid's sleepover |
| u2277-N | neutral | decoy_missing | 4 | my uncle's shift |
| u2098-K | decoy | opposite_cue | 5 | landlord |
| u0898-K | single | cue_missing | 4 | school |
| u2118-A | decoy | decoy_missing | 1 | my neighbor's kid's school bus |
| u0970-K | single | assistant_adds | 4 | parents / family: family |
| u0623-A | pair | opposite_cue | 3 | school |
| u1059-K | single | assistant_first | 1 | parents / family: family |
| u2260-N | neutral | debris | 1 | or one that feels more new?},{ |
| u0567-K | pair | assistant_adds | 1 | age words: adult |
| u0567-A | pair | assistant_adds | 1 | age words: kids |
| u0282-K | pair | assistant_adds | 5 | age words: adult |
| u1969-K | decoy | decoy_missing | 1 | my neighbor's commute |
| u1226-K | single | cue_missing | 3 | school |
| u2142-A | decoy | decoy_missing | 1 | my neighbor's kid's school bus |
| u2517-N | neutral | decoy_missing | 8 | my cousin's paycheck |
| u1482-A | single | cue_missing | 1 | work |
| u2215-A | decoy | decoy_missing | 2 | my neighbor's kid's sleepover |
| u0245-A | pair | opposite_cue | 4 | school |
| u2179-A | decoy | decoy_missing | 1 | my neighbor's kid's sleepover |
| u2224-A | decoy | decoy_missing | 1 | my neighbor's kid's school bus |
| u0915-K | single | assistant_first | 1 | age words: adult |
| u0915-K | single | cue_missing | 5 | school |
| u1548-A | single | cue_missing | 1 | coworker |
| u1327-K | single | assistant_adds | 7 | adult life: your work |
| u2323-N | neutral | early_cue | 6 | A: arthritis |
| u1179-K | single | assistant_adds | 3 | age words: adult |
| u0548-A | pair | assistant_adds | 6 | parents / family: family |
| u0277-A | pair | cue_missing | 1 | work |
| u3201-N | neutral | decoy_missing | 1 | my neighbor's kid's sleepover |
| u0626-K | pair | assistant_adds | 5 | parents / family: parent |
| u2616-N | neutral | assistant_first | 4 | parents / family: family |
| u0190-K | pair | cue_missing | 6 | school |
| u0527-K | pair | assistant_adds | 8 | age words: adult |
| u2066-K | decoy | opposite_cue | 4 | landlord |
| u2232-A | decoy | cue_missing | 2 | nurse |
| u2141-A | decoy | decoy_missing | 1 | my neighbor's kid's bedtime |
| u2141-A | decoy | early_cue | 1 | A: my apartment |
| u0391-A | pair | cue_missing | 6 | work |
| u2638-N | neutral | decoy_missing | 1 | my neighbor's kid's sleepover |
| u2020-K | decoy | cue_missing | 3 | school |
| u2765-N | neutral | early_cue | 7 | A: expecting a |
| u1315-K | single | cue_missing | 3 | school |
| u0821-K | single | assistant_adds | 1 | school: homework |
| u3191-N | neutral | assistant_first | 1 | parents / family: mother |
| u0481-K | pair | assistant_adds | 7 | adult life: your job |
| u0646-A | pair | opposite_cue | 5 | school |
| u3061-N | neutral | decoy_missing | 4 | my uncle's boss |
| u1054-K | single | assistant_adds | 1 | age words: adult |
| u2671-N | neutral | assistant_first | 1 | parents / family: mother |
| u3021-N | neutral | decoy_missing | 1 | my neighbor's kid's sleepover |
| u2317-N | neutral | assistant_first | 4 | parents / family: family |
| u2655-N | neutral | early_cue | 1 | K: teacher |
| u0207-K | pair | assistant_adds | 4 | age words: adult |
| u3038-N | neutral | decoy_missing | 1 | my neighbor's kid's sleepover |
| u1697-A | proxy | early_cue | 1 | A: my son |
| u0445-A | pair | cue_missing | 4 | client |
| u0541-A | pair | assistant_adds | 8 | parents / family: mum |
| u1229-K | single | assistant_adds | 5 | school: teacher |
| u3128-N | neutral | assistant_first | 2 | age words: adult |
| u0247-K | pair | cue_missing | 6 | school |
| u2163-A | decoy | cue_missing | 7 | work |
| u2653-N | neutral | decoy_missing | 1 | my neighbor's kid's sleepover |
| u3177-N | neutral | decoy_missing | 1 | my neighbor's kid's school bus |
| u2369-N | neutral | decoy_missing | 1 | my cousin's teacher |
| u2092-K | decoy | assistant_adds | 5 | parents / family: family |
| u1249-K | single | opposite_cue | 6 | retire |
| u1889-A | proxy | early_cue | 1 | K: allowance |
| u1246-K | single | assistant_adds | 1 | school: teacher |
| u0074-A | pair | cue_missing | 7 | work |
| u0789-K | single | assistant_adds | 4 | school: homework |
| u1062-K | single | cue_missing | 1 | school |
| u0495-K | pair | cue_missing | 6 | school |
| u2810-N | neutral | decoy_missing | 6 | my neighbor's kid's school bus |
| u0634-K | pair | cue_missing | 7 | school |
| u0634-K | pair | assistant_adds | 7 | age words: adult |
| u0634-K | pair | assistant_adds | 10 | age words: adult |
| u2070-K | decoy | decoy_missing | 1 | my cousin's paycheck |
| u3239-N | neutral | decoy_missing | 2 | my neighbor's kid's sleepover |
| u0313-A | pair | assistant_adds | 2 | adult life: your job |
| u2729-N | neutral | decoy_missing | 1 | my neighbor's kid's school bus |
| u0596-K | pair | assistant_adds | 5 | school: school |
| u2794-N | neutral | early_cue | 5 | K: recess |
| u0956-K | single | assistant_adds | 1 | school: teacher |
| u2475-N | neutral | early_cue | 4 | A: deadline |
| u1991-K | decoy | debris | 1 | even more. What’s the riddle?{ |
| u2185-A | decoy | early_cue | 3 | K: bedtime |
| u2185-A | decoy | assistant_first | 4 | parents / family: parental |
| u0834-K | single | assistant_adds | 1 | school: teacher |
| u0267-A | pair | opposite_cue | 6 | school |
| u2528-N | neutral | assistant_first | 5 | age words: adult |
| u0160-K | pair | opposite_cue | 7 | meeting |
| u0127-K | pair | assistant_adds | 4 | age words: adult |
| u0127-K | pair | opposite_cue | 7 | landlord |
| u1108-K | single | assistant_adds | 5 | age words: adult |
| u0163-A | pair | cue_missing | 1 | work |
| u2315-N | neutral | assistant_first | 2 | age words: kids |
| u2003-K | decoy | opposite_cue | 9 | my business |
| u2322-N | neutral | early_cue | 6 | A: expecting a |
| u1575-A | single | assistant_first | 5 | age words: adults |
| u2123-A | decoy | decoy_missing | 2 | my neighbor's kid's school bus |
| u2279-N | neutral | early_cue | 3 | K: school |
| u2279-N | neutral | early_cue | 3 | K: school bus |
| u2279-N | neutral | decoy_missing | 4 | my neighbor's kid's school bus |
| u2941-N | neutral | decoy_missing | 2 | my uncle's boss |
