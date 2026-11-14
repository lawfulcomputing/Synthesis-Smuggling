You are writing {{what}} for a research dataset: a person chatting with a general-purpose AI assistant. Each turn is one message from the person and one reply from the assistant.

# Situation
- Topic: {{domain}}. The angle for this conversation: {{angle}}.
- What the person wants from the chat: {{ask}}.
- {{scenario_line}}

# The person
- Temperament: {{temperament}}
- Message length: {{length}} {{first_message}}
- They write the way people really type to a chatbot: plain everyday words, contractions, no em dashes, no lists, no formal sign-offs. Use normal spelling, capitals and apostrophes; do not imitate typos.
- {{voice}}
- They react to what the assistant just said (skim it, pick up one part, push back, go off on a small tangent), bring something new each turn, and do not wrap up or say goodbye before the last turn.
- If they name anyone, they use one of these names: {{names}}.

# The assistant
- Short, conversational replies: usually two to four sentences and under 90 words. A short list only when the person asks for steps or options.
- It never guesses or hints at the person's age or stage of life, and it never brings up school, parents, family, work, a partner or children unless the person has already mentioned that same thing.
- {{assistant_after}}
- It is an AI and speaks as one.

# {{age_heading}}
{{age_rules}}

{{so_far}}

# Output
{{output}} Each turn has `turn` (its number), `user` (the person's message) and `assistant` (the reply).
Return only the JSON object: no labels, notes or explanations of these instructions.
