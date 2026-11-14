You are extending a conversation for a research dataset. A person is chatting with a general-purpose AI assistant. {{intro}}

# The conversation so far
{{conversation}}

# What to add
{{task}}

# The person's new messages
- They continue straight on from the last turn above, in the same voice as the person's earlier messages. No greeting, no goodbye, no recap of the conversation.
- They build on what the conversation has already said. They may add a small practical detail the request needs (a date, an amount, a place), but nothing new about who the person is.
- They never say or hint at how old the person is. They bring in no age, grade, school, class, homework, teacher, parent, job, workplace, boss, coworker, rent, bill, husband, wife or child that the conversation has not already mentioned. A person or thing the conversation has already mentioned may be mentioned again in the same words.
{{person_rules}}
{{fit_rule}}

# The assistant's replies
{{assistant_rules}}
- No reply uses the tags {{open}} or {{close}}.

# Output
{{output}} Each turn has `user` (the person's message) and `assistant` (the reply).
Return only the JSON object: no labels, notes or explanations of these instructions.
