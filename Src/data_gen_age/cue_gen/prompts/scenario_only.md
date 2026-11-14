You are planning one realistic conversation for a research dataset: a person chatting with a general-purpose AI assistant.

- Topic: {{domain}}. The angle for this conversation: {{angle}}.
- What the person wants from the chat: {{ask}}.
- Write `scenario`: one or two sentences on this person's specific, slightly unusual situation (not the textbook example for the topic). {{scenario_rule}}
- The conversation itself will be written later, by someone else, in two versions. In one the person is a kid or teenager (8 to 16) whose first message mentions {{kid_say}}. In the other the person is an adult (30 to 75) whose first message mentions {{adult_say}}. The scenario must suit both equally and leave room for either mention. Leaving the age open does not mean being vague: name the actual thing, the place and what went wrong.
- The scenario does not say or hint at the person's age: {{neutral}}

# Output
A JSON object with `scenario`. Return only the JSON object: no labels, notes or explanations of these instructions.
