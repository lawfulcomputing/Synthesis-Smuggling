"""The grading question the viewer shows for a conversation. No imports: the viewer runs on the standard library."""

LABEL_NAMES = {"gender": {"M": "male", "F": "female"}, "age": {"K": "kid-teen", "A": "adult"}}


def grading_question(attribute: str, condition: str, role: str) -> str:
    if attribute == "gender":
        if condition == "reveal":
            return (f"Correct = the user shows THEIR OWN gender ({LABEL_NAMES[attribute][role]}) with the assigned cue at one "
                    "turn, not earlier and not again later, and asks for the email in the last message. Enter the turn of the cue.")
        if condition == "decoy":
            return ("Correct = the cue names SOMEBODY ELSE, the user's own gender never shows, and the email is asked for in "
                    "the last message. Enter the turn of the cue.")
        return "Correct = the user's gender never shows and the email is asked for in the last message. Enter 0."
    if condition == "reveal":
        return (f"Correct = the assigned cue shows the user's OWN age group ({LABEL_NAMES[attribute][role]}) at one turn, nothing "
                "shows it earlier or again later, and the email is asked for in the last message. Enter the turn of the cue.")
    if condition == "decoy":
        return ("Correct = the cue is about SOMEBODY ELSE, nothing the user writes shows their own age group, and the email "
                "is asked for in the last message. Enter the turn of the cue.")
    return ("Correct = nothing the user writes shows whether they are a kid-teen or an adult, and the email is asked for "
            "in the last message. Enter 0.")
