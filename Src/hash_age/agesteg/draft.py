"""Finding the draft in a reply: the text between the opening and the closing tag."""


def find_draft(text, open_tag, close_tag):
    """-> None when the reply has no opening tag, else
    {text, start, end, closed, extra}: the draft without surrounding whitespace, its character span in `text`,
    whether the closing tag was written, and whether a second opening tag follows (only the first draft counts).
    An unclosed draft runs to the end of the reply."""
    at = text.find(open_tag)
    if at < 0:
        return None
    start = at + len(open_tag)
    stop = text.find(close_tag, start)
    closed = stop >= 0
    end = stop if closed else len(text)
    body = text[start:end]
    lead, trail = len(body) - len(body.lstrip()), len(body) - len(body.rstrip())
    return {"text": body.strip(), "start": start + lead, "end": max(start + lead, end - trail), "closed": closed,
            "extra": text.find(open_tag, end) >= 0}
