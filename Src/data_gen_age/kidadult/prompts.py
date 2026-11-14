"""{{name}} placeholders in the age generator's prompt parts; rendering fails loudly on any unfilled one."""
import re

_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def render(template, **values):
    missing = sorted({m for m in _PLACEHOLDER.findall(template) if m not in values})
    if missing:
        raise KeyError(f"template placeholders not provided: {missing}")
    return _PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), template)
