"""Content hashing for chunk change detection.

The hash must be stable against cosmetic edits (a reflowed paragraph, a
trailing space) but sensitive to real ones. So prose whitespace is
collapsed before hashing and code blocks are left byte-for-byte —
indentation is semantic in Python, and normalizing it would make half the
corpus look changed on every version.
"""

import hashlib
import re

from markdown_it import MarkdownIt

from ingestion.silver.chunking import _protected_ranges

_md = MarkdownIt()
_WHITESPACE_RUN = re.compile(r"[ \t]+")


def normalize(text: str) -> str:
    """Collapse prose whitespace, leaving protected blocks untouched.

    Protected blocks are code fences, indented code blocks and tables —
    whatever `_protected_ranges` reports for this text. Within prose,
    consecutive non-blank lines are joined into one line, so a soft-wrapped
    paragraph hashes the same however its line breaks fall.
    """
    lines = text.splitlines()
    protected = _protected_ranges(_md.parse(text))

    def in_protected(i: int) -> bool:
        return any(start <= i < end for start, end in protected)

    out: list[str] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            out.append(" ".join(paragraph))
            paragraph.clear()

    for i, line in enumerate(lines):
        if in_protected(i):
            flush_paragraph()
            out.append(line)
            continue

        collapsed = _WHITESPACE_RUN.sub(" ", line).strip()

        if not collapsed:
            flush_paragraph()
            # Drop consecutive blank lines: one blank line is a paragraph
            # break, more than one is just formatting.
            if out and not out[-1]:
                continue
            out.append("")
            continue

        paragraph.append(collapsed)

    flush_paragraph()

    while out and not out[0]:
        out.pop(0)
    while out and not out[-1]:
        out.pop()

    return "\n".join(out)


def content_hash(text: str) -> str:
    """SHA-256 of the normalized text, as a hex string."""
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    original = """## Validators

A validator runs after parsing.

```python
def check(v):
    return v
```

Done.
"""

    reflowed = """## Validators

A validator   runs
after parsing.


```python
def check(v):
    return v
```

Done.
"""

    code_changed = """## Validators

A validator runs after parsing.

```python
def check(v):
        return v
```

Done.
"""

    h1 = content_hash(original)
    h2 = content_hash(reflowed)
    h3 = content_hash(code_changed)

    print("prose reflow ignored:     ", h1 == h2)
    print("code indent change caught:", h1 != h3)
