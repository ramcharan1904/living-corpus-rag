"""Split markdown documents into heading-based chunks.

Sections are located by token, then sliced out of the original text by
line number — the markdown is never rebuilt from tokens, because
raw_text is hashed and must be byte-identical to the source.
"""

from dataclasses import dataclass

import tiktoken
from markdown_it import MarkdownIt

TARGET_TOKENS = 600
MAX_TOKENS = 800
MIN_TRAILING_PART_TOKENS = 100

_enc = tiktoken.get_encoding("cl100k_base")
_md = MarkdownIt()


@dataclass
class Chunk:
    heading_path: str
    chunk_index: int
    raw_text: str
    embed_text: str
    token_count: int


def _count(text: str) -> int:
    return len(_enc.encode(text))


def _frontmatter_end(lines: list[str]) -> int:
    """Return the first body line index, skipping YAML frontmatter.

    Two files in this corpus (pydantic_settings.md, migration.md, in all
    three versions) open with a --- delimited YAML block. markdown-it reads
    a bare --- as setext h2 syntax, so without this the `description:` line
    becomes a bogus heading.
    """
    if not lines or lines[0].strip() != "---":
        return 0

    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return i + 1

    return 0


def _heading_positions(tokens) -> list[tuple[int, int, str]]:
    """Return (start_line, level, title) for every h1/h2/h3."""
    positions = []

    for i, token in enumerate(tokens):
        if token.type != "heading_open":
            continue

        level = int(token.tag[1:])

        if level not in (1, 2, 3):
            continue

        inline_token = tokens[i + 1]
        start_line = token.map[0]

        positions.append((start_line, level, inline_token.content))

    return positions


def _build_path(stack: list[tuple[int, str]], level: int, title: str, doc_title: str) -> str:
    """Update the heading stack in place and return the full path."""
    while stack and stack[-1][0] >= level:
        stack.pop()

    stack.append((level, title))

    return " > ".join([doc_title] + [t for _, t in stack])


def _protected_ranges(tokens) -> list[tuple[int, int]]:
    """Line ranges that must never be split through.

    `fence` is a ``` block. `code_block` is a four-space-indented block,
    which is how code appears inside MkDocs admonitions (!!! note, ??? api).
    Hand-tracking backticks would miss every code_block.
    """
    return [
        tuple(t.map)
        for t in tokens
        if t.type in ("fence", "code_block", "table_open") and t.map
    ]


def _paragraph_blocks(text: str) -> list[str]:
    """Split on blank lines, but never inside a protected range."""
    lines = text.splitlines(keepends=True)
    protected = _protected_ranges(_md.parse(text))

    def in_protected(i: int) -> bool:
        return any(start <= i < end for start, end in protected)

    blocks: list[str] = []
    current: list[str] = []

    for i, line in enumerate(lines):
        current.append(line)

        if line.strip() == "" and not in_protected(i) and current:
            blocks.append("".join(current))
            current = []

    if current:
        blocks.append("".join(current))

    return blocks


def _split_oversized(raw: str, path: str) -> list[tuple[str, str]]:
    """Greedily pack paragraph blocks into parts under MAX_TOKENS.

    Parts are labelled 'part 1', 'part 2' — never 'part 1 of 3', because a
    section that is 2 parts in one version and 3 in another must still match
    part-for-part across versions.
    """
    if _count(raw) <= MAX_TOKENS:
        return [(path, raw)]

    parts: list[str] = []
    current = ""

    for block in _paragraph_blocks(raw):
        candidate = current + block

        if current and _count(candidate) > TARGET_TOKENS:
            parts.append(current)
            current = block
        else:
            current = candidate

    if current.strip():
        parts.append(current)

    # A small trailing remainder is rarely useful on its own — usually
    # anchor comments or a stray line. Fold it back into the previous part.
    if len(parts) > 1 and _count(parts[-1]) < MIN_TRAILING_PART_TOKENS:
        parts[-2] += parts[-1]
        parts.pop()

    if len(parts) == 1:
        return [(path, parts[0])]

    return [(f"{path} (part {i})", p) for i, p in enumerate(parts, start=1)]


def chunk_markdown(text: str, doc_title: str) -> list[Chunk]:
    lines = text.splitlines(keepends=True)
    tokens = _md.parse(text)

    body_start = _frontmatter_end(lines)
    positions = [p for p in _heading_positions(tokens) if p[0] >= body_start]

    sections: list[tuple[str, str]] = []

    # Text before the first heading is real content in this corpus
    # (intros, API links), so keep it under the document title alone.
    first_heading_line = positions[0][0] if positions else len(lines)
    preamble = "".join(lines[body_start:first_heading_line])

    if preamble.strip():
        sections.append((doc_title, preamble))

    stack: list[tuple[int, str]] = []

    for i, (start, level, title) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(lines)
        path = _build_path(stack, level, title, doc_title)
        raw = "".join(lines[start:end])

        if raw.strip():
            sections.append((path, raw))

    chunks: list[Chunk] = []

    for path, raw in sections:
        for part_path, part_raw in _split_oversized(raw, path):
            embed_text = f"{part_path}\n\n{part_raw}"
            chunks.append(
                Chunk(
                    heading_path=part_path,
                    chunk_index=len(chunks),
                    raw_text=part_raw,
                    embed_text=embed_text,
                    token_count=_count(part_raw),
                )
            )

    return chunks


if __name__ == "__main__":
    import sys
    from pathlib import Path

    path = Path(sys.argv[1])
    doc_title = path.stem.replace("_", " ").title()
    chunks = chunk_markdown(path.read_text(), doc_title)

    for c in chunks:
        flag = "  <<< OVER" if c.token_count > MAX_TOKENS else ""
        print(f"{c.token_count:5d}  {c.heading_path}{flag}")

    print("\n--- small chunks ---")
    for c in chunks:
        if c.token_count < 100:
            print(f"{c.token_count}  {c.heading_path}")
            print(repr(c.raw_text[:200]))
            print()

    print("--- unbalanced fences ---")
    for c in chunks:
        if c.raw_text.count("```") % 2 != 0:
            print("UNBALANCED:", c.heading_path)