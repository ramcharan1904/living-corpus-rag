"""Shared source-matching rule.

Used by both check_sources.py (does this golden source exist in the DB?)
and retrieval_eval.py (did retrieval return the golden source?), so the
rule can't drift between validating the golden set and scoring retrieval
against it - a looser rule in one place than the other would make either
a false "all clean" or a false "miss".
"""


def is_hit(result_paths: list[str], source: str) -> bool:
    """True if any result path is the source section or inside it.

    A path matches if it equals the source, or starts with the source
    followed by ' > ' or ' (part '. Plain "starts with" is not enough:
    "Aliases and more" must NOT match source "Alias".
    """
    for path in result_paths:
          if (
            path == source
            or path.startswith(source + " > ")
            or path.startswith(source + " (part ")
        ):
              return True
    return False
