"""Tests for eval/matching.py's is_hit.

Specified here against its intended behavior, not its current
NotImplementedError body - expected to fail (error, not assert-fail)
until implemented. Pure Python, no DB needed.
"""

from eval.matching import is_hit


def test_exact_match():
    assert is_hit(["Alias"], "Alias") is True


def test_child_path():
    assert is_hit(["Alias > Alias Configuration > ConfigDict Settings"], "Alias") is True


def test_part_suffix():
    assert is_hit(["Alias (part 2)"], "Alias") is True


def test_false_prefix():
    """Plain str.startswith is not enough - "Aliases" shares a prefix with
    "Alias" but is a different section, not a child of it."""
    assert is_hit(["Aliases and more"], "Alias") is False


def test_no_match():
    assert is_hit(["Fields > Field aliases"], "Alias") is False


def test_empty_list():
    assert is_hit([], "Alias") is False
