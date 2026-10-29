"""Tests for ingestion/silver/chunking.py's heading-path markup stripping."""

from ingestion.silver.chunking import _build_path


def test_build_path_strips_mkdocs_anchor_id():
    """MkDocs {#...} anchors are stripped, the same way <a name> HTML tags
    already are - both are markup, neither is part of the heading's identity."""
    stack: list[tuple[int, str]] = []
    path = _build_path(
        stack,
        level=2,
        title="BaseModel instantiated {#base-model-instantiated}",
        doc_title="Usage Errors",
    )

    assert path == "Usage Errors > BaseModel instantiated"
