"""Tests for the two RAM WRITES stubs in retrieval/pgvector_store.py.

Both are specified here against their intended behavior, not their current
NotImplementedError bodies - they're expected to fail (error, not just
assert-fail) until implemented.

version_filter_sql tests are Postgres-backed (skip cleanly if unreachable,
via the shared pg_conn fixture). rrf is pure Python - no DB needed.
"""

import pytest

from retrieval.pgvector_store import rrf, version_filter_sql
from tests.conftest import FIXTURE_MARKER


def _insert_chunk(conn, doc_id, heading_path, valid_from_rank, valid_to_rank):
    with conn.transaction():
        row = conn.execute(
            """
            INSERT INTO chunks
                (doc_id, heading_path, chunk_index, raw_text, embed_text,
                 content_hash, valid_from_rank, valid_to_rank)
            VALUES (%s, %s, 0, 'text', 'text', %s, %s, %s)
            RETURNING chunk_id
            """,
            (doc_id, heading_path, heading_path, valid_from_rank, valid_to_rank),
        ).fetchone()
    return row[0]


def _is_valid_at(conn, chunk_id: int, target_rank: int) -> bool:
    where_sql = version_filter_sql()
    with conn.transaction():
        row = conn.execute(
            f"SELECT 1 FROM chunks c WHERE c.chunk_id = %(chunk_id)s AND ({where_sql})",
            {"chunk_id": chunk_id, "target_rank": target_rank},
        ).fetchone()
    return row is not None


def test_version_filter_sql_closed_interval(pg_conn, version_filter_doc):
    """A chunk valid 1-2 (closed at 2) is returned at ranks 1 and 2, not 3."""
    doc_id = version_filter_doc
    chunk_id = _insert_chunk(pg_conn, doc_id, f"{FIXTURE_MARKER}closed", valid_from_rank=1, valid_to_rank=2)

    assert _is_valid_at(pg_conn, chunk_id, 1) is True
    assert _is_valid_at(pg_conn, chunk_id, 2) is True
    assert _is_valid_at(pg_conn, chunk_id, 3) is False


def test_version_filter_sql_open_interval(pg_conn, version_filter_doc):
    """An open chunk from rank 2 (valid_to_rank NULL) is returned at 2 and 3, not 1."""
    doc_id = version_filter_doc
    chunk_id = _insert_chunk(pg_conn, doc_id, f"{FIXTURE_MARKER}open", valid_from_rank=2, valid_to_rank=None)

    assert _is_valid_at(pg_conn, chunk_id, 1) is False
    assert _is_valid_at(pg_conn, chunk_id, 2) is True
    assert _is_valid_at(pg_conn, chunk_id, 3) is True


def test_rrf_hand_computed():
    """rankings = [[1, 2, 3], [2, 3, 1]], k=60.

    chunk 1: rank 1 in list0 (1/61) + rank 3 in list1 (1/63) = 0.032266...
    chunk 2: rank 2 in list0 (1/62) + rank 1 in list1 (1/61) = 0.032522...
    chunk 3: rank 3 in list0 (1/63) + rank 2 in list1 (1/62) = 0.032002...
    Best first: 2, 1, 3 - chunk 2 wins on consensus (good-but-not-best rank
    in both lists) over chunk 1 (best rank in only one list).
    """
    rankings = [[1, 2, 3], [2, 3, 1]]
    expected_scores = {
        1: 1 / 61 + 1 / 63,
        2: 1 / 62 + 1 / 61,
        3: 1 / 63 + 1 / 62,
    }

    result = rrf(rankings, k=60)

    assert [chunk_id for chunk_id, _ in result] == [2, 1, 3]
    for chunk_id, score in result:
        assert score == pytest.approx(expected_scores[chunk_id])


def test_rrf_partial_overlap():
    """A chunk missing from a list contributes nothing for that list, not
    a penalty - it's still included, scored only from the lists it's in.

    rankings = [[10, 20], [20, 30]], k=60.
    chunk 10: rank 1 in list0 only = 1/61 = 0.016393...
    chunk 20: rank 2 in list0 (1/62) + rank 1 in list1 (1/61) = 0.032522...
    chunk 30: rank 2 in list1 only = 1/62 = 0.016129...
    Best first: 20 (in both lists), 10, 30.
    """
    rankings = [[10, 20], [20, 30]]
    expected_scores = {
        10: 1 / 61,
        20: 1 / 62 + 1 / 61,
        30: 1 / 62,
    }

    result = rrf(rankings, k=60)

    assert [chunk_id for chunk_id, _ in result] == [20, 10, 30]
    for chunk_id, score in result:
        assert score == pytest.approx(expected_scores[chunk_id])
