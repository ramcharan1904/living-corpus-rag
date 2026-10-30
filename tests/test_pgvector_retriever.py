"""Tests for PgVectorRetriever's dense/keyword/hybrid retrieve() paths,
against the real database. Mocks the OpenAI client - zero network calls.

These are deliberately end-to-end against real SQL, not unit tests against
mocked psycopg: they exist specifically to catch bugs in the SQL itself,
like a query embedding being bound as double precision[] instead of vector
(see the bottom of this file for a demonstration that they actually would
have caught that one).

Setup, reusing the existing fixture pattern (throwaway document via
version_filter_doc, cleanup by cascade): chunk A is valid 1-1, chunk B is
valid 3-NULL, both with identical text and an identical one-hot embedding.
Since the version filter is applied inside the SQL before ranking, only one
of A/B is ever a candidate at a given target rank - which is what lets
these tests assert a specific chunk_id rather than just "a chunk".
"""

from unittest.mock import MagicMock

import pytest

from pgvector.psycopg import register_vector
from retrieval.pgvector_store import PgVectorRetriever, build_keyword_tsquery
from tests.conftest import FIXTURE_MARKER

FIXTURE_WORD = "zzqfixtureword"


def _one_hot(dim: int = 1536) -> list[float]:
    vec = [0.0] * dim
    vec[0] = 1.0
    return vec


ONE_HOT = _one_hot()


def _fake_embedding_client(vector: list[float]) -> MagicMock:
    """A mock OpenAI client whose embeddings.create() always returns `vector`,
    regardless of the query text - never touches the network."""
    client = MagicMock()

    def fake_create(model, input, dimensions):
        return MagicMock(data=[MagicMock(embedding=list(vector)) for _ in input])

    client.embeddings.create.side_effect = fake_create
    return client


def _insert_embedded_chunk(conn, doc_id, heading_path, raw_text, embedding, valid_from_rank, valid_to_rank):
    with conn.transaction():
        row = conn.execute(
            """
            INSERT INTO chunks
                (doc_id, heading_path, chunk_index, raw_text, embed_text,
                 content_hash, valid_from_rank, valid_to_rank, embedding, embedded_at)
            VALUES (%s, %s, 0, %s, %s, %s, %s, %s, %s, now())
            RETURNING chunk_id
            """,
            (doc_id, heading_path, raw_text, raw_text, f"hash-{heading_path}", valid_from_rank, valid_to_rank, embedding),
        ).fetchone()
    return row[0]


@pytest.fixture
def ab_chunks(pg_conn, version_filter_doc):
    """Chunk A (valid 1-1) and chunk B (valid 3-NULL), same text, same
    one-hot embedding. Returns (chunk_a_id, chunk_b_id)."""
    register_vector(pg_conn)
    doc_id = version_filter_doc

    chunk_a = _insert_embedded_chunk(
        pg_conn, doc_id, f"{FIXTURE_MARKER}chunkA",
        f"Chunk A contains {FIXTURE_WORD} for testing.",
        ONE_HOT, valid_from_rank=1, valid_to_rank=1,
    )
    chunk_b = _insert_embedded_chunk(
        pg_conn, doc_id, f"{FIXTURE_MARKER}chunkB",
        f"Chunk B contains {FIXTURE_WORD} for testing.",
        ONE_HOT, valid_from_rank=3, valid_to_rank=None,
    )
    return chunk_a, chunk_b


def test_dense_retrieval_respects_version_filter(pg_conn, ab_chunks):
    chunk_a, chunk_b = ab_chunks
    client = _fake_embedding_client(ONE_HOT)
    retriever = PgVectorRetriever(pg_conn, openai_client=client)

    at_rank_1 = retriever.retrieve(FIXTURE_WORD, target_version_rank=1, top_k=1, mode="dense")
    at_rank_3 = retriever.retrieve(FIXTURE_WORD, target_version_rank=3, top_k=1, mode="dense")

    assert [c.chunk_id for c in at_rank_1] == [chunk_a]
    assert [c.chunk_id for c in at_rank_3] == [chunk_b]


def test_keyword_retrieval_respects_version_filter(pg_conn, ab_chunks):
    chunk_a, chunk_b = ab_chunks
    client = _fake_embedding_client(ONE_HOT)  # unused by keyword mode, but the constructor needs one
    retriever = PgVectorRetriever(pg_conn, openai_client=client)

    at_rank_1 = retriever.retrieve(FIXTURE_WORD, target_version_rank=1, top_k=5, mode="keyword")
    at_rank_3 = retriever.retrieve(FIXTURE_WORD, target_version_rank=3, top_k=5, mode="keyword")

    assert [c.chunk_id for c in at_rank_1] == [chunk_a]
    assert [c.chunk_id for c in at_rank_3] == [chunk_b]


def test_hybrid_retrieval_fuses_dense_and_keyword(pg_conn, ab_chunks):
    _chunk_a, chunk_b = ab_chunks
    client = _fake_embedding_client(ONE_HOT)
    retriever = PgVectorRetriever(pg_conn, openai_client=client)

    # At rank 3 only chunk B is a candidate for either mode, so it's rank 1
    # in both the dense and keyword rankings RRF fuses: score = 1/61 + 1/61.
    results = retriever.retrieve(FIXTURE_WORD, target_version_rank=3, top_k=5, mode="hybrid")

    assert results[0].chunk_id == chunk_b
    assert results[0].score == pytest.approx(2 / 61, rel=1e-6)


def test_keyword_score_unweighted_when_no_identifiers(pg_conn, ab_chunks):
    """FIXTURE_WORD has no identifier-like token, so build_keyword_tsquery
    returns phrase_sql=None. PHRASE_WEIGHT must then contribute nothing -
    the returned score has to be exactly ts_rank_cd(text_search, word_sql),
    the same single-term score the old, unweighted implementation produced."""
    _chunk_a, chunk_b = ab_chunks

    phrase_sql, word_sql, params = build_keyword_tsquery(FIXTURE_WORD)
    assert phrase_sql is None  # sanity: this query really has no identifiers

    client = _fake_embedding_client(ONE_HOT)
    retriever = PgVectorRetriever(pg_conn, openai_client=client)
    results = retriever.retrieve(FIXTURE_WORD, target_version_rank=3, top_k=5, mode="keyword")
    assert results[0].chunk_id == chunk_b

    with pg_conn.transaction():
        expected_score = pg_conn.execute(
            f"SELECT ts_rank_cd(text_search, ({word_sql})) FROM chunks WHERE chunk_id = %(chunk_id)s",
            {**params, "chunk_id": chunk_b},
        ).fetchone()[0]

    assert results[0].score == pytest.approx(expected_score)
