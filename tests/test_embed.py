"""Tests for ingestion/gold/embed.py. Mocks the OpenAI client - zero
network calls. Postgres-backed; skips cleanly if the DB isn't reachable
(via the shared pg_conn fixture in conftest.py).
"""

from unittest.mock import MagicMock

from pgvector.psycopg import register_vector

from ingestion.gold import embed
from tests.conftest import FIXTURE_MARKER


def _fake_client() -> MagicMock:
    """A mock OpenAI client whose embeddings.create() never touches the network."""
    client = MagicMock()

    def fake_create(model, input, dimensions):
        return MagicMock(data=[MagicMock(embedding=[0.0] * dimensions) for _ in input])

    client.embeddings.create.side_effect = fake_create
    return client


def _insert_chunks(conn, doc_id, version_rank, indices, embedded: bool) -> None:
    """Insert one chunk per index, with or without a pre-existing embedding."""
    embedded_at_expr = "now()" if embedded else "NULL"
    with conn.transaction():
        for i in indices:
            embedding = [0.5] * 1536 if embedded else None
            conn.execute(
                f"""
                INSERT INTO chunks
                    (doc_id, heading_path, chunk_index, raw_text, embed_text,
                     content_hash, valid_from_rank, embedding, embedded_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, {embedded_at_expr})
                """,
                (
                    doc_id,
                    f"{FIXTURE_MARKER}chunk-{i}",
                    i,
                    f"raw text {i}",
                    f"embed text {i}",
                    f"hash-{i}",
                    version_rank,
                    embedding,
                ),
            )


def test_embed_all_only_touches_null_rows(pg_conn, embed_fixture_doc):
    register_vector(pg_conn)
    doc_id, version_rank = embed_fixture_doc

    _insert_chunks(pg_conn, doc_id, version_rank, range(0, 2), embedded=False)
    _insert_chunks(pg_conn, doc_id, version_rank, range(2, 3), embedded=True)

    client = _fake_client()
    per_version = embed.embed_all(pg_conn, client)

    assert client.embeddings.create.call_count == 1
    sent_texts = client.embeddings.create.call_args.kwargs["input"]
    assert sorted(sent_texts) == ["embed text 0", "embed text 1"]
    assert per_version == {version_rank: 2}

    with pg_conn.transaction():
        row = pg_conn.execute(
            "SELECT embedding FROM chunks WHERE doc_id = %s AND heading_path = %s",
            (doc_id, f"{FIXTURE_MARKER}chunk-2"),
        ).fetchone()
    assert row[0].to_list()[:3] == [0.5, 0.5, 0.5]


def test_embed_all_rerun_embeds_nothing(pg_conn, embed_fixture_doc):
    register_vector(pg_conn)
    doc_id, version_rank = embed_fixture_doc

    _insert_chunks(pg_conn, doc_id, version_rank, range(0, 3), embedded=False)

    client = _fake_client()
    first = embed.embed_all(pg_conn, client)
    assert first == {version_rank: 3}
    assert client.embeddings.create.call_count == 1

    client.embeddings.create.reset_mock()
    second = embed.embed_all(pg_conn, client)

    assert second == {}
    client.embeddings.create.assert_not_called()


def test_embed_all_splits_into_batches(pg_conn, embed_fixture_doc):
    register_vector(pg_conn)
    doc_id, version_rank = embed_fixture_doc

    n = 2 * embed.BATCH_SIZE + 50  # 250 rows when BATCH_SIZE is 100
    _insert_chunks(pg_conn, doc_id, version_rank, range(0, n), embedded=False)

    client = _fake_client()
    per_version = embed.embed_all(pg_conn, client)

    assert client.embeddings.create.call_count == 3
    batch_sizes = [len(call.kwargs["input"]) for call in client.embeddings.create.call_args_list]
    assert batch_sizes == [embed.BATCH_SIZE, embed.BATCH_SIZE, 50]
    assert per_version == {version_rank: n}
