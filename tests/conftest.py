import shutil

import psycopg
import pytest

from config import REPO_ROOT, settings

# A version_rank real seed data will never use, and a filename marker real
# corpus docs will never contain, so fixture rows/files can never collide
# with real ingested data and cleanup can find everything it created.
TEST_VERSION_RANK = 999
TEST_VERSION_TAG = "test-fixture"
FIXTURE_MARKER = "__test_fixture__"


def _cleanup(conn: psycopg.Connection) -> None:
    # If a prior statement on this connection left an implicit transaction
    # open (any bare execute() outside a `with conn.transaction():` block
    # does this), roll it back first so the block below is a real top-level
    # transaction and not a nested savepoint - a savepoint's release doesn't
    # commit the outer transaction, so its effects can be lost on conn.close().
    if conn.info.transaction_status != psycopg.pq.TransactionStatus.IDLE:
        conn.rollback()

    with conn.transaction():
        # Deleting documents cascades to their doc_versions rows (ON DELETE
        # CASCADE on doc_versions.doc_id), so this alone clears both tables.
        conn.execute(
            "DELETE FROM documents WHERE library = %s AND rel_path LIKE %s",
            ("pydantic", f"%{FIXTURE_MARKER}%"),
        )
        conn.execute(
            "DELETE FROM versions WHERE version_rank = %s",
            (TEST_VERSION_RANK,),
        )
    bronze_dir = REPO_ROOT / settings.bronze_path / TEST_VERSION_TAG
    shutil.rmtree(bronze_dir, ignore_errors=True)


@pytest.fixture
def pg_conn():
    try:
        conn = psycopg.connect(settings.postgres_dsn, connect_timeout=3)
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres not reachable at {settings.postgres_host}:{settings.postgres_port}: {exc}")
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def test_version(pg_conn, tmp_path):
    """Seed a throwaway `versions` row pointing at a fake corpus under tmp_path.

    Yields the version_tag. Cleans up its versions/documents/doc_versions
    rows and its bronze/ output dir, both before (in case a previous run
    crashed mid-test) and after.
    """
    _cleanup(pg_conn)

    corpus_root = tmp_path / "corpus"
    (corpus_root / "docs" / "concepts").mkdir(parents=True)
    (corpus_root / "docs" / "concepts" / f"{FIXTURE_MARKER}widget.md").write_bytes(
        b"# Widget\n\nfixture content\n"
    )
    (corpus_root / "docs" / f"{FIXTURE_MARKER}top.md").write_bytes(b"# Top-level fixture\n")

    with pg_conn.transaction():
        pg_conn.execute(
            """
            INSERT INTO versions (version_rank, version_tag, commit_sha, corpus_path)
            VALUES (%s, %s, %s, %s)
            """,
            (TEST_VERSION_RANK, TEST_VERSION_TAG, "deadbeef", str(corpus_root)),
        )

    try:
        yield TEST_VERSION_TAG
    finally:
        _cleanup(pg_conn)
