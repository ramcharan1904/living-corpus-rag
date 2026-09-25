"""Silver ingestion: chunk each document's Bronze copy and write
version-aware chunk rows to Postgres.

For each document at a version, chunk it, hash each chunk, and compare
against the currently open row (valid_to_rank IS NULL) for the same
(doc_id, heading_path):

    no open row              -> new:       insert, valid_from_rank = this rank
    open row, same hash      -> unchanged: leave the open interval as is
    open row, different hash -> changed:   close the old row at this rank - 1, insert new
    open row not seen again  -> retired:   close it at this rank - 1 (checked
                                            once per document, after all its
                                            chunks are processed)

Usage:
    python -m ingestion.silver.write v2.4.0
"""

import argparse
from pathlib import Path

import psycopg

from config import REPO_ROOT, settings
from ingestion.silver.chunking import Chunk, chunk_markdown
from ingestion.silver.hash import content_hash


def _doc_title(rel_path: str) -> str:
    return Path(rel_path).stem.replace("_", " ").title()


def get_version_rank(conn: psycopg.Connection, version_tag: str) -> int:
    with conn.transaction():
        row = conn.execute(
            "SELECT version_rank FROM versions WHERE version_tag = %s",
            (version_tag,),
        ).fetchone()
    if row is None:
        raise SystemExit(f"no row in versions for version_tag={version_tag!r}")
    return row[0]


def _iter_doc_versions(conn: psycopg.Connection, version_rank: int) -> list[tuple[int, str, str]]:
    with conn.transaction():
        rows = conn.execute(
            """
            SELECT dv.doc_id, d.rel_path, dv.raw_path
            FROM doc_versions dv
            JOIN documents d ON d.doc_id = dv.doc_id
            WHERE dv.version_rank = %s
            ORDER BY d.rel_path
            """,
            (version_rank,),
        ).fetchall()
    return rows


def _open_chunk(conn: psycopg.Connection, doc_id: int, heading_path: str) -> tuple[int, str] | None:
    with conn.transaction():
        row = conn.execute(
            """
            SELECT chunk_id, content_hash FROM chunks
            WHERE doc_id = %s AND heading_path = %s AND valid_to_rank IS NULL
            """,
            (doc_id, heading_path),
        ).fetchone()
    return row


def _close_chunk(conn: psycopg.Connection, chunk_id: int, closed_at_rank: int) -> None:
    with conn.transaction():
        conn.execute(
            "UPDATE chunks SET valid_to_rank = %s WHERE chunk_id = %s",
            (closed_at_rank, chunk_id),
        )


def _insert_chunk(conn: psycopg.Connection, doc_id: int, chunk: Chunk, chunk_hash: str, version_rank: int) -> None:
    with conn.transaction():
        conn.execute(
            """
            INSERT INTO chunks
                (doc_id, heading_path, chunk_index, raw_text, embed_text, content_hash, valid_from_rank, valid_to_rank)
            VALUES (%s, %s, %s, %s, %s, %s, %s, NULL)
            """,
            (doc_id, chunk.heading_path, chunk.chunk_index, chunk.raw_text, chunk.embed_text, chunk_hash, version_rank),
        )


def _open_heading_paths(conn: psycopg.Connection, doc_id: int) -> dict[str, int]:
    """Currently open chunk_id, keyed by heading_path, for this document."""
    with conn.transaction():
        rows = conn.execute(
            "SELECT heading_path, chunk_id FROM chunks WHERE doc_id = %s AND valid_to_rank IS NULL",
            (doc_id,),
        ).fetchall()
    return {heading_path: chunk_id for heading_path, chunk_id in rows}


def process_document(conn: psycopg.Connection, doc_id: int, rel_path: str, raw_path: str, version_rank: int) -> dict[str, int]:
    """Chunk one document's Bronze copy and reconcile it against open chunk rows.

    Returns counts for this document: new, changed, unchanged, retired.
    """
    text = (REPO_ROOT / raw_path).read_text(encoding="utf-8")
    chunks = chunk_markdown(text, _doc_title(rel_path))

    counts = {"new": 0, "changed": 0, "unchanged": 0, "retired": 0}
    seen_paths: set[str] = set()

    for chunk in chunks:
        seen_paths.add(chunk.heading_path)
        chash = content_hash(chunk.raw_text)

        existing = _open_chunk(conn, doc_id, chunk.heading_path)

        if existing is None:
            _insert_chunk(conn, doc_id, chunk, chash, version_rank)
            counts["new"] += 1
        elif existing[1] == chash:
            counts["unchanged"] += 1
        else:
            _close_chunk(conn, existing[0], version_rank - 1)
            _insert_chunk(conn, doc_id, chunk, chash, version_rank)
            counts["changed"] += 1

    # Retire: chunk_ids still open for this doc whose heading_path this
    # version's chunks never touched at all. Rows we just inserted or left
    # unchanged above are excluded automatically, since their heading_path
    # is in seen_paths.
    still_open = _open_heading_paths(conn, doc_id)
    for heading_path, chunk_id in still_open.items():
        if heading_path not in seen_paths:
            _close_chunk(conn, chunk_id, version_rank - 1)
            counts["retired"] += 1

    return counts


def _start_run(conn: psycopg.Connection, version_rank: int) -> int:
    with conn.transaction():
        row = conn.execute(
            """
            INSERT INTO ingest_runs (version_rank, docs_seen, chunks_new, chunks_changed, chunks_unchanged, chunks_retired)
            VALUES (%s, 0, 0, 0, 0, 0)
            RETURNING run_id
            """,
            (version_rank,),
        ).fetchone()
    return row[0]


def _update_run(conn: psycopg.Connection, run_id: int, totals: dict[str, int]) -> None:
    with conn.transaction():
        conn.execute(
            """
            UPDATE ingest_runs
            SET docs_seen = %s, chunks_new = %s, chunks_changed = %s, chunks_unchanged = %s, chunks_retired = %s
            WHERE run_id = %s
            """,
            (totals["docs_seen"], totals["new"], totals["changed"], totals["unchanged"], totals["retired"], run_id),
        )


def _finish_run(conn: psycopg.Connection, run_id: int) -> None:
    with conn.transaction():
        conn.execute("UPDATE ingest_runs SET finished_at = now() WHERE run_id = %s", (run_id,))


def write_version(conn: psycopg.Connection, version_tag: str) -> dict[str, int]:
    version_rank = get_version_rank(conn, version_tag)
    doc_rows = _iter_doc_versions(conn, version_rank)

    if not doc_rows:
        raise SystemExit(f"no doc_versions rows for version_tag={version_tag!r} - run Bronze fetch first")

    run_id = _start_run(conn, version_rank)
    totals = {"docs_seen": 0, "new": 0, "changed": 0, "unchanged": 0, "retired": 0}

    for doc_id, rel_path, raw_path in doc_rows:
        with conn.transaction():
            counts = process_document(conn, doc_id, rel_path, raw_path, version_rank)

        totals["docs_seen"] += 1
        totals["new"] += counts["new"]
        totals["changed"] += counts["changed"]
        totals["unchanged"] += counts["unchanged"]
        totals["retired"] += counts["retired"]

        # Written after every document, not just at the end, so ingest_runs
        # reflects real progress even if a later document fails.
        _update_run(conn, run_id, totals)

    _finish_run(conn, run_id)

    total_chunks = totals["new"] + totals["changed"] + totals["unchanged"]
    pct_unchanged = (totals["unchanged"] / total_chunks * 100) if total_chunks else 0.0

    print(
        f"{version_tag}: {totals['docs_seen']} docs, "
        f"{totals['new']} new, {totals['changed']} changed, "
        f"{totals['unchanged']} unchanged, {totals['retired']} retired "
        f"({pct_unchanged:.1f}% unchanged)"
    )

    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version_tag", help="version_tag from the versions table, e.g. v2.4.0")
    args = parser.parse_args()

    with psycopg.connect(settings.postgres_dsn) as conn:
        write_version(conn, args.version_tag)


if __name__ == "__main__":
    main()
