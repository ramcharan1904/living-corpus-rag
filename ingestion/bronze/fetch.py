"""Bronze ingestion: pull raw docs for one corpus version into bronze/<version_tag>/
and record the per-file linkage (documents, doc_versions) in Postgres.

Usage:
    python -m ingestion.bronze.fetch v2.13.5
"""

import argparse
import hashlib
from pathlib import Path

import psycopg

from config import REPO_ROOT, settings

LIBRARY = "pydantic"

# Recursively walked for .md files.
SCOPED_SUBDIRS = ("docs/concepts", "docs/errors", "docs/integrations")

# Only direct children are taken here (not walked recursively), to avoid
# pulling in docs/api, docs/examples, etc.
TOP_LEVEL_DIR = "docs"



def iter_scoped_md_files(corpus_root: Path):
    """Yield Markdown files in scope, de-duplicated, in a stable order."""
    seen: set[Path] = set()

    for rel_dir in SCOPED_SUBDIRS:
        base = corpus_root / rel_dir
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.md")):
            if path not in seen:
                seen.add(path)
                yield path

    docs_dir = corpus_root / "docs"
    if docs_dir.is_dir():
        for path in sorted(docs_dir.glob("*.md")):
            if path.name in settings.excluded_top_level:
                continue
            if path not in seen:
                seen.add(path)
                yield path

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get_version_row(conn: psycopg.Connection, version_tag: str) -> tuple[int, Path]:
    row = conn.execute(
        "SELECT version_rank, corpus_path FROM versions WHERE version_tag = %s",
        (version_tag,),
    ).fetchone()
    if row is None:
        raise SystemExit(f"no row in versions for version_tag={version_tag!r}")
    version_rank, corpus_path = row
    return version_rank, (REPO_ROOT / corpus_path).resolve()


def upsert_document(conn: psycopg.Connection, rel_path: str) -> int:
    # ON CONFLICT ... DO UPDATE (a no-op field reset) so this always
    # returns doc_id, whether the row was just inserted or already existed.
    row = conn.execute(
        """
        INSERT INTO documents (library, rel_path)
        VALUES (%s, %s)
        ON CONFLICT (library, rel_path) DO UPDATE SET rel_path = EXCLUDED.rel_path
        RETURNING doc_id
        """,
        (LIBRARY, rel_path),
    ).fetchone()
    return row[0]


def doc_version_exists(conn: psycopg.Connection, doc_id: int, version_rank: int) -> bool:
    row = conn.execute(
        "SELECT 1 FROM doc_versions WHERE doc_id = %s AND version_rank = %s",
        (doc_id, version_rank),
    ).fetchone()
    return row is not None


def fetch_version(conn: psycopg.Connection, version_tag: str) -> None:
    version_rank, corpus_root = get_version_row(conn, version_tag)
    if not corpus_root.is_dir():
        raise SystemExit(f"corpus_path does not exist: {corpus_root}")

    bronze_root = (REPO_ROOT / settings.bronze_path / version_tag).resolve()

    n_new = 0
    n_skipped = 0

    for src_path in iter_scoped_md_files(corpus_root):
        rel_path = src_path.relative_to(corpus_root).as_posix()

        with conn.transaction():
            doc_id = upsert_document(conn, rel_path)

            if doc_version_exists(conn, doc_id, version_rank):
                n_skipped += 1
                continue

            # Hash and copy the raw bytes (not decoded text), so the hash
            # stays reproducible regardless of platform/encoding handling.
            raw_bytes = src_path.read_bytes()
            content_hash = sha256_bytes(raw_bytes)

            dest_path = bronze_root / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(raw_bytes)

            conn.execute(
                """
                INSERT INTO doc_versions (doc_id, version_rank, content_hash, raw_path)
                VALUES (%s, %s, %s, %s)
                """,
                (doc_id, version_rank, content_hash, str(dest_path.relative_to(REPO_ROOT))),
            )
            n_new += 1

    print(f"{version_tag}: {n_new} new, {n_skipped} skipped (already ingested), {n_new + n_skipped} total in scope")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version_tag", help="version_tag from the versions table, e.g. v2.13.5")
    args = parser.parse_args()

    with psycopg.connect(settings.postgres_dsn) as conn:
        fetch_version(conn, args.version_tag)


if __name__ == "__main__":
    main()
