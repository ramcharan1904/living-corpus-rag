"""Verify golden questions' `sources` heading paths actually exist in the
database, at the version they claim to.

For every question x version with a sources entry (expanding the `all`
shorthand to all three ranks), confirm at least one chunk valid at that
rank has a heading_path that equals the source, or starts with it followed
by ' > ' or ' (part '. Hand-written source paths have been wrong before
(q003) - a wrong path silently turns correct retrieval into a reported
miss in Step 6's recall eval, so this needs to run clean first.

Never modifies questions.yaml - only reports mismatches, with the closest
real heading paths at that rank, for a human to fix.

Usage:
    python -m eval.check_sources
"""

import difflib

import psycopg
import yaml

from config import REPO_ROOT, settings
from eval.matching import is_hit
from retrieval.pgvector_store import version_filter_sql

QUESTIONS_PATH = REPO_ROOT / "eval" / "golden" / "questions.yaml"
VERSION_TAGS = ("v2.4.0", "v2.8.2", "v2.13.5")


def load_questions() -> list[dict]:
    with QUESTIONS_PATH.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["questions"]


def expand_sources(sources: dict) -> dict[str, str]:
    """Expand the `all` shorthand (same convention as `answers.all`) to
    an explicit entry for every version tag."""
    if "all" in sources:
        return {tag: sources["all"] for tag in VERSION_TAGS}
    return dict(sources)


def get_version_ranks(conn: psycopg.Connection) -> dict[str, int]:
    with conn.transaction():
        rows = conn.execute("SELECT version_tag, version_rank FROM versions").fetchall()
    return dict(rows)


def heading_paths_at_rank(conn: psycopg.Connection, rank: int) -> list[str]:
    version_sql = version_filter_sql()
    with conn.transaction():
        rows = conn.execute(
            f"SELECT DISTINCT heading_path FROM chunks c WHERE {version_sql}",
            {"target_rank": rank},
        ).fetchall()
    return [r[0] for r in rows]


def check_all(conn: psycopg.Connection) -> list[dict]:
    version_ranks = get_version_ranks(conn)
    questions = load_questions()

    # Only 3 distinct ranks exist - fetch each rank's heading paths once.
    paths_by_rank = {rank: heading_paths_at_rank(conn, rank) for rank in set(version_ranks.values())}

    mismatches = []
    for q in questions:
        sources = q.get("sources")
        if not sources:
            continue
        for tag, source in expand_sources(sources).items():
            rank = version_ranks[tag]
            candidates = paths_by_rank[rank]
            if is_hit(candidates, source):
                continue
            mismatches.append(
                {
                    "id": q["id"],
                    "version_tag": tag,
                    "source": source,
                    "closest": difflib.get_close_matches(source, candidates, n=3, cutoff=0.0),
                }
            )
    return mismatches


def main() -> None:
    with psycopg.connect(settings.postgres_dsn) as conn:
        mismatches = check_all(conn)

    if not mismatches:
        print("All golden sources matched a real heading path. Nothing to fix.")
        return

    print(f"{len(mismatches)} mismatch(es):\n")
    for m in mismatches:
        print(f"{m['id']} @ {m['version_tag']}: source = {m['source']!r}")
        if m["closest"]:
            print("  closest real heading paths at that rank:")
            for c in m["closest"]:
                print(f"    - {c!r}")
        else:
            print("  (no chunks at all exist at that rank - nothing to compare against)")
        print()


if __name__ == "__main__":
    main()
