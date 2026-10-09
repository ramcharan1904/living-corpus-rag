"""Gold ingestion: embed each chunk's embed_text with OpenAI, skipping
chunks that already have an embedding.

A chunk identical across several versions is one row - Silver's SCD2
write keeps it open rather than duplicating it - so it gets embedded
exactly once no matter how many versions it is valid in. This is where
the incremental-ingestion savings from Silver become real embedding-cost
savings.

Usage:
    python -m ingestion.gold.embed
"""

import time

import psycopg
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from pgvector.psycopg import register_vector

from config import settings

MODEL = "text-embedding-3-small"
DIMENSIONS = 1536
BATCH_SIZE = 100
MAX_ATTEMPTS = 5

# APIConnectionError/APITimeoutError/InternalServerError are always worth
# retrying. RateLimitError is split out in embed_texts() below, since it
# covers both a transient "slow down" and a permanent "no quota left" -
# see the comment there. AuthenticationError/BadRequestError are never
# retried at all: the same request fails again every time.


def get_client() -> OpenAI:
    key = settings.openai_api_key.get_secret_value()
    if not key:
        raise SystemExit("OPENAI_API_KEY is empty - set it in .env")
    return OpenAI(api_key=key)


def fetch_unembedded(conn: psycopg.Connection) -> list[tuple[int, str, int]]:
    """(chunk_id, embed_text, valid_from_rank) for every chunk needing an embedding."""
    with conn.transaction():
        rows = conn.execute(
            "SELECT chunk_id, embed_text, valid_from_rank FROM chunks WHERE embedding IS NULL ORDER BY chunk_id"
        ).fetchall()
    return rows


def _retry_or_raise(attempt: int, exc: Exception) -> None:
    if attempt == MAX_ATTEMPTS:
        raise exc
    wait = 2**attempt
    print(f"  retry {attempt}/{MAX_ATTEMPTS} after {type(exc).__name__}: sleeping {wait}s")
    time.sleep(wait)


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.embeddings.create(model=MODEL, input=texts, dimensions=DIMENSIONS)
            return [item.embedding for item in response.data]
        except RateLimitError as exc:
            # OpenAI uses HTTP 429 for both "slow down" (transient) and "no
            # quota/credits left" (permanent until billing is fixed) - both
            # raise RateLimitError, but only type == "insufficient_quota" is
            # the latter. Retrying that just burns backoff time for nothing.
            if exc.type == "insufficient_quota":
                raise SystemExit(
                    "OpenAI account has no quota/credits remaining. Add credits at "
                    "https://platform.openai.com/settings/organization/billing/ and "
                    "re-run - already-embedded chunks are skipped automatically."
                ) from exc
            _retry_or_raise(attempt, exc)
        except (APIConnectionError, APITimeoutError, InternalServerError) as exc:
            _retry_or_raise(attempt, exc)
    raise RuntimeError("unreachable")


def save_batch(conn: psycopg.Connection, chunk_ids: list[int], embeddings: list[list[float]]) -> None:
    with conn.transaction():
        for chunk_id, embedding in zip(chunk_ids, embeddings):
            conn.execute(
                "UPDATE chunks SET embedding = %s, embedded_at = now() WHERE chunk_id = %s",
                (embedding, chunk_id),
            )


def bump_embed_calls(conn: psycopg.Connection, version_rank: int, n: int) -> None:
    """Add n to embed_calls on this version's most recent ingest_runs row.

    Additive, not a plain set: a crash-and-resume run embeds the same
    version's chunks across two separate invocations of this script, and
    each should add its share rather than overwrite the other's count.
    """
    with conn.transaction():
        conn.execute(
            """
            UPDATE ingest_runs
            SET embed_calls = coalesce(embed_calls, 0) + %s
            WHERE run_id = (
                SELECT run_id FROM ingest_runs
                WHERE version_rank = %s
                ORDER BY run_id DESC
                LIMIT 1
            )
            """,
            (n, version_rank),
        )


def embed_all(conn: psycopg.Connection, client: OpenAI) -> dict[int, int]:
    """Embed every chunk with embedding IS NULL. Returns {version_rank: chunks embedded}."""
    rows = fetch_unembedded(conn)
    per_version: dict[int, int] = {}

    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        chunk_ids = [r[0] for r in batch]
        texts = [r[1] for r in batch]
        ranks = [r[2] for r in batch]

        embeddings = embed_texts(client, texts)
        save_batch(conn, chunk_ids, embeddings)

        for rank in ranks:
            per_version[rank] = per_version.get(rank, 0) + 1

        print(f"  embedded batch {i // BATCH_SIZE + 1} ({len(batch)} chunks)")

    for rank, count in per_version.items():
        bump_embed_calls(conn, rank, count)

    return per_version


def main() -> None:
    client = get_client()
    with psycopg.connect(settings.postgres_dsn) as conn:
        register_vector(conn)
        per_version = embed_all(conn, client)

    total = sum(per_version.values())
    print(f"\nembedded {total} chunks total")
    for rank in sorted(per_version):
        print(f"  rank {rank}: {per_version[rank]} embedded")
    if not per_version:
        print("  nothing to embed")


if __name__ == "__main__":
    main()
