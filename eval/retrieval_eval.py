"""Recall@5 evaluation: for every golden question x version with a source,
compare version-aware retrieval (target_version_rank = that version) against
baseline (target_version_rank = 3, the newest) across all three modes.

Query embeddings are cached by (model, input, dimensions) - in practice, by
question text, since dimensions/model never vary within a run. Dense's call
and hybrid's internal dense call, and the aware/baseline variants of either,
all embed the identical question text, so each question costs at most one
real embedding call for the whole run, not one per mode x variant.

Usage:
    python -m eval.retrieval_eval
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import psycopg
import yaml
from openai import OpenAI

from config import REPO_ROOT, settings
from eval.matching import is_hit
from retrieval.base import RetrievedChunk
from retrieval.pgvector_store import PgVectorRetriever

QUESTIONS_PATH = REPO_ROOT / "eval" / "golden" / "questions.yaml"
RESULTS_DIR = REPO_ROOT / "eval" / "results"
VERSION_TAGS = ("v2.4.0", "v2.8.2", "v2.13.5")
BASELINE_RANK = 3
TOP_K = 5
MODES = ("dense", "keyword", "hybrid")
CATEGORIES = ("version_sensitive", "new_feature", "removed", "version_stable")

# load_questions/expand_sources are duplicated from check_sources.py rather
# than imported - see the review note on whether to extract a shared
# eval/golden.py instead.


def load_questions() -> list[dict]:
    with QUESTIONS_PATH.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["questions"]


def expand_sources(sources: dict) -> dict[str, str]:
    if "all" in sources:
        return {tag: sources["all"] for tag in VERSION_TAGS}
    return dict(sources)


class _CachingEmbeddings:
    """Wraps a real embeddings client so the same (model, input,
    dimensions) request is never sent twice in one run."""

    def __init__(self, real_embeddings):
        self._real = real_embeddings
        self._cache: dict[tuple, object] = {}
        self.call_count = 0

    def create(self, model, input, dimensions):
        key = (model, tuple(input), dimensions)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        self.call_count += 1
        response = self._real.create(model=model, input=input, dimensions=dimensions)
        self._cache[key] = response
        return response


class CachingOpenAIClient:
    def __init__(self, real_client: OpenAI):
        self.embeddings = _CachingEmbeddings(real_client.embeddings)


def get_version_ranks(conn: psycopg.Connection) -> dict[str, int]:
    with conn.transaction():
        rows = conn.execute("SELECT version_tag, version_rank FROM versions").fetchall()
    return dict(rows)


def build_eval_rows(questions: list[dict], version_ranks: dict[str, int]) -> list[dict]:
    """One row per question x version that has a sources entry (`all`
    expanded). Versions with no entry are "Not covered" and skipped - they
    belong to the judge stage, not recall."""
    rows = []
    for q in questions:
        sources = q.get("sources")
        if not sources:
            continue
        for tag, source in expand_sources(sources).items():
            rows.append(
                {
                    "id": q["id"],
                    "category": q["category"],
                    "question": q["question"],
                    "version_tag": tag,
                    "rank": version_ranks[tag],
                    "source": source,
                }
            )
    return rows


def _chunk_summary(chunks: list[RetrievedChunk]) -> list[dict]:
    return [
        {
            "heading_path": c.heading_path,
            "valid_from_rank": c.valid_from_rank,
            "valid_to_rank": c.valid_to_rank,
            "score": c.score,
        }
        for c in chunks
    ]


def run_eval(retriever: PgVectorRetriever, rows: list[dict]) -> list[dict]:
    results = []
    for row in rows:
        for mode in MODES:
            aware = retriever.retrieve(row["question"], target_version_rank=row["rank"], top_k=TOP_K, mode=mode)
            base = retriever.retrieve(row["question"], target_version_rank=BASELINE_RANK, top_k=TOP_K, mode=mode)

            for variant, hits in (("aware", aware), ("base", base)):
                paths = [c.heading_path for c in hits]
                results.append(
                    {
                        "id": row["id"],
                        "category": row["category"],
                        "version_tag": row["version_tag"],
                        "mode": mode,
                        "variant": variant,
                        "source": row["source"],
                        "hit": is_hit(paths, row["source"]),
                        "top5": _chunk_summary(hits),
                    }
                )
    return results


def _cell(results: list[dict], mode: str, variant: str) -> str:
    subset = [r for r in results if r["mode"] == mode and r["variant"] == variant]
    hits = sum(r["hit"] for r in subset)
    return f"{hits}/{len(subset)}"


def print_table(results: list[dict]) -> None:
    columns = [(mode, variant) for mode in MODES for variant in ("aware", "base")]
    header = ["category"] + [f"{m}_{v}" for m, v in columns]
    widths = [max(len(h), 14) for h in header]

    def row_line(label: str, cells: list[str]) -> str:
        return " | ".join(f"{c:>{w}}" for c, w in zip([label] + cells, widths))

    print(row_line("category", [f"{m}_{v}" for m, v in columns]))
    for cat in CATEGORIES:
        cat_results = [r for r in results if r["category"] == cat]
        print(row_line(cat, [_cell(cat_results, m, v) for m, v in columns]))
    print(row_line("TOTAL", [_cell(results, m, v) for m, v in columns]))


def save_results(results: list[dict]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS_DIR / f"retrieval_{timestamp}.json"
    path.write_text(json.dumps(results, indent=2))
    return path


def version_correct_hit(result: dict, asked_rank: int) -> bool:
    """Stricter than `result["hit"]`: also requires the specific chunk
    that matched is_hit to be valid at asked_rank, not merely at whichever
    rank retrieve() happened to be queried with.

    This matters only for "base" rows. retrieve() always filters inside
    the SQL by the rank it was actually asked for - row["rank"] for
    "aware", 3 for "base" - so the chunk returned is already guaranteed
    valid at THAT rank. For "aware" that rank IS asked_rank, so this can
    never change an aware result. For "base", asked_rank is the question's
    true version, not 3 - so a same-named heading whose CURRENT content
    (valid at rank 3) differs from what existed at the true asked rank
    would satisfy is_hit on text alone while answering the wrong version.
    """
    for chunk in result["top5"]:
        if not is_hit([chunk["heading_path"]], result["source"]):
            continue
        if chunk["valid_from_rank"] <= asked_rank and (
            chunk["valid_to_rank"] is None or chunk["valid_to_rank"] >= asked_rank
        ):
            return True
    return False


def rescore(results: list[dict], version_ranks: dict[str, int]) -> list[dict]:
    """Recompute every result's hit under the version-correct rule, reading
    only each result's saved top5 - no new retrieval calls."""
    rescored = []
    for r in results:
        asked_rank = version_ranks[r["version_tag"]]
        rescored.append({**r, "hit": version_correct_hit(r, asked_rank)})
    return rescored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rescore",
        metavar="RESULTS_JSON",
        help="Re-score an existing results file under the stricter "
        "version-correct rule instead of running retrieval again - no "
        "API calls, no new retrieve() calls.",
    )
    args = parser.parse_args()

    with psycopg.connect(settings.postgres_dsn) as conn:
        version_ranks = get_version_ranks(conn)

        if args.rescore:
            results = json.loads(Path(args.rescore).read_text())
            print("=== Pass 1: is_hit (heading path match only) ===")
            print_table(results)
            print("\n=== Pass 2: version-correct (is_hit AND chunk valid at the asked rank) ===")
            print_table(rescore(results, version_ranks))
            return

        real_client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
        caching_client = CachingOpenAIClient(real_client)
        rows = build_eval_rows(load_questions(), version_ranks)
        retriever = PgVectorRetriever(conn, openai_client=caching_client)
        results = run_eval(retriever, rows)

    print_table(results)
    path = save_results(results)
    print(f"\n{caching_client.embeddings.call_count} embedding API call(s) made this run.")
    print(f"Detailed results saved to {path}")


if __name__ == "__main__":
    main()
