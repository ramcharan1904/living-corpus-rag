"""pgvector implementation of the Retriever interface (see retrieval/base.py).

dense: cosine distance (<=>) on chunks.embedding, version filter in the WHERE.

keyword: NOT websearch_to_tsquery/plainto_tsquery - golden questions are full
    sentences, and ANDing every word (what those functions do) matches
    almost no chunk, silently degrading hybrid to dense-only. Instead:
    OR across the query's words after dropping a short stopword list (the
    tsvector config is 'simple' - see migrations/0003_gold.sql - which,
    unlike 'english', keeps stopwords), plus a phrase-adjacency match for
    each identifier-like token (containing '_' or '()'), so e.g.
    "validate_by_name" matches as the adjacent phrase validate <-> by <->
    name, not three words scattered anywhere in the chunk. Built via
    Postgres's tsquery '||' (OR) operator over to_tsquery()/
    phraseto_tsquery() sub-expressions, not string concatenation - each
    function does the part it's actually good at.

hybrid: dense and keyword each run over a larger candidate pool
    (HYBRID_CANDIDATE_POOL), then fused with RRF.

version_filter_sql() and rrf() are left to be implemented - see their
docstrings for the exact contract this file's calling code depends on.
"""

import re
from dataclasses import replace

import numpy as np

import psycopg
from openai import OpenAI
from pgvector.psycopg import register_vector

from config import settings
from retrieval.base import RetrievedChunk

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMENSIONS = 1536
HYBRID_CANDIDATE_POOL = 50

# 'simple' doesn't strip these, unlike 'english'. Without removing them, a
# question like "How do I configure..." ORs in "how"/"do"/"i" as if they
# were meaningful search terms, drowning out the words that actually matter.
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "how", "what", "when", "where", "why", "which",
    "who", "whom", "to", "of", "in", "on", "for", "with", "and", "or",
    "but", "if", "then", "so", "this", "that", "these", "those", "i",
    "you", "he", "she", "it", "we", "they", "my", "your", "his", "her",
    "its", "our", "their", "as", "at", "by", "from", "will", "can",
}

_WORD = re.compile(r"[a-zA-Z0-9_]+")
_IDENTIFIER_LIKE = re.compile(r"\S*[_()]\S*")


def version_filter_sql() -> str:
    """SQL WHERE-fragment for the version rule.

    A chunk is valid at rank r when:
        valid_from_rank <= r AND (valid_to_rank IS NULL OR valid_to_rank >= r)

    Contract depended on by this file's calling code: return that condition
    as a string using a single named placeholder %(target_rank)s (appearing
    twice is fine - named params can repeat). Must be spliced into the WHERE
    clause itself, before ORDER BY/LIMIT - never applied as a post-filter on
    already-limited results, since retrieve-then-filter can return fewer
    than top_k rows.
    """
    return (
    "c.valid_from_rank <= %(target_rank)s "
     "AND (c.valid_to_rank IS NULL OR c.valid_to_rank >= %(target_rank)s)"
    )


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion.

    Each inner list in `rankings` is chunk_ids in rank order (best first,
    rank 1). Returns (chunk_id, fused_score) pairs, best (highest score)
    first. A chunk's score is the sum, over every list it appears in, of
    1 / (k + rank) - rank starting at 1. A chunk absent from a list
    contributes nothing for that list, not a penalty. 
    """
    scores = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)

def build_keyword_tsquery(query: str) -> tuple[str, dict[str, str]]:
    """Build a 'simple'-config tsquery SQL expression for a natural-language query.

    Returns (sql_expression, params): sql_expression is a SQL fragment of
    type tsquery (safe to splice into a larger query, e.g. inside
    `text_search @@ (...)`), and params are the named values it needs bound.

    OR across the query's significant words (after dropping STOPWORDS),
    each as its own to_tsquery('simple', word) term, plus a
    phraseto_tsquery('simple', token) term for each identifier-like token
    (containing '_' or '()').
    """
    identifier_tokens = [t.strip("()") for t in _IDENTIFIER_LIKE.findall(query)]
    identifier_lower = {t.lower() for t in identifier_tokens}

    plain_words = [
        w for w in _WORD.findall(query.lower())
        if w not in STOPWORDS and w not in identifier_lower
    ]

    terms_sql: list[str] = []
    params: dict[str, str] = {}

    for i, word in enumerate(plain_words):
        key = f"kw_word_{i}"
        terms_sql.append(f"to_tsquery('simple', %({key})s)")
        params[key] = word

    for i, token in enumerate(identifier_tokens):
        key = f"kw_ident_{i}"
        terms_sql.append(f"phraseto_tsquery('simple', %({key})s)")
        params[key] = token

    if not terms_sql:
        # Every word was a stopword (e.g. "What is it?") - fall back to the
        # raw query so this never produces an empty/invalid tsquery.
        fallback = (_WORD.findall(query.lower()) or ["simple"])[0]
        terms_sql.append("to_tsquery('simple', %(kw_fallback)s)")
        params["kw_fallback"] = fallback

    return " || ".join(terms_sql), params


class PgVectorRetriever:
    """pgvector-backed Retriever (see retrieval/base.py for the interface)."""

    def __init__(self, conn: psycopg.Connection, openai_client: OpenAI | None = None):
        register_vector(conn)
        self.conn = conn
        self.client = openai_client or OpenAI(api_key=settings.openai_api_key.get_secret_value())

    def retrieve(
        self,
        query: str,
        target_version_rank: int,
        top_k: int,
        mode: str = "hybrid",
    ) -> list[RetrievedChunk]:
        if mode == "dense":
            return self._dense(query, target_version_rank, top_k)
        if mode == "keyword":
            return self._keyword(query, target_version_rank, top_k)
        if mode == "hybrid":
            return self._hybrid(query, target_version_rank, top_k)
        raise ValueError(f"unknown mode: {mode!r}")

    def _embed_query(self, query: str) -> np.ndarray:
        response = self.client.embeddings.create(model=EMBED_MODEL, input=[query], dimensions=EMBED_DIMENSIONS)
        return np.array(response.data[0].embedding)

    def _dense(self, query: str, target_version_rank: int, top_k: int) -> list[RetrievedChunk]:
        qvec = self._embed_query(query)
        version_sql = version_filter_sql()
        with self.conn.transaction():
            rows = self.conn.execute(
                f"""
                SELECT c.chunk_id, c.doc_id, d.rel_path, c.heading_path, c.raw_text,
                       c.valid_from_rank, c.valid_to_rank,
                       1 - (c.embedding <=> %(qvec)s) AS score
                FROM chunks c
                JOIN documents d ON d.doc_id = c.doc_id
                WHERE c.embedding IS NOT NULL AND ({version_sql})
                ORDER BY c.embedding <=> %(qvec)s
                LIMIT %(top_k)s
                """,
                {"qvec": qvec, "target_rank": target_version_rank, "top_k": top_k},
            ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def _keyword(self, query: str, target_version_rank: int, top_k: int) -> list[RetrievedChunk]:
        tsquery_sql, kw_params = build_keyword_tsquery(query)
        version_sql = version_filter_sql()
        params = {**kw_params, "target_rank": target_version_rank, "top_k": top_k}
        with self.conn.transaction():
            rows = self.conn.execute(
                f"""
                SELECT c.chunk_id, c.doc_id, d.rel_path, c.heading_path, c.raw_text,
                       c.valid_from_rank, c.valid_to_rank,
                       ts_rank_cd(c.text_search, ({tsquery_sql})) AS score
                FROM chunks c
                JOIN documents d ON d.doc_id = c.doc_id
                WHERE c.text_search @@ ({tsquery_sql}) AND ({version_sql})
                ORDER BY score DESC
                LIMIT %(top_k)s
                """,
                params,
            ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def _hybrid(self, query: str, target_version_rank: int, top_k: int) -> list[RetrievedChunk]:
        dense_hits = self._dense(query, target_version_rank, HYBRID_CANDIDATE_POOL)
        keyword_hits = self._keyword(query, target_version_rank, HYBRID_CANDIDATE_POOL)

        dense_ids = [h.chunk_id for h in dense_hits]
        keyword_ids = [h.chunk_id for h in keyword_hits]

        by_id = {h.chunk_id: h for h in dense_hits + keyword_hits}
        fused = rrf([dense_ids, keyword_ids])[:top_k]

        return [replace(by_id[chunk_id], score=score) for chunk_id, score in fused if chunk_id in by_id]

    def _row_to_chunk(self, row) -> RetrievedChunk:
        chunk_id, doc_id, rel_path, heading_path, raw_text, valid_from_rank, valid_to_rank, score = row
        return RetrievedChunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            rel_path=rel_path,
            heading_path=heading_path,
            raw_text=raw_text,
            valid_from_rank=valid_from_rank,
            valid_to_rank=valid_to_rank,
            score=score,
        )
