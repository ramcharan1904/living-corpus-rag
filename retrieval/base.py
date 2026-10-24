"""The retrieval interface every backend (pgvector now, Weaviate later)
implements identically, so comparing them is a measurement, not a rewrite.

The version-aware/baseline distinction is not two code paths: baseline is
just this same retrieve() called with target_version_rank=3 (the newest
version). Whatever a backend does for version filtering, it does for both.
"""

from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass
class RetrievedChunk:
    chunk_id: int
    doc_id: int
    rel_path: str
    heading_path: str
    raw_text: str
    valid_from_rank: int
    valid_to_rank: int | None
    score: float


class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        target_version_rank: int,
        top_k: int,
        mode: Literal["dense", "keyword", "hybrid"] = "hybrid",
    ) -> list[RetrievedChunk]: ...
