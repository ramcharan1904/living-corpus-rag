# living-corpus-rag

Version-aware RAG system over Pydantic documentation. Retrieves and reasons
across three points in the Pydantic v2 history (early / mid / late), sourced
from sibling worktrees at `../pyd-v2early`, `../pyd-v2mid`, `../pyd-v2late`.

## Layout

- `ingestion/` — bronze (raw) / silver (cleaned) / gold (chunked+embedded) pipeline stages
- `retrieval/` — retrieval logic over Weaviate + Postgres metadata
- `graph/` — LangGraph orchestration
- `api/` — FastAPI service
- `eval/` — `golden/` question sets and `fixtures/` test data
- `migrations/` — Postgres schema (plain SQL, applied via `docker-entrypoint-initdb.d`)
- `docker/` — supporting Dockerfiles
- `tests/` — pytest suite

## Setup

```bash
cp .env.example .env
docker compose up -d
pip install -e ".[dev]"
pytest
```

This repo is currently scaffolding only — no ingestion, retrieval, or graph
logic has been implemented yet.
