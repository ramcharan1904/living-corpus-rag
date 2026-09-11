-- Version registry. Rank makes versions sortable; tags do not sort correctly
-- as strings ('v2.13.5' < 'v2.4.0' is true, which is wrong).
CREATE TABLE versions (
    version_rank   INT PRIMARY KEY,          -- 1, 2, 3 in chronological order
    version_tag    TEXT NOT NULL UNIQUE,     -- 'v2.4.0'
    commit_sha     TEXT NOT NULL,
    corpus_path    TEXT NOT NULL,            -- '../pyd-v2early'
    ingested_at    TIMESTAMPTZ
);

-- A logical document, tracked across versions by its path.
CREATE TABLE documents (
    doc_id         BIGSERIAL PRIMARY KEY,
    rel_path       TEXT NOT NULL,            -- 'docs/concepts/fields.md'
    library        TEXT NOT NULL DEFAULT 'pydantic',
    first_seen_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (library, rel_path)
);

-- One row per (document, version) that actually exists at that version.
CREATE TABLE doc_versions (
    doc_version_id BIGSERIAL PRIMARY KEY,
    doc_id         BIGINT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    version_rank   INT NOT NULL REFERENCES versions(version_rank),
    content_hash   TEXT NOT NULL,            -- hash of the whole file
    raw_path       TEXT NOT NULL,            -- where Bronze stored it
    fetched_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (doc_id, version_rank)
);

-- Chunks, with a validity interval expressed in version ranks.
CREATE TABLE chunks (
    chunk_id           BIGSERIAL PRIMARY KEY,
    doc_id             BIGINT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    heading_path       TEXT NOT NULL,        -- 'Fields > Field inspection'
    chunk_index        INT NOT NULL,         -- order within the document
    text               TEXT NOT NULL,
    content_hash       TEXT NOT NULL,
    valid_from_rank    INT NOT NULL REFERENCES versions(version_rank),
    valid_to_rank      INT REFERENCES versions(version_rank),  -- NULL = still current
    embedded_at        TIMESTAMPTZ,
    weaviate_uuid      UUID,
    UNIQUE (doc_id, heading_path, valid_from_rank)
);

CREATE INDEX idx_chunks_hash      ON chunks (content_hash);
CREATE INDEX idx_chunks_validity  ON chunks (valid_from_rank, valid_to_rank);
CREATE INDEX idx_chunks_doc       ON chunks (doc_id);
CREATE INDEX idx_docver_hash      ON doc_versions (content_hash);

-- Ingestion telemetry. This is where the headline number comes from.
CREATE TABLE ingest_runs (
    run_id            BIGSERIAL PRIMARY KEY,
    version_rank      INT NOT NULL REFERENCES versions(version_rank),
    started_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at       TIMESTAMPTZ,
    docs_seen         INT,
    chunks_new        INT,
    chunks_changed    INT,
    chunks_unchanged  INT,
    chunks_retired    INT,
    embed_calls       INT,
    notes             TEXT
);
