-- Gold layer: vector search and keyword search over chunks.
--
-- Text search config is 'simple', not 'english' - stemming collapses
-- validate/validation/validated/validator to one root, and "validation"
-- is in nearly every chunk, so english would make keyword search match
-- almost everything. simple keeps identifier words apart (verbatim),
-- at the cost of not matching plural/tense variants in ordinary prose.
CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE chunks ADD COLUMN embedding vector(1536);

ALTER TABLE chunks ADD COLUMN text_search tsvector
    GENERATED ALWAYS AS (to_tsvector('simple', raw_text)) STORED;

CREATE INDEX idx_chunks_text_search ON chunks USING GIN (text_search);
