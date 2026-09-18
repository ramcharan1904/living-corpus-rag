-- Seed the version registry. The three Pydantic v2 doc snapshots this
-- corpus is built from. Ranks are chronological and sortable; version
-- tags are not ('v2.13.5' < 'v2.4.0' as a string).
INSERT INTO versions (version_rank, version_tag, commit_sha, corpus_path) VALUES
  (1, 'v2.4.0',  '1f59075c69e7172f8019b0cf621f0d9fde406307', '../pyd-v2early'),
  (2, 'v2.8.2',  '4978ee235bf5f0fa11159b2dfe46068ef3deba0a', '../pyd-v2mid'),
  (3, 'v2.13.5', '001dea020e0809844e5b17666432c9135a976f46', '../pyd-v2late')
ON CONFLICT (version_rank) DO NOTHING;