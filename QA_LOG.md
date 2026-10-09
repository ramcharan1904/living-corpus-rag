# Q&A Log — living-corpus-rag build

A running record of the questions asked and the decisions/understanding reached
while building this project with Claude Code. Organized chronologically by
pipeline stage. Not a design doc — a reference for "why is it this way."

---

## Phase 0 — Scaffold

No notable Q&A beyond the initial scaffolding request (directory structure,
docker-compose, migrations, config module, smoke test). See `README.md` and
`docker-compose.yml` for the resulting structure.

---

## Phase 1 — Bronze (`ingestion/bronze/fetch.py`)

**Q: Explain the flow of fetch.py with examples.**
Walked through: `main()` → `get_version_row()` (resolves `version_tag` to
`version_rank` + absolute `corpus_path`) → `iter_scoped_md_files()` (recursive
walk of `docs/concepts`, `docs/errors`, `docs/integrations`; non-recursive
`docs/*.md`) → per-file loop: `upsert_document()` (get-or-create `documents`
row) → `doc_version_exists()` (skip-if-already-ingested check) → hash raw
*bytes* (not decoded text, so the hash is reproducible across platforms) →
copy to `bronze/<version_tag>/<rel_path>` → insert `doc_versions` row.

**Q: Brief review of what's built so far?**
Flagged open items at the time: actual file count (43) ran above the
35–40 estimate; only one version had been run through Bronze; no automated
tests yet; nothing committed; Postgres on port 5433 not 5432 on this machine.

**Q: Help me write a test that the exclusion list filters correctly, and one
that a second run inserts nothing — don't write the code yet.**
Design-only discussion first. Found there was no exclusion list yet (scope
was purely inclusion-based) — this fed directly into the next item.

**Decision: which top-level files to exclude, and where the exclusion list
lives.**
Final answer (from the user): exclude `contributing.md`, `pydantic_people.md`,
`help_with_pydantic.md`. Deliberately *keep* `version-policy.md` — version
policy is something a real Pydantic user would ask about, unlike the
contributor list. Lives at `config.Settings.excluded_top_level`, applied only
to the non-recursive `docs/*.md` pass.

**Q: Is port 5432 available (so the test DB could use the default port)?**
No — a native `postgresql-x64-18` Windows service holds it permanently on
this machine. Couldn't stop it (needs an elevated shell this session doesn't
have). Decision: tests reuse the existing dev container on port 5433, scoped
safely with a throwaway `version_rank=999` and `__test_fixture__`-marked
filenames so fixture data can never collide with real ingested data.

**Bug found while writing the tests:** a bare `conn.execute()` (no
`with conn.transaction():` wrapper) in a test helper left an implicit
transaction open. Every later `with conn.transaction():` on that connection
became a nested *savepoint* instead of an independent commit, and closing the
connection mid-transaction discarded work inconsistently (empirically: one
table's delete "stuck," another's didn't, from the same block). **Rule drawn
from this: every `conn.execute()`, including plain reads, must sit inside a
`with conn.transaction():` block.** Fixed in the test fixture; later found
the *same* bug in `fetch.py`'s own `get_version_row()` (see below).

**Q: Add `ingestion/silver/hash.py` as pasted.**
Found the module's own bundled self-check contradicted its docstring: a
reflowed paragraph was *not* hashing the same as the original, because
`normalize()` collapsed whitespace within each line but never merged
soft-wrapped lines back into one line. Fixed by grouping consecutive
non-blank prose lines into a single paragraph before hashing; code blocks
still pass through byte-for-byte, untouched.

**Important correction: Bronze's crash-resilience claim was wrong.**
While building `ingestion/silver/write.py`, re-examined `fetch.py` and found
`get_version_row()` had the exact bare-`execute()` bug described above. This
meant the earlier claim "each file's transaction is independent, so a crash
on file #30 wouldn't lose files 1–29" was **false** — everything was actually
one long nested-savepoint chain that only truly committed when the whole
`with psycopg.connect(...) as conn:` block exited. Fixed by wrapping
`get_version_row()`'s query properly; verified the fix with a direct
`tx_status` check, and confirmed idempotent re-runs and the test suite still
passed afterward.

---

## Phase 1.5 — Docker / environment

**Q: `docker compose up -d` — handle the port 5432 conflict.**
Same native Postgres 18 service as above. Offered two options (remap to
another port vs. stop the native service); stopping it needs admin rights
this session doesn't have, so landed on remapping — `POSTGRES_PORT=5433` in
`.env`.

---

## Phase 2, Step 1 — pgvector + text search (`migrations/0003_gold.sql`)

**Q: Before choosing `english` or `simple` for the tsvector config, show me
how each tokenizes `validate_by_name`.**
Ran `to_tsvector('english', ...)` vs `to_tsvector('simple', ...)`. Headline
finding: *both* configs split on underscores — `validate_by_name` is never
indexed as one atomic token either way. The real difference: `english` drops
stopwords and stems (`validate`→`valid`, `populate`→`popul`), `simple` keeps
every word verbatim. **Decision: `simple`** — reasoning given: `english`
would stem `validate/validation/validated/validator` all down to `valid`,
and "validation" appears in nearly every chunk, so keyword search would
match almost everything.

**Q: Before finalizing, show `plainto_tsquery` vs `phraseto_tsquery` for
`validate_by_name` under `simple`.**
`plainto_tsquery` → `'validate' & 'by' & 'name'` (unordered AND — loose,
since "by" is common). `phraseto_tsquery` → `'validate' <-> 'by' <-> 'name'`
(adjacency-ordered — effectively reconstructs the identifier). Noted for
**Step 4 (not yet implemented)**: the plan's original choice of
`websearch_to_tsquery`/`plainto_tsquery` with all terms ANDed together would
badly break recall on natural-language golden questions (an AND of every
word matches almost nothing). Planned fix for when Step 4 arrives: OR across
the query's words (after stripping a small stopword list, since `simple`
keeps them) plus a `phraseto_tsquery` sub-query OR'd in for each
identifier-like token (containing `_` or `` ` ``). Must show the generated
tsquery and result count for q001 before moving past that step.

**Rebuild + regression check.** After switching to `pgvector/pgvector:pg16`
and adding the migration, wiped and rebuilt the DB, re-ran Bronze+Silver for
all three versions, and confirmed the exact same counts as before the
rebuild (389 new / 57.7% unchanged / 10.7% unchanged + 141 retired) —
proof the pipeline is deterministic and the schema change didn't silently
alter behavior.

**Column-naming mismatch caught, not silently fixed.** The migration named
the generated tsvector column `raw_text_tsv`; a later probe query expected
`text_search`. Flagged the mismatch and asked which name was correct rather
than guessing. **Decision: rename to `text_search`.** Required a second
full rebuild + re-run to take effect, since migrations only apply on a fresh
volume.

---

## Phase 2, Step 2 — Embeddings (`ingestion/gold/embed.py`)

**Q: Why `SecretStr` for `openai_api_key` — to make sure the key isn't
leaked into logs, right?**
Confirmed yes. `SecretStr` masks the value in `repr()`/`str()` (prints
`**********`), so an accidental `print(settings)`, a naive logging call, or
an uncaught traceback that reprs the object won't leak it. The real value
stays reachable, but only via an explicit `.get_secret_value()` call — so
any code path that exposes it does so on purpose, never by accident.

**`.gitignore` check requested: is `.env.example` tracked?**
Yes — `git ls-files .env.example` confirmed it, and `.gitignore` already had
`!.env.example` right after the `.env.*` wildcard from the original
scaffold. No change needed.

**Q: Explain the `UPDATE ... SET embedding = %s` query — it looks new.**
Covered: `%s` is psycopg's parameterized placeholder, not Python string
formatting — the SQL text and values are sent to Postgres separately, which
is what makes it safe against injection despite the superficial resemblance.
Params map positionally to the `%s`s in order. The actually-new part: the
first parameter is a raw Python `list` of 1536 floats — binding that
against a `vector` column requires `pgvector.psycopg.register_vector(conn)`
to be called once per connection; without it, psycopg has no idea how to
serialize a Python list into pgvector's wire format.

**Checked proactively, given the project's history of this exact bug:**
does `register_vector()` leave a dangling implicit transaction open (like
the bare-`execute()` bug found twice already)? Verified empirically via a
direct `tx_status` check — no, it's clean.

**Q: Explain `valid_from_rank` / `valid_to_rank` — why are they required?**
Core idea: each `chunks` row represents content valid across a *range* of
version ranks, not one version. `valid_from_rank` = the rank that introduced
this exact content; `valid_to_rank` = the last rank it was still current, or
`NULL` = still current. Walked through the real "Alias" heading rows from
the DB: one row closed at rank 1 (the old v2.4.0 content), one open from
rank 2 onward (content unchanged between v2.8.2 and v2.13.5). Without
`valid_from_rank`, you couldn't tell "always existed" from "added partway
through." Without `valid_to_rank`, every row would look permanently current,
and a stale answer would still match current-version queries. (Forward note:
the later Weaviate sync step uses `valid_to_rank = 999` instead of `NULL` as
its "still open" sentinel — same concept, different marker, since Weaviate
doesn't do `IS NULL` range comparisons as cleanly.)

**Q: So the first (historical, closed) Alias row also gets embedded now,
right?**
Confirmed yes, and why it has to be: `fetch_unembedded()`'s query has no
`valid_to_rank` filter at all, by design. Version-aware retrieval at rank 1
needs a real vector for the *historical* content to be findable via dense
search — if only open/current rows were embedded, asking "what did v2.4.0
say about X" would return nothing (or the wrong, current answer) for
anything that later changed. Cross-checked against the user's own
prediction (389/199/453 = 1,041 total) as supporting evidence — that number
only makes sense if every row, open and closed, gets embedded.

**Real failure on first run, diagnosed and fixed:** `embed.py` failed with
`openai.RateLimitError` — root cause was `insufficient_quota` (zero OpenAI
credits), not a transient rate limit. The original retry logic treated
*every* `RateLimitError` as transient and burned through 5 attempts
(~30s of backoff) before failing. Inspected the exception's actual
attributes (`.type`, `.code`, inherited from `APIStatusError`/`APIError`)
and found OpenAI distinguishes the two cases via `exc.type`:
`"insufficient_quota"` is permanent (billing issue), anything else under
`RateLimitError` is genuinely transient. Fixed by checking that type and
failing fast with an actionable message instead of retrying. Verified the DB
stayed completely untouched (0 embedded, `embed_calls` still null) across
both failed attempts — safe to re-run once credits are added, since
`fetch_unembedded()`'s `WHERE embedding IS NULL` means nothing gets
double-charged.

**Status at time of writing:** blocked on the user adding OpenAI credits.
Everything else in Step 2 (`config.py`, `.env.example`, `pyproject.toml`
deps, `embed.py` itself) has been written and reviewed file-by-file; tests
for `embed.py` and the real run are still pending.

---

## Working conventions established along the way

- Work one file/step at a time; stop after each for full diff + plain-words
  explanation + one concrete example + any uncertainty flagged.
- Small diffs; no refactoring outside the current step; nothing committed
  by Claude — the user reviews and commits.
- Steps marked "RAM WRITES": signature + docstring + tests only, body
  `raise NotImplementedError`.
- Every `conn.execute()` sits inside `with conn.transaction():`, reads
  included — found and fixed the opposite twice (a test fixture, then
  `fetch.py`) before making this a hard rule.
- Scripts are idempotent: re-running does no extra work and is safe to do
  as often as needed.
- Tests make zero live network calls; mock external clients; Postgres-backed
  tests skip cleanly if the DB isn't reachable.
- Migrations only take effect on a fresh Postgres volume — any schema change
  means `docker compose down -v && up -d` plus re-running Bronze+Silver for
  all versions, then confirming counts still match as a regression check.
- Postgres runs on host port 5433, not 5432, on this machine.
