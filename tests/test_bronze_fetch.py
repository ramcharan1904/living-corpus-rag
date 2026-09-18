from ingestion.bronze import fetch


def test_iter_scoped_md_files_excludes_configured_top_level(tmp_path, monkeypatch):
    # Pin the exclusion set explicitly rather than relying on whatever the
    # environment's .env happens to have, so this test can't go flaky
    # depending on who's running it.
    monkeypatch.setattr(
        fetch.settings,
        "excluded_top_level",
        {"contributing.md", "pydantic_people.md", "help_with_pydantic.md"},
    )

    corpus_root = tmp_path / "corpus"

    included = [
        corpus_root / "docs" / "concepts" / "fields.md",
        corpus_root / "docs" / "errors" / "errors.md",
        corpus_root / "docs" / "index.md",
        # Deliberately kept per the exclusion policy - proves this isn't
        # just "everything at the top level gets dropped".
        corpus_root / "docs" / "version-policy.md",
    ]
    excluded = [
        corpus_root / "docs" / "contributing.md",
        corpus_root / "docs" / "pydantic_people.md",
        corpus_root / "docs" / "help_with_pydantic.md",
        # Out of scope entirely (not in SCOPED_SUBDIRS, not a top-level file).
        corpus_root / "docs" / "api" / "base_model.md",
    ]
    for path in included + excluded:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")

    found = {p.relative_to(corpus_root).as_posix() for p in fetch.iter_scoped_md_files(corpus_root)}
    expected = {p.relative_to(corpus_root).as_posix() for p in included}

    assert found == expected


def _doc_version_count(conn, version_tag: str) -> int:
    # Wrapped in a transaction so this never leaves an implicit transaction
    # open on the connection (psycopg3 starts one on any bare execute() when
    # not in autocommit mode, and it lingers until something closes it -
    # which then makes a later, unrelated with conn.transaction() block a
    # nested savepoint instead of a real commit).
    with conn.transaction():
        row = conn.execute(
            """
            SELECT count(*)
            FROM doc_versions dv
            JOIN versions v ON v.version_rank = dv.version_rank
            WHERE v.version_tag = %s
            """,
            (version_tag,),
        ).fetchone()
    return row[0]


def test_second_run_inserts_nothing(pg_conn, test_version):
    version_tag = test_version

    fetch.fetch_version(pg_conn, version_tag)
    first_count = _doc_version_count(pg_conn, version_tag)
    assert first_count == 2  # the two fixture files in conftest.test_version

    fetch.fetch_version(pg_conn, version_tag)
    second_count = _doc_version_count(pg_conn, version_tag)

    assert second_count == first_count
