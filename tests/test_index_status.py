from app.index_status import get_vector_index_status


def test_get_vector_index_status_returns_missing_for_missing_index(tmp_path) -> None:
    index_dir = tmp_path / "vector_store"

    status = get_vector_index_status(index_dir)

    assert status["exists"] is False
    assert status["chunk_count"] == 0
    assert status["embedding_provider"] is None
    assert status["index_path"].endswith("index.json")


def test_get_vector_index_status_returns_summary_for_existing_index(
    tmp_path,
    monkeypatch,
) -> None:
    index_dir = tmp_path / "vector_store"
    index_dir.mkdir()
    (index_dir / "index.json").write_text("{}", encoding="utf-8")

    def fake_load_vector_index(index_dir_arg):
        assert index_dir_arg == index_dir

        return {
            "embedding_provider": "hash",
            "embedding_model": "hash-64",
            "embedding_dimension": 64,
            "chunks": [
                {"chunk_id": "chunk-1"},
                {"chunk_id": "chunk-2"},
            ],
        }

    monkeypatch.setattr(
        "app.index_status.load_vector_index",
        fake_load_vector_index,
    )

    status = get_vector_index_status(index_dir)

    assert status["exists"] is True
    assert status["chunk_count"] == 2
    assert status["embedding_provider"] == "hash"
    assert status["embedding_model"] == "hash-64"
    assert status["embedding_dimension"] == 64