import importlib


def test_streamlit_app_imports() -> None:
    module = importlib.import_module("app.streamlit_app")

    assert hasattr(module, "main")
    assert hasattr(module, "ensure_index_exists")
    assert hasattr(module, "render_sources")
    assert hasattr(module, "render_grounding_status")