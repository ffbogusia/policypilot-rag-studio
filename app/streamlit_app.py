import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from app.demo_questions import get_default_question, get_demo_questions
from app.eval_diagnostics import run_eval_diagnostics
from app.rag_execution import run_policy_question
from ingestion.build_index import build_local_index
from app.index_status import get_vector_index_status


INDEX_DIR = Path(".cache/vector_store")
INDEX_FILE = INDEX_DIR / "index.json"


def ensure_index_exists() -> bool:
    """Return True if the local vector index exists."""

    return INDEX_FILE.exists()


def render_sources(sources: list[dict[str, object]]) -> None:
    """Render cited sources."""

    st.subheader("Sources")

    if not sources:
        st.info("No sources were cited for this answer.")
        return

    for source in sources:
        with st.expander(f"{source['rank']}. {source['title']} — {source['heading']}"):
            st.write(f"**Chunk ID:** `{source['chunk_id']}`")
            st.write(f"**Source path:** `{source['source_path']}`")
            st.write(f"**Score:** `{source['score']}`")
            st.write(source["preview"])


def render_grounding_status(status: str, refusal: bool) -> None:
    """Render grounding/refusal status."""

    if refusal:
        st.warning(
            "Refusal: the system did not find enough evidence in the policy documents."
        )
        return

    if status == "PASS":
        st.success("Grounding: PASS — the answer cites retrieved policy chunks.")
    elif status == "WARN":
        st.warning("Grounding: WARN — the answer may need human review.")
    else:
        st.error("Grounding: FAIL — the answer is not sufficiently supported.")


def main() -> None:
    st.set_page_config(
        page_title="PolicyPilot RAG Studio",
        page_icon="📚",
        layout="wide",
    )

    st.sidebar.title("PolicyPilot RAG Studio")
    st.sidebar.write(
        "Local-first RAG lab for policy documents, citations and evaluation."
    )
    st.sidebar.markdown("---")
    st.sidebar.write("**Default mode:** local / zero-cost")
    st.sidebar.write("**Vector index:** local JSON cache")
    st.sidebar.write("**Cloud:** optional later, disabled by default")

    execution_mode_label = st.sidebar.selectbox(
        "Execution mode",
        options=["Classic RAG", "LangGraph RAG"],
        help=(
            "Classic RAG uses direct function calls. "
            "LangGraph RAG runs the same flow as explicit workflow nodes."
        ),
    )

    execution_mode = (
        "langgraph"
        if execution_mode_label == "LangGraph RAG"
        else "classic"
    )

    st.sidebar.caption(f"Current mode: {execution_mode_label}")

    st.title("PolicyPilot RAG Studio")
    st.caption(
        "Local-first RAG assistant with citations, grounding checks and transparent retrieval."
    )

    st.markdown(
        """
        Ask a question about synthetic company policy documents.
        The app retrieves relevant chunks, generates a grounded answer,
        shows citations and explains whether the answer is supported by sources.
        """
    )

    st.info(
        "This is a local portfolio demo. It does not use paid cloud APIs by default."
    )
    with st.expander("Vector index status"):
        index_status = get_vector_index_status(INDEX_DIR)

        if index_status["exists"]:
            col1, col2, col3 = st.columns(3)
            col1.metric("Chunks", index_status["chunk_count"])
            col2.metric("Provider", index_status["embedding_provider"] or "-")
            col3.metric("Dimension", index_status["embedding_dimension"] or "-")

            st.write(f"**Model:** `{index_status['embedding_model']}`")
            st.write(f"**Index path:** `{index_status['index_path']}`")
        else:
            st.warning("Local vector index was not found.")
            st.write(f"Expected path: `{index_status['index_path']}`")

    if not ensure_index_exists():
        st.warning("Local vector index was not found.")

        if st.button("Build local index now"):
            with st.spinner("Building local index..."):
                index_path = build_local_index(
                    docs_dir="data/sample_policies",
                    output_dir=INDEX_DIR,
                    provider_name="hash",
                )

            st.success(f"Index built successfully: {index_path}")

        st.stop()

    demo_questions = get_demo_questions()
    demo_labels = [question["label"] for question in demo_questions]

    selected_label = st.selectbox(
        "Choose a demo question",
        options=demo_labels,
    )

    selected_demo_question = next(
        question for question in demo_questions if question["label"] == selected_label
    )

    question = st.text_area(
        "Question",
        value=selected_demo_question["question"] or get_default_question(),
        height=100,
    )

    top_k = st.slider(
        "Number of retrieved chunks",
        min_value=1,
        max_value=8,
        value=5,
    )

    with st.expander("Evaluation diagnostics"):
        st.write(
            "Run the local golden-question evaluation against the current vector index."
        )

        if st.button("Run local evaluation"):
            with st.spinner("Running local RAG evaluation..."):
                diagnostics = run_eval_diagnostics(
                    index_dir=INDEX_DIR,
                    top_k=top_k,
                )

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Passed", diagnostics["passed"])
            col2.metric("Failed", diagnostics["failed"])
            col3.metric("Total", diagnostics["total"])
            col4.metric("Pass rate", f"{diagnostics['pass_rate']:.0%}")

            failed_questions = diagnostics["failed_questions"]

            if failed_questions:
                st.warning("Some evaluation questions failed.")
                st.json(failed_questions)
            else:
                st.success("All evaluation questions passed.")

    if st.button("Ask PolicyPilot", type="primary"):
        if not question.strip():
            st.error("Please enter a question.")
            st.stop()

        with st.spinner("Retrieving sources and generating answer..."):
            result = run_policy_question(
                question=question,
                index_dir=INDEX_DIR,
                top_k=top_k,
                execution_mode=execution_mode,
            )

        st.subheader("Answer")
        st.write(result.answer)

        render_grounding_status(
            status=result.grounding_status,
            refusal=result.refusal,
        )

        render_sources(result.sources)

        with st.expander("Debug details"):
            st.write(f"**Execution mode:** `{result.debug.get('execution_mode')}`")
            st.write(f"**Model:** `{result.model_name}`")
            st.write(f"**Grounding status:** `{result.grounding_status}`")
            st.write(f"**Grounding reason:** {result.debug.get('grounding_reason')}")
            st.write(
                f"**Retrieved chunks:** {result.debug.get('retrieved_chunk_count')}"
            )

            retrieved_chunks = result.debug.get("retrieved_chunks", [])

            if retrieved_chunks:
                st.write("**Retrieved chunks used by the RAG pipeline:**")

                for chunk in retrieved_chunks:
                    cited_marker = (
                        "✅ cited"
                        if chunk["chunk_id"] in result.cited_chunk_ids
                        else "not cited"
                    )

                    with st.expander(
                        f"Rank {chunk['rank']} | Score {chunk['score']} | "
                        f"{chunk['title']} | {cited_marker}"
                    ):
                        st.write(f"**Chunk ID:** `{chunk['chunk_id']}`")
                        st.write(f"**Heading:** {chunk['heading']}")
                        st.write(f"**Source path:** `{chunk['source_path']}`")
                        st.write(chunk["preview"])

            workflow_debug = result.debug.get("workflow")

            if workflow_debug:
                st.write("**LangGraph workflow debug:**")
                st.json(workflow_debug)

            st.write("**Cited chunk IDs:**")
            st.json(result.cited_chunk_ids)

            st.write("**Prompt preview:**")
            st.code(result.debug.get("prompt_preview", ""), language="text")
    if st.button("Ask PolicyPilot", type="primary"):
        if not question.strip():
            st.error("Please enter a question.")
            st.stop()

        with st.spinner("Retrieving sources and generating answer..."):
            result = run_policy_question(
                question=question,
                index_dir=INDEX_DIR,
                top_k=top_k,
                execution_mode=execution_mode,
            )

        # ...reszta bloku odpowiedzi...


if __name__ == "__main__":
    main()