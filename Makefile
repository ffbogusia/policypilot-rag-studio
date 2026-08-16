.RECIPEPREFIX := >

PYTHON ?= python
EMBEDDING_PROVIDER ?= hash
REPORT_OUT ?= eval/eval_report.md

.PHONY: help test index eval app docker-config clean

help:
> @echo "Available commands:"
> @echo "  make test           Run the pytest suite"
> @echo "  make index          Build the local vector index"
> @echo "  make eval           Run local RAG evaluation and write Markdown report"
> @echo "  make app            Build index and start the Streamlit app"
> @echo "  make docker-config  Validate docker-compose.yml"
> @echo "  make clean          Remove local generated caches"

test:
> $(PYTHON) -m pytest -q

index:
> $(PYTHON) -m ingestion.build_index --provider $(EMBEDDING_PROVIDER)

eval: index
> $(PYTHON) -m eval.run_eval --report-out $(REPORT_OUT)

app: index
> streamlit run app/streamlit_app.py

docker-config:
> docker compose config

clean:
> rm -rf .cache/vector_store
> rm -rf .pytest_cache
> rm -rf .ruff_cache
> find . -type d -name "__pycache__" -prune -exec rm -rf {} +