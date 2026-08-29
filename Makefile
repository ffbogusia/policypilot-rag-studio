.RECIPEPREFIX := >

PYTHON ?= python
EMBEDDING_PROVIDER ?= hash
REPORT_OUT ?= eval/eval_report.md
QUALITY_MIN_PASS_RATE ?= 0.9

.PHONY: help test index eval quality-gate app mcp-smoke docker-config clean

help:
> @echo "Available commands:"
> @echo "  make test           Run the pytest suite"
> @echo "  make index          Build the local vector index"
> @echo "  make eval           Run local RAG evaluation and write Markdown report"
> @echo "  make app            Build index and start the Streamlit app"
> @echo "  make docker-config  Validate docker-compose.yml"
> @echo "  make clean          Remove local generated caches"
> @echo "  make mcp-smoke      Build index and run MCP smoke check"
> @echo "  make quality-gate   Run RAG evaluation with minimum pass rate"

test:
> $(PYTHON) -m pytest -q

index:
> $(PYTHON) -m ingestion.build_index --provider $(EMBEDDING_PROVIDER)

eval: index
> $(PYTHON) -m eval.run_eval --report-out $(REPORT_OUT)

quality-gate: index
> $(PYTHON) -m eval.run_eval --min-pass-rate $(QUALITY_MIN_PASS_RATE) --report-out $(REPORT_OUT)

app: index
> streamlit run app/streamlit_app.py

docker-config:
> docker compose config

clean:
> rm -rf .cache/vector_store
> rm -rf .pytest_cache
> rm -rf .ruff_cache
> find . -type d -name "__pycache__" -prune -exec rm -rf {} +

mcp-smoke: index
> $(PYTHON) -m mcp_server.smoke_check