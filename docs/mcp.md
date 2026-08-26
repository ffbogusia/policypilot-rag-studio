# MCP Integration

PolicyPilot RAG Studio exposes selected local RAG capabilities as Model Context Protocol tools.

The MCP layer is designed as a thin wrapper around the existing Python implementation. The default project setup remains local-first, free to run, and based on synthetic policy documents.

## Purpose

The MCP integration makes the core RAG workflow callable as tools:

- answer a policy question with grounded citations,
- search policy documents and return retrieved chunks,
- inspect one indexed chunk by ID,
- run the local RAG evaluation suite.

This keeps the RAG logic reusable outside the Streamlit UI.

## Components

| File | Purpose |
| --- | --- |
| `mcp_server/tools.py` | Plain Python tool functions. They contain the reusable logic and return JSON-friendly dictionaries. |
| `mcp_server/server.py` | MCP server wrapper. It exposes selected functions from `tools.py` as MCP tools. |
| `mcp_server/smoke_check.py` | Local smoke check for verifying that the MCP tool layer can call the underlying RAG functionality. |
| `tests/test_mcp_tools.py` | Unit tests for the JSON-friendly tool functions. |
| `tests/test_mcp_server.py` | Tests for the MCP wrapper functions. |
| `tests/test_mcp_smoke_check.py` | Test for the smoke check status output. |

## Available tools

### `answer_policy_question`

Answers a policy-related question using the selected RAG execution mode.

Inputs:

- `question`
- `index_dir`
- `top_k`
- `execution_mode`

Supported execution modes:

- `classic`
- `langgraph`

The response contains the answer, refusal status, grounding status, cited chunk IDs, sources and debug metadata.

### `search_policy_docs`

Searches the indexed policy documents and returns retrieved chunks.

This tool is useful when the caller needs evidence chunks without generating a final answer.

Inputs:

- `query`
- `index_dir`
- `top_k`

### `get_chunk_by_id`

Returns one indexed chunk by its chunk ID.

This is useful for inspecting a specific source chunk returned by search, citations or the RAG Debugger.

Inputs:

- `chunk_id`
- `index_dir`

### `run_rag_eval`

Runs the local golden-question evaluation suite and returns a structured summary.

Inputs:

- `golden_path`
- `index_dir`
- `top_k`

The response includes pass/fail counts, pass rate and per-question evaluation details.

## Local usage

Before running MCP-related checks, build the local index:

```bash
make index
```

Run the full test suite:

```bash
make test
```

Run the MCP smoke check:

```bash
python -m mcp_server.smoke_check
```

Expected output includes status lines for:

- answer tool,
- search tool,
- chunk lookup tool,
- evaluation tool.

## Design notes

The MCP layer intentionally does not duplicate the RAG logic.

The split is:

```text
tools.py       -> reusable Python functions
server.py      -> MCP wrapper
smoke_check.py -> local verification helper
```

This makes the project easier to test because most behavior can be verified without starting an external MCP client.

## Local-first constraints

The MCP integration follows the same constraints as the rest of the project:

- no paid cloud resources are required,
- no real company documents are included,
- sample documents are synthetic,
- local vector indexes are generated artifacts and should not be committed,
- secrets must be provided through environment variables if needed in future extensions.

## Current limitations

The current MCP layer focuses on local tool exposure and testability.

It does not yet include:

- a production deployment setup,
- remote authentication,
- hosted MCP infrastructure,
- cloud-based vector storage,
- paid LLM integrations.

These are intentionally left out of the default version to keep the project simple, reproducible and cost-controlled.
