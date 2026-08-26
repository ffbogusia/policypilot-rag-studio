# MCP Integration

I added an MCP (Model Context Protocol) layer to PolicyPilot RAG Studio that exposes some of the RAG functionality as callable tools. The idea was simple: the logic I already had in Python shouldn't be locked inside the Streamlit UI — it should be usable from other places too.

The project itself still runs entirely locally, with no paid cloud services, on synthetic policy documents. MCP doesn't change any of that — it's just a thin wrapper on top of the existing code.

## Why

I wanted the core RAG workflow to be callable as a set of tools:

- answer a policy question and back it up with citations,
- search the indexed documents and return the matching chunks,
- look up a specific chunk by its ID,
- run the local evaluation suite against a golden question set.

This way the RAG logic isn't tied to the UI and can be reused elsewhere.

## How it's organized

| File | What it does |
| --- | --- |
| `mcp_server/tools.py` | Plain Python functions — this is where the actual logic lives, returning data that's already JSON-friendly. |
| `mcp_server/server.py` | The MCP wrapper — takes the functions from `tools.py` and exposes them as MCP tools. |
| `mcp_server/smoke_check.py` | A quick local check to make sure the MCP layer is actually talking to the underlying RAG logic correctly. |
| `tests/test_mcp_tools.py` | Unit tests for the functions in `tools.py`. |
| `tests/test_mcp_server.py` | Tests for the MCP wrapper. |
| `tests/test_mcp_smoke_check.py` | Test for the smoke check output. |

## Available tools

### `answer_policy_question`

Answers a policy-related question using the selected RAG execution mode.

Inputs:

- `question`
- `index_dir`
- `top_k`
- `execution_mode`

Supported modes:

- `classic`
- `langgraph`

The response includes the answer itself, whether the model refused to answer, whether the answer is actually grounded in sources, the cited chunk IDs, the sources, and some debug metadata.

### `search_policy_docs`

Searches the indexed documents and returns the matching chunks.

Useful when you just need the evidence chunks without generating a final answer.

Inputs:

- `query`
- `index_dir`
- `top_k`

### `get_chunk_by_id`

Returns a specific indexed chunk by its ID.

Handy for inspecting a source chunk that showed up in search results, citations, or the RAG Debugger.

Inputs:

- `chunk_id`
- `index_dir`

### `run_rag_eval`

Runs the local golden-question evaluation suite and returns a structured summary.

Inputs:

- `golden_path`
- `index_dir`
- `top_k`

The response includes pass/fail counts, the pass rate, and per-question details.

## Running it locally

Build the local index first:

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

The output should show status lines for the answer tool, the search tool, the chunk lookup tool, and the evaluation tool.

## A note on the architecture

I wanted the MCP layer to reuse the RAG logic, not duplicate it. So the split is:

```text
tools.py       -> the actual logic
server.py      -> MCP wrapper
smoke_check.py -> local verification
```

That makes most of the behavior testable without ever having to spin up an external MCP client.

## Constraints I'm sticking to

The MCP layer follows the same rules as the rest of the project:

- no paid cloud resources,
- no real company documents — everything is synthetic,
- local vector indexes are generated artifacts and aren't committed to the repo,
- any secrets added later should go through environment variables.

## What's not here yet

Right now the MCP layer is focused on exposing tools locally and making them testable. It doesn't include:

- a production deployment setup,
- remote authentication,
- hosted MCP infrastructure,
- cloud-based vector storage,
- paid LLM integrations.

I left these out on purpose to keep the default version simple, reproducible, and cheap to run.