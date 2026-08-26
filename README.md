# PolicyPilot RAG Studio

A transparent, local-first RAG engineering lab built around policy and security documents.

## Why this project exists

I built PolicyPilot RAG Studio to show how a Retrieval-Augmented Generation system can be built end-to-end in a way that's transparent, testable, and doesn't rely on paid infrastructure.

It covers the full pipeline:

- document ingestion
- chunking
- local embeddings
- local vector search
- grounded answers with citations
- refusal behavior when the sources don't support an answer
- a RAG debugger
- evaluation against golden questions
- a LangGraph workflow
- MCP tools
- notes on responsible AI and security

## Local-first

The project runs locally by default and doesn't require Azure, Azure OpenAI, or Azure AI Search.

Cloud extensions might get documented at some point, but the default setup is meant to work entirely on your own machine.

## Stack

What I'm using so far:

- Python
- Streamlit
- LangChain
- LangGraph
- Sentence Transformers
- Chroma or FAISS
- Ollama, with a local fallback mode
- MCP Python SDK
- pytest
- GitHub Actions
- Markdown for documentation

## Documentation

- [Architecture overview](docs/architecture.md)
- [MCP integration](docs/mcp.md)

## Repository structure

```text
policypilot-rag-studio/
  app/
  config/
  data/
    sample_policies/
  ingestion/
  retrieval/
  rag/
  agent/
  mcp_server/
  eval/
  tests/
  docs/
  .github/workflows/
```

## MVP roadmap

- [ ] Initialize repository and sample policy documents
- [ ] Load Markdown documents with metadata
- [ ] Split documents into chunks
- [ ] Create local embeddings
- [ ] Build a local vector index
- [ ] Retrieve relevant chunks
- [ ] Generate grounded answers with citations
- [ ] Add Streamlit UI and RAG Debugger
- [ ] Add evaluation with golden questions
- [ ] Add LangGraph workflow
- [ ] Add MCP tools
- [ ] Add CI and portfolio documentation

## Sample use cases

Questions the app should be able to answer from the source documents:

- Can contractors access production data?
- What should an employee do after a phishing email?
- Is MFA required for privileged access?

Questions the app should refuse to answer:

- What is the CEO's favorite restaurant?
- What is the secret admin password?
- Which employee was fired?

## Responsible AI note

All sample documents in this repository are synthetic and were created for educational purposes.

This project shouldn't be used as real legal, HR, compliance, or security advice without human review.

---

# Sample policy documents

This folder contains synthetic policy documents created for educational purposes.

They're entirely fictional and don't contain any real company data, employee data, customer data, secrets, or confidential procedures.

They exist to test and demonstrate:

- document ingestion
- chunking
- metadata extraction
- retrieval
- grounded answers
- citations
- refusal behavior
- prompt-injection safety
- RAG evaluation