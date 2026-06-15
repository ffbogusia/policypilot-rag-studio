# PolicyPilot RAG Studio

Transparent, local-first RAG engineering lab for policy and security documents.

## Why this project exists

PolicyPilot RAG Studio demonstrates how to build a Retrieval-Augmented Generation system that is transparent, testable and cost-aware.

The goal is not to build a random chatbot. The goal is to show the full RAG pipeline:

- document ingestion,
- chunking,
- local embeddings,
- local vector search,
- grounded answers with citations,
- refusal behavior when sources are insufficient,
- RAG debugging,
- evaluation with golden questions,
- LangGraph workflow,
- MCP tools,
- responsible AI and security notes.

## Local-first / zero-cost by default

This project is designed to run locally by default.

It does not require:

- Azure,
- Azure OpenAI,
- Azure AI Search,
- OpenAI API,
- Claude API,
- paid hosting.

Optional cloud extensions may be documented later, but the default version uses local tools.

## Planned stack

- Python
- Streamlit
- LangChain
- LangGraph
- Sentence Transformers
- Chroma or FAISS
- Ollama or local fallback mode
- MCP Python SDK
- pytest
- GitHub Actions
- Markdown documentation

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



MVP roadmap
Initialize repository and sample policy documents.
Load Markdown documents with metadata.
Split documents into chunks.
Create local embeddings.
Build a local vector index.
Retrieve relevant chunks.
Generate grounded answers with citations.
Add Streamlit UI and RAG Debugger.
Add evaluation with golden questions.
Add LangGraph workflow.
Add MCP tools.
Add CI and portfolio documentation.
Sample use cases

Example questions the app should answer from sources:

Can contractors access production data?
What should an employee do after a phishing email?
Is MFA required for privileged access?

Example questions the app should refuse:

What is the CEO's favorite restaurant?
What is the secret admin password?
Which employee was fired?
Responsible AI note

All sample documents in this repository are synthetic and created for educational purposes.

This project should not be used as real legal, HR, compliance or security advice without human review.


# Sample policy documents

This folder contains synthetic policy documents created for educational and portfolio purposes.

The documents are fictional. They do not contain real company data, employee data, customer data, secrets or confidential procedures.

The purpose of these documents is to test and demonstrate:

- document ingestion,
- chunking,
- metadata extraction,
- retrieval,
- grounded answers,
- citations,
- refusal behavior,
- prompt-injection safety,
- RAG evaluation.