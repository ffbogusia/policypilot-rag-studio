# Architecture Walkthrough

PolicyPilot RAG Studio is a local-first RAG portfolio project for working with synthetic policy documents.

The project demonstrates how policy documents can be loaded, split into chunks, embedded, searched, used as grounded evidence for answers, and evaluated with golden questions.

## 1. Local-first design

The default project mode is local-first and zero-cost.

That means:

- sample documents are stored locally in `data/sample_policies/`
- embeddings can be generated locally
- the vector index is stored locally in `.cache/vector_store/`
- the Streamlit app runs locally
- tests and evaluation run locally
- no paid API is required for the default workflow

Cloud services can be added later as optional extensions, but they are not required for the MVP.

## 2. Current RAG pipeline

The current RAG pipeline looks like this:

```text
Synthetic policy documents
        ↓
Document loader
        ↓
DocumentRecord objects
        ↓
Chunking
        ↓
ChunkRecord objects
        ↓
Embeddings
        ↓
Local vector index
        ↓
Retrieval
        ↓
Retrieved chunks
        ↓
Answer generation
        ↓
Citations + grounding check
        ↓
Streamlit UI + RAG Debugger
        ↓
Golden-question evaluation

3. Main modules
data/sample_policies/

This folder contains synthetic policy documents.

The documents are intentionally synthetic so the project can be safely shown in a public portfolio without exposing real company data.

Examples:

access policy
MFA policy
incident response policy
password policy
remote work policy
privacy policy
responsible AI usage policy
ingestion/load_documents.py

This module loads Markdown policy files from disk.

It extracts:

document text
metadata from frontmatter
source path
document ID

The output is a list of DocumentRecord objects.

ingestion/chunk_documents.py

This module splits loaded documents into smaller chunks.

Chunking is important because RAG systems usually retrieve smaller pieces of documents, not entire long files.

Each chunk keeps metadata such as:

source document
title
heading
chunk ID

The output is a list of ChunkRecord objects.

ingestion/create_embeddings.py

This module turns text chunks into vectors.

A vector is a numerical representation of text. Similar texts should have vectors that are closer to each other.

The project supports local embedding providers, including a deterministic hash provider for simple local testing.

ingestion/build_index.py

This module builds the local vector index.

It loads documents, chunks them, embeds the chunks, and saves the result to:

.cache/vector_store/index.json

This index is ignored by Git because it is generated locally.

retrieval/vector_search.py

This module searches the local vector index.

Given a user question, it:

embeds the question
compares the question vector with chunk vectors
ranks chunks by similarity score
returns the top matching chunks

The output is a RetrievalResult.

retrieval/retriever.py

This module is a small routing layer for retrieval.

Currently it routes to vector search, but it gives the project a clean place to support more retrieval modes later.

rag/answer_generator.py

This module generates a grounded answer from retrieved chunks.

It does not simply answer from memory. It uses the retrieved policy chunks as evidence.

It can also refuse to answer when the available documents do not provide enough support.

The output is an AnswerResult.

rag/citation_builder.py

This module builds citations from retrieved chunks.

Citations help users see which document chunks were used as sources for the answer.

rag/grounding_checker.py

This module checks whether the answer is grounded in retrieved evidence.

It helps detect whether the answer has citations and whether refusal behavior is appropriate.

app/streamlit_app.py

This is the user interface.

The Streamlit app allows a user to:

ask policy questions
see the generated answer
inspect sources
inspect grounding status
open the RAG Debugger
RAG Debugger

The RAG Debugger shows what happened before the answer was generated.

It displays:

retrieved chunk rank
similarity score
chunk ID
source path
heading
preview text
whether a chunk was cited

This makes the RAG pipeline more transparent and easier to debug.

eval/golden_questions.jsonl

This file contains golden questions.

A golden question defines expected behavior, for example:

which source should be retrieved
whether the system should refuse to answer
which key phrases should appear in the answer
eval/run_eval.py

This module runs the local RAG evaluation.

It loads golden questions, runs the RAG pipeline, compares actual results with expected behavior, and prints a PASS/FAIL report.

It can also write a Markdown report to:

eval/eval_report.md
4. Request flow in the app

When a user asks a question in Streamlit, the flow is:

User question in Streamlit
        ↓
answer_question()
        ↓
retrieve()
        ↓
search_vector_index()
        ↓
top matching chunks
        ↓
generate_answer_from_retrieval()
        ↓
answer + citations + debug data
        ↓
Streamlit displays result

In simple words:

The user asks a policy question.
The system searches the local policy index.
The most relevant chunks are returned.
The answer generator builds an answer from those chunks.
The UI shows the answer, sources, and debug details.
5. Evaluation flow

The evaluation flow is:

golden_questions.jsonl
        ↓
load_golden_questions()
        ↓
run_evaluation()
        ↓
answer_question()
        ↓
evaluate_answer()
        ↓
PASS / FAIL results
        ↓
Markdown report

The evaluation checks:

expected source match
refusal behavior
required answer phrases
grounding status

This helps verify whether the RAG system behaves as expected.

6. Why this project uses synthetic documents

The project uses synthetic policy documents because portfolio projects should not expose real customer, employer, or internal company data.

Synthetic documents make it possible to demonstrate realistic RAG behavior safely.

7. Current limitations

The current MVP has intentional limitations:

the default hash embedding provider is deterministic but not semantically strong
the local JSON vector index is useful for learning and debugging, but not production-grade
the fallback answer generator is simple and deterministic
the project does not use real confidential documents
the project does not require cloud deployment by default

These limitations are acceptable for a local-first learning and portfolio project.

8. Planned engineering layers

The next planned layers are:

Docker

Docker will make the project easier to run in a reproducible local environment.

Planned files:

Dockerfile
.dockerignore
docker-compose.yml
Developer task commands

A Makefile or similar task runner can provide shortcuts such as:

make test
make index
make eval
make app
LangGraph workflow

LangGraph can be used to represent the RAG process as explicit workflow nodes.

A future LangGraph version may split the pipeline into steps such as:

retrieve chunks
        ↓
check evidence
        ↓
answer or refuse
        ↓
return citations and debug data

This would make the agentic workflow easier to inspect and extend.

MCP server

An MCP server can expose the RAG system as tools.

Possible tools:

search policy documents
answer a policy question
get a chunk by ID
run evaluation

This would allow other clients to interact with the RAG pipeline through tool calls.