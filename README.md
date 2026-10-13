# RAG Application

A Retrieval-Augmented Generation (RAG) application built with Python, FastAPI, LangChain, ChromaDB, local Hugging Face embeddings, a Cross-Encoder reranker and Google Gemini.

Documents are split into token-aware chunks, embedded locally and stored in a persistent ChromaDB collection. Questions are answered by retrieving the most relevant chunks, reranking them, and asking Gemini to write an answer that cites its sources.

Everything is available through a REST API (FastAPI), so documents can be loaded, added, updated and deleted one at a time, and questions can be asked programmatically (for example by an evaluation framework).

---

## Contents

- [Features](#features)
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Configuration (.env)](#configuration-env)
- [Run the API](#run-the-api)
- [API reference](#api-reference)
- [Typical workflow](#typical-workflow)
- [Command-line Q&A (main.py)](#command-line-qa-mainpy)
- [Data and security](#data-and-security)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

- REST API (FastAPI) with an auto-generated test page at `/docs`.
- **Initial load** of every supported file in the docs folder. Files already in the database are skipped, so running it again does not re-embed anything.
- **Add, update and delete a single document** without touching the rest of the database.
- **Question → answer** endpoint: retrieve, rerank, generate, and return the answer with its sources.
- Loads PDF (`.pdf`) and Word (`.docx`) files, recursively from the docs folder.
- One bad file never stops a batch; it is reported with a reason.
- Token-aware chunking (`tiktoken`), local embeddings (Sentence Transformers), cosine distance in ChromaDB.
- Cross-Encoder reranking before generation.
- Answers grounded in the retrieved context, with `[S1]`, `[S2]` citations.

---

## How it works

### Ingestion (writes to the database)

```
File in docs folder -> read text -> split into chunks -> embed each chunk -> store in ChromaDB
```

Every chunk is stored with:

| Field | Example | Purpose |
|---|---|---|
| `doc_id` | `policies/leave.pdf` | The file's path **inside the docs folder**, with forward slashes. Same on every machine, and matches S3 keys later. |
| Chunk ID | `policies/leave.pdf::3` | `{doc_id}::{chunk number}`. Predictable, so one file's chunks can be found, replaced or deleted. |

### Question answering (reads from the database)

```
Question -> embed -> ChromaDB finds RERANK_CANDIDATES nearest chunks (default 20)
         -> Cross-Encoder reranker keeps the best top_k (default 5)
         -> Gemini writes the answer citing [S1], [S2] ...
         -> answer + sources
```

The **same embedding model** must be used for ingestion and questions, or search results are meaningless.

---

## Project structure

```text
RAG/
├── app.py                     # FastAPI app: all REST endpoints
├── main.py                    # Interactive command-line Q&A (no ingestion)
├── requirements.txt
├── .env.example               # Template for .env
├── .gitignore
├── README.md
├── data/                      # Git-ignored
│   ├── source/                # Your documents (never committed)
│   └── chroma_db/             # Vector database (generated)
├── ingestion/
│   ├── document_loader.py     # Find and read .pdf/.docx files, assign doc_id
│   ├── text_chunker.py        # Token-aware chunking
│   ├── vector_store.py        # ChromaDB access, chunk IDs, list loaded files
│   ├── initial_load.py        # Initial load: every file, skipping loaded ones
│   ├── document_ops.py        # Add / update / delete one file
│   └── embedding.py
├── retrieval/
│   ├── retriever.py           # Vector search + reranking
│   ├── reranker.py            # Cross-Encoder
│   └── query_processor.py
└── generation/
    └── generator.py           # Gemini answer with citations
```

---

## Requirements

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- A Google Gemini API key
- Internet access on the first run (embedding and reranker models are downloaded from Hugging Face and cached)
- Optional: LibreOffice, only if legacy `.doc` support is re-enabled (see [Known limitations](#known-limitations))

---

## Setup

All commands are run from the project root.

### Option A: uv (recommended)

uv is a fast Python package and environment manager. It is not part of FastAPI; it only sets up the environment.

**Windows (PowerShell)**
```powershell
python -m pip install uv
python -m uv venv
python -m uv pip install --python .venv -r requirements.txt
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
pip install uv
uv venv
uv pip install -r requirements.txt
source .venv/bin/activate
```

Notes:
- `uv venv` creates a `.venv` folder in the project root: an isolated environment with its own packages.
- On Windows, when uv is run as `python -m uv`, pass `--python .venv` so packages go into the project's `.venv` and not into the global Python.
- If activation is blocked on Windows, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
- If the download fails with an SSL or certificate error (corporate networks), add `--native-tls` to the install command.

### Option B: pip

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1    macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Check the environment

```bash
python -c "import sys; print(sys.prefix)"
```
The path should end with `.venv`. The prompt should also show the environment name, for example `(RAG)`.

### VS Code

Install the **Python** extension (publisher: Microsoft), then:
`Ctrl+Shift+P` → **Python: Select Interpreter** → choose the entry with `.venv` (marked *Workspace*).
The status bar then shows the venv, and new terminals open inside it.

### Configure and add documents

1. Copy `.env.example` to `.env` and fill in your values (see below).
2. Put `.pdf` / `.docx` files in `data/source/` (subfolders are allowed).

---

## Configuration (.env)

Create `.env` in the project root. **Never commit it.**

```env
# Documents and database
SOURCE_DIR=data/source
CHROMA_DIR=data/chroma_db
COLLECTION_NAME=rag_documents

# Chunking (in tokens, not characters)
CHUNK_SIZE=250
CHUNK_OVERLAP=50

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Retrieval and reranking
RERANK_CANDIDATES=20
TOP_K=5
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_DEVICE=cpu

# Gemini
GOOGLE_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
GEMINI_TEMPERATURE=0

# Only needed if .doc support is re-enabled
# LIBREOFFICE_PATH=C:/Program Files/LibreOffice/program/soffice.exe
```

| Variable | Default | Notes |
|---|---|---|
| `SOURCE_DIR` | `data/source` | Docs folder. Relative paths are resolved from the project root. |
| `CHROMA_DIR` | `data/chroma_db` | Where ChromaDB stores its files. |
| `COLLECTION_NAME` | `rag_documents` | Changing it points to a new, empty collection. |
| `CHUNK_SIZE` | `250` | **Tokens.** Keep it at or below ~250: `all-MiniLM-L6-v2` only reads the first 256 tokens of a chunk, so anything beyond that is not searchable. |
| `CHUNK_OVERLAP` | `50` | Tokens shared between neighbouring chunks. Must be smaller than `CHUNK_SIZE`. |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Changing it requires deleting and rebuilding the database. |
| `RERANK_CANDIDATES` | `20` | How many chunks vector search hands to the reranker. Also the maximum `top_k`. |
| `TOP_K` | `5` | Chunks used for an answer in `main.py`. The API takes `top_k` per request (default 5). |
| `GOOGLE_API_KEY` | — | Required for answers. |

Changing `CHUNK_SIZE`, `CHUNK_OVERLAP` or `EMBEDDING_MODEL` changes how chunks are made. Delete `data/chroma_db` and run the initial load again afterwards.

---

## Run the API

```bash
# Development (auto-restarts when you save a .py file)
python -m uvicorn app:app --reload

# Demo / server (no auto-restart)
python -m uvicorn app:app
```

Open **http://127.0.0.1:8000/docs** for the interactive test page.

- `app:app` means "the file `app.py`, the variable `app` inside it".
- `python -m uvicorn` works even when `uvicorn` is not on PATH.
- **Do not use `--reload` on a server or in production.** It is a development convenience: it runs two processes and restarts (interrupting in-flight requests) whenever code changes.
- The server listens on `127.0.0.1` (this machine only) and port `8000`.
- The embedding model, reranker and Gemini client load on the first request that needs them, so that request is slower.

---

## API reference

All paths in request bodies use the file's location **inside the docs folder**, e.g. `"folder/file.pdf"`. A full path also works if it is inside the docs folder. In JSON, use forward slashes (`C:/...`) or double backslashes (`C:\\...`); single backslashes break the JSON (422).

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/ingest/initial` | Load every supported file; skip files already loaded |
| `GET` | `/documents` | List loaded files with chunk counts |
| `POST` | `/documents` | Add one file |
| `PUT` | `/documents` | Update one file |
| `DELETE` | `/documents?file_path=...` | Delete one file's chunks |
| `POST` | `/query` | Ask a question, get an answer with sources |

### `POST /ingest/initial`

Scans the docs folder (recursively), loads each supported file that is not yet in the database, and reports what happened. Files already loaded are skipped, so calling it twice adds nothing the second time.

Response `200`:
```json
{
  "total_files": 4,
  "ingested": [{"doc_id": "folder/a.pdf", "chunks": 12}],
  "skipped_already_loaded": ["b.docx"],
  "skipped_unsupported": ["notes.txt"],
  "failed": [{"doc_id": "scan.pdf", "reason": "No text found (empty or scanned file)"}]
}
```
`total_files` = ingested + skipped_already_loaded + skipped_unsupported + failed.
Errors: `400` if the docs folder does not exist.

### `GET /documents`

```json
{"folder/a.pdf": 12, "b.docx": 7}
```
Read-only. Use it to check the effect of every other call. Files that failed or are unsupported never appear here.

### `POST /documents` (add)

The file must already be in the docs folder.

Request:
```json
{"file_path": "folder/new.pdf"}
```
Response `201`:
```json
{"doc_id": "folder/new.pdf", "chunks": 9}
```

| Code | When |
|---|---|
| `201` | Added |
| `400` | Unsupported type, path outside the docs folder, or unreadable/empty/scanned file |
| `404` | File not found |
| `409` | Already in the database. Use `PUT` to replace it. |

### `PUT /documents` (update)

Replaces a file's chunks with its current content. The new version is read first; if that fails, the old chunks are kept. Chunks left over from a longer old version are removed.

Request:
```json
{"file_path": "folder/new.pdf"}
```
Response `200`:
```json
{"doc_id": "folder/new.pdf", "chunks_before": 9, "chunks_after": 6, "old_chunks_removed": 3}
```

| Code | When |
|---|---|
| `200` | Updated |
| `400` | Unsupported type, outside the docs folder, or unreadable new version (old chunks kept) |
| `404` | File not found, or not in the database yet. Use `POST` to add it. |

### `DELETE /documents?file_path=folder/new.pdf`

Removes every chunk of that file from the database. **The file itself is not deleted** from the docs folder.

Response `200`:
```json
{"doc_id": "folder/new.pdf", "chunks_deleted": 6, "file_still_in_folder": true}
```
If `file_still_in_folder` is `true`, the next initial load will add the file back.
Errors: `404` if the file is not in the database.

### `POST /query`

Request:
```json
{"question": "What does the leave policy say about carry-over?", "top_k": 5}
```
- `question`: required, not empty.
- `top_k`: optional, 1–20, default 5. How many reranked chunks are given to Gemini.

Response `200`:
```json
{
  "question": "What does the leave policy say about carry-over?",
  "answer": "Up to 10 days can be carried over [S1].",
  "sources": [
    {"source_id": "S1", "doc_id": "policies/leave.pdf", "page_number": 4, "reranker_score": 6.2153}
  ]
}
```
- `source_id` matches the `[S1]`, `[S2]` citations in the answer.
- `reranker_score`: higher means more relevant. It is a model score, not a probability.
- `page_number` is `null` for `.docx` files.

| Code | When |
|---|---|
| `200` | Answered |
| `400` | Question is only whitespace |
| `422` | Empty question, or `top_k` outside 1–20 |
| `502` | Gemini call failed (API key, network, quota) |

Ask about **content written in the documents**. File names and document numbers are not part of the searchable text.

---

## Typical workflow

1. Put documents in `data/source/`.
2. Start the API: `python -m uvicorn app:app`.
3. `POST /ingest/initial` once. Check `GET /documents`.
4. Later changes, one file at a time:
   - new file copied into the folder → `POST /documents`
   - file edited → `PUT /documents`
   - file removed → `DELETE /documents?file_path=...`
5. Ask questions with `POST /query`.

---

## Command-line Q&A (main.py)

```bash
python main.py
```
Starts an interactive question loop in the terminal. It **does not load documents**; run `POST /ingest/initial` first. Type `exit` to stop.

---

## Data and security

- `data/` is in `.gitignore`: documents, the database and any local state are **never committed**. Verify before every push:
  ```bash
  git ls-files data .env      # must print nothing
  ```
- `.env` holds the Gemini API key and is git-ignored. Do not paste keys into code or commits.
- Do not put client or company names, or real document names, in code, commit messages or PR descriptions.
- Asking a question sends the retrieved chunk text to Gemini. Loading, chunking and embedding stay on your machine.

### Upgrading from an older version

Databases built before `doc_id` was introduced have chunks without it. They cannot be updated or deleted per file, and loading again would duplicate them. Delete `data/chroma_db` once and run `POST /ingest/initial`.

---

## Known limitations

| Limitation | Effect |
|---|---|
| Scanned / image-only PDFs | No OCR yet. Reported as `failed: No text found`. |
| `.xlsx`, `.pptx`, `.htm`, `.txt` | Not supported. Listed under `skipped_unsupported`. |
| Legacy `.doc` | Disabled (needs LibreOffice). Re-enable by adding `".doc"` back to `SUPPORTED_EXTENSIONS` in `ingestion/document_loader.py`. |
| Initial load is synchronous | The request runs until every file is embedded. Fine for a POC; large corpora (GBs) need a background job or scheduler with a status endpoint. |
| Initial load does not detect edits or deletions | Use `PUT` / `DELETE` for changed or removed files. |
| No relevance cutoff | `/query` always returns `top_k` sources, even when the answer is "not enough information". Treat `sources` as *retrieved*, not necessarily *used*. |
| Same content in two formats | A document stored as both `.pdf` and `.docx` can return the same passage twice. |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `uvicorn` / `uv` is not recognized | Use `python -m uvicorn ...` / `python -m uv ...` |
| `Activate.ps1` cannot be loaded | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again |
| `ModuleNotFoundError` | The venv is not active, or packages were installed elsewhere. Activate `.venv` and reinstall from `requirements.txt`. |
| `422 JSON decode error` in `/docs` | Backslashes in a path. Use `/` or `\\`. |
| `404 Not Found` at `http://127.0.0.1:8000/` | Expected. Open `/docs`. |
| `405 Method Not Allowed` | Endpoint opened in the browser address bar (sends GET). Use `/docs`. |
| Port 8000 already in use | Stop the other server (Ctrl+C), or add `--port 8001` |
| Cannot delete `data/chroma_db` on Windows ("used by another process") | Stop the server and `main.py`, then `Get-Process python* \| Stop-Process -Force` and delete again |
| `GOOGLE_API_KEY is missing` | Add it to `.env` in the project root and restart the server |
| "unauthenticated requests to the HF Hub" warning | Harmless. Optionally set `HF_TOKEN` for faster model downloads. |
| ChromaDB distance-metric error | The collection was created with a different metric. Back up and delete `data/chroma_db`, then run the initial load. |

---

## License

MIT
