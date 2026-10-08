"""
REST API for the RAG application.

Start (development):  python -m uvicorn app:app --reload
Start (demo/server):  python -m uvicorn app:app
Open in browser:      http://127.0.0.1:8000/docs
"""

from dotenv import load_dotenv

# Read .env BEFORE importing our modules, because they read settings at import time.
load_dotenv()

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from generation import generate_answer
from ingestion.document_ops import (
    NotIngestedError,
    add_document,
    delete_document,
    update_document,
)
from ingestion.initial_load import initial_load
from ingestion.vector_store import get_ingested_doc_ids
from retrieval import retrieve_documents

app = FastAPI(title="RAG API")


# ---------------------------------------------------------------------
# Request shapes (Pydantic): FastAPI checks them and shows them on /docs
# ---------------------------------------------------------------------

class FileRequest(BaseModel):
    file_path: str = Field(
        ...,
        description="Path relative to the docs folder, or a full path inside it. Use / not \\.",
        examples=["folder/file.pdf"],
    )


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["What is this document about?"])
    top_k: int = Field(5, ge=1, le=20, description="How many chunks to use for the answer.")


def _to_http_error(error: Exception) -> HTTPException:
    """Map our Python errors to HTTP status codes."""
    if isinstance(error, (FileNotFoundError, NotIngestedError)):
        return HTTPException(status_code=404, detail=str(error))   # doesn't exist
    if isinstance(error, FileExistsError):
        return HTTPException(status_code=409, detail=str(error))   # already there
    return HTTPException(status_code=400, detail=str(error))       # bad input / unreadable file


# ---------------------------------------------------------------------
# Task 1: initial load + list
# ---------------------------------------------------------------------

@app.post("/ingest/initial")
def ingest_initial():
    """Load every supported file in SOURCE_DIR. Files already loaded are skipped."""
    try:
        return initial_load()
    except FileNotFoundError as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.get("/documents")
def list_documents():
    """What's in the database right now: {file name: number of chunks}."""
    return get_ingested_doc_ids()


# ---------------------------------------------------------------------
# Tasks 2-4: add / update / delete ONE file
# ---------------------------------------------------------------------

@app.post("/documents", status_code=201)
def add_document_endpoint(request: FileRequest):
    """Ingest ONE new file that is already in the docs folder."""
    try:
        return add_document(request.file_path)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        raise _to_http_error(error)


@app.put("/documents")
def update_document_endpoint(request: FileRequest):
    """Replace a file's chunks with its current content."""
    try:
        return update_document(request.file_path)
    except (ValueError, FileNotFoundError, NotIngestedError) as error:
        raise _to_http_error(error)


@app.delete("/documents")
def delete_document_endpoint(file_path: str = Query(..., examples=["folder/file.pdf"])):
    """Remove all chunks of ONE file from the database (the file itself is not touched)."""
    try:
        return delete_document(file_path)
    except (ValueError, NotIngestedError) as error:
        raise _to_http_error(error)


# ---------------------------------------------------------------------
# Task 5: ask a question -> answer (retrieve + rerank + generate)
# ---------------------------------------------------------------------

@app.post("/query")
def query_endpoint(request: QueryRequest):
    """Same as asking in main.py: retrieve top chunks, then generate an answer with sources."""
    try:
        documents = retrieve_documents(query=request.question, top_k=request.top_k)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    try:
        answer = generate_answer(query=request.question, documents=documents)
    except Exception as error:  # LLM call failed (API key, network, quota...)
        raise HTTPException(status_code=502, detail=f"Answer generation failed: {error}")

    sources = [
        {
            "source_id": f"S{number}",          # matches the [S1], [S2] citations in the answer
            "doc_id": doc.metadata.get("doc_id"),
            "page_number": doc.metadata.get("page_number"),
            "reranker_score": round(float(doc.metadata.get("reranker_score", 0.0)), 4),
        }
        for number, doc in enumerate(documents, start=1)
    ]

    return {"question": request.question, "answer": answer, "sources": sources}
