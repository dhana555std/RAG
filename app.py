"""
REST API for the RAG application (Task 1: Initial Load).

Start the server:   uvicorn app:app --reload
Open in browser:    http://127.0.0.1:8000/docs
"""

from dotenv import load_dotenv

# Read .env BEFORE importing our modules, because they read settings at import time.
load_dotenv()

from fastapi import FastAPI, HTTPException

from ingestion.initial_load import initial_load
from ingestion.vector_store import get_ingested_doc_ids

app = FastAPI(title="RAG API")


@app.post("/ingest/initial")
def ingest_initial():
    """Load every supported file in SOURCE_DIR. Files already loaded are skipped."""
    try:
        return initial_load()
    except FileNotFoundError as error:
        # The docs folder in .env doesn't exist -> tell the caller clearly (HTTP 400).
        raise HTTPException(status_code=400, detail=str(error))


@app.get("/documents")
def list_documents():
    """What's in the database right now: {file name: number of chunks}."""
    return get_ingested_doc_ids()
