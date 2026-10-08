"""
Tasks 2-4: add, update and delete ONE document.

All three take the file's path, either:
  - relative to SOURCE_DIR, e.g. "folder/file.pdf", or
  - a full path inside SOURCE_DIR, e.g. "C:/Users/me/RAG/data/source/folder/file.pdf"

The file must live inside SOURCE_DIR, so its doc_id is the same one the
initial load uses. (Later, an S3 event will give us the file's key the same way.)
"""

from pathlib import Path

from ingestion.document_loader import (
    SOURCE_DIR,
    SUPPORTED_EXTENSIONS,
    load_file,
    to_doc_id,
)
from ingestion.text_chunker import chunk_documents
from ingestion.vector_store import (
    generate_document_id,
    get_vector_store,
    store_chunks,
)


class NotIngestedError(LookupError):
    """The file has no chunks in ChromaDB."""


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _resolve(file_path: str) -> Path:
    """Turn the caller's path into a full path and make sure it's inside SOURCE_DIR."""
    cleaned = (file_path or "").strip().strip('"')
    if not cleaned:
        raise ValueError("file_path is empty.")

    path = Path(cleaned)
    if not path.is_absolute():
        path = SOURCE_DIR / path
    path = path.resolve()

    try:
        path.relative_to(SOURCE_DIR.resolve())
    except ValueError:
        raise ValueError(
            f"File must be inside the docs folder: {SOURCE_DIR}"
        ) from None

    return path


def _check_type(path: Path) -> None:
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type '{path.suffix}'. Allowed: {allowed}")


def _existing_chunk_ids(doc_id: str) -> list[str]:
    """IDs of every chunk that belongs to this file."""
    return get_vector_store().get(where={"doc_id": doc_id}, include=[])["ids"]


def _load_and_chunk(path: Path) -> list:
    try:
        pages = load_file(path)
    except Exception as error:
        # Corrupt, password-protected or unreadable file -> report it as bad input (400).
        raise ValueError(f"Could not read '{path.name}': {type(error).__name__}: {error}") from error
    if not pages:
        raise ValueError("No text found (empty or scanned file).")
    chunks = chunk_documents(pages)
    if not chunks:
        raise ValueError("Text was found but produced zero chunks.")
    return chunks


# ---------------------------------------------------------------------
# Task 2: add
# ---------------------------------------------------------------------

def add_document(file_path: str) -> dict:
    path = _resolve(file_path)
    _check_type(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    doc_id = to_doc_id(path)
    if _existing_chunk_ids(doc_id):
        raise FileExistsError(
            f"'{doc_id}' is already ingested. Use update to replace it."
        )

    chunks = _load_and_chunk(path)
    store_chunks(chunks)

    return {"doc_id": doc_id, "chunks": len(chunks)}


# ---------------------------------------------------------------------
# Task 3: update
# ---------------------------------------------------------------------

def update_document(file_path: str) -> dict:
    path = _resolve(file_path)
    _check_type(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    doc_id = to_doc_id(path)
    old_ids = set(_existing_chunk_ids(doc_id))
    if not old_ids:
        raise NotIngestedError(f"'{doc_id}' is not ingested yet. Use add.")

    # Read the new version FIRST. If it fails, the old chunks stay untouched.
    chunks = _load_and_chunk(path)

    # Same IDs ('<doc_id>::0', '::1', ...) are overwritten with the new text.
    store_chunks(chunks)

    # If the new version is shorter, remove the old chunks it no longer has.
    new_ids = {generate_document_id(chunk) for chunk in chunks}
    leftover = sorted(old_ids - new_ids)
    if leftover:
        get_vector_store().delete(ids=leftover)

    return {
        "doc_id": doc_id,
        "chunks_before": len(old_ids),
        "chunks_after": len(chunks),
        "old_chunks_removed": len(leftover),
    }


# ---------------------------------------------------------------------
# Task 4: delete
# ---------------------------------------------------------------------

def delete_document(file_path: str) -> dict:
    # The file may already be gone from the folder, so we don't require it to exist.
    path = _resolve(file_path)
    doc_id = to_doc_id(path)

    ids = _existing_chunk_ids(doc_id)
    if not ids:
        raise NotIngestedError(f"'{doc_id}' is not in the database.")

    get_vector_store().delete(ids=ids)

    return {
        "doc_id": doc_id,
        "chunks_deleted": len(ids),
        # If the file is still in the folder, the next initial load will add it back.
        "file_still_in_folder": path.exists(),
    }
