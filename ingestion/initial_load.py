"""
Task 1 - Initial Load.

Loads every supported file from SOURCE_DIR into ChromaDB, one file at a time.
Files already in ChromaDB are skipped, so running it twice adds nothing.
One broken file is reported and does not stop the others.
"""

from ingestion.document_loader import (
    SUPPORTED_EXTENSIONS,
    list_source_files,
    load_file,
    to_doc_id,
)
from ingestion.text_chunker import chunk_documents
from ingestion.vector_store import get_ingested_doc_ids, store_chunks


def initial_load() -> dict:
    files = list_source_files()
    already_loaded = get_ingested_doc_ids()

    report = {
        "total_files": len(files),
        "ingested": [],
        "skipped_already_loaded": [],
        "skipped_unsupported": [],
        "failed": [],
    }

    for file_path in files:
        doc_id = to_doc_id(file_path)

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            report["skipped_unsupported"].append(doc_id)
            continue

        if doc_id in already_loaded:
            report["skipped_already_loaded"].append(doc_id)
            continue

        try:
            pages = load_file(file_path)
            if not pages:
                raise ValueError("No text found (empty or scanned file)")
            chunks = chunk_documents(pages)
            store_chunks(chunks)
            report["ingested"].append({"doc_id": doc_id, "chunks": len(chunks)})
        except Exception as error:
            # One bad file must not stop the others.
            report["failed"].append({"doc_id": doc_id, "reason": str(error)})

    return report
