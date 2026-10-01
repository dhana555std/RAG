
"""
Vector store management using ChromaDB.

Responsibilities:
1. Initialize a persistent ChromaDB vector store.
2. Configure cosine distance as the collection's distance metric.
3. Store document chunks with stable IDs.
4. Validate that an existing collection uses cosine distance.

Important:
ChromaDB's distance metric is fixed when a collection is created.
An existing collection created with L2 distance must be deleted and
re-created before cosine distance can be used.
"""

import hashlib
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ---------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

CHROMA_DIR = Path(
    os.getenv("CHROMA_DIR", "data/chroma_db")
)

if not CHROMA_DIR.is_absolute():
    CHROMA_DIR = PROJECT_ROOT / CHROMA_DIR

COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "rag_documents"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

DISTANCE_METRIC = "cosine"


# ---------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------

def get_embedding_function() -> HuggingFaceEmbeddings:
    """
    Initialize the HuggingFace embedding model.

    The same model and configuration must be used during ingestion
    and retrieval.
    """

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )


# ---------------------------------------------------------------------
# Vector store initialization
# ---------------------------------------------------------------------

def get_vector_store() -> Chroma:
    """
    Initialize and return the persistent ChromaDB vector store.

    New collections are created with cosine distance.

    Existing collections are checked to ensure that they already
    use cosine distance. An existing L2 collection is not modified
    automatically.
    """

    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    embedding_function = get_embedding_function()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={
            "hnsw:space": DISTANCE_METRIC
        }
    )

    # Validate the actual collection configuration.
    collection_metadata = (
        vector_store._collection.metadata or {}
    )

    actual_metric = collection_metadata.get("hnsw:space")

    if actual_metric != DISTANCE_METRIC:
        raise ValueError(
            f"ChromaDB collection '{COLLECTION_NAME}' "
            f"uses '{actual_metric or 'default (L2)'}' distance. "
            f"Expected '{DISTANCE_METRIC}'. "
            "Delete the existing collection and re-run ingestion "
            "to create it with cosine distance."
        )

    return vector_store


# ---------------------------------------------------------------------
# Document storage
# ---------------------------------------------------------------------

def generate_document_id(document: Document) -> str:
    """
    Generate a stable SHA256 ID for a document chunk.

    The ID is based on the source, chunk position and content.
    """

    source = document.metadata.get("source", "")

    start_index = document.metadata.get(
        "start_index",
        ""
    )

    unique_content = (
        f"{source}|{start_index}|{document.page_content}"
    )

    return hashlib.sha256(
        unique_content.encode("utf-8")
    ).hexdigest()


def store_chunks(
    chunks: List[Document]
) -> Optional[Chroma]:
    """
    Store document chunks and their embeddings in ChromaDB.

    Args:
        chunks: List of LangChain Document objects.

    Returns:
        Initialized Chroma vector store, or None if no chunks
        were supplied.
    """

    if not chunks:
        print("No document chunks available for storage.")
        return None

    vector_store = get_vector_store()

    document_ids = [
        generate_document_id(chunk)
        for chunk in chunks
    ]

    vector_store.add_documents(
        documents=chunks,
        ids=document_ids
    )

    print(
        f"Successfully stored {len(chunks)} chunks "
        f"in ChromaDB collection '{COLLECTION_NAME}'."
    )

    return vector_store


# ---------------------------------------------------------------------
# Vector store information
# ---------------------------------------------------------------------

def get_collection_count() -> int:
    """
    Return the number of records currently stored in the collection.
    """

    vector_store = get_vector_store()

    return vector_store._collection.count()

