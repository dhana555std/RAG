
"""
Document reranking using a HuggingFace Cross-Encoder.

Responsibilities:
1. Load a local Cross-Encoder model.
2. Accept a query and candidate document chunks.
3. Calculate relevance scores for each query-document pair.
4. Sort documents by their reranking scores.
5. Return documents in descending relevance order.

No external LLM API is required.
"""

import os
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


# ---------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

RERANKER_DEVICE = os.getenv(
    "RERANKER_DEVICE",
    "cpu"
)


# ---------------------------------------------------------------------
# Cross-Encoder initialization
# ---------------------------------------------------------------------

_cross_encoder = None


def get_reranker() -> CrossEncoder:
    """
    Initialize the Cross-Encoder model.

    The model is loaded only once and reused for subsequent queries.
    """

    global _cross_encoder

    if _cross_encoder is None:

        print(
            f"\nLoading reranker model: {RERANKER_MODEL}"
        )

        _cross_encoder = CrossEncoder(
            model_name_or_path=RERANKER_MODEL,
            device=RERANKER_DEVICE
        )

        print("Reranker model loaded successfully.")

    return _cross_encoder


# ---------------------------------------------------------------------
# Document reranking
# ---------------------------------------------------------------------

def rerank_documents(
    query: str,
    documents: List[Document]
) -> List[Document]:
    """
    Rerank retrieved documents using Cross-Encoder relevance scores.

    A Cross-Encoder evaluates the query and document together,
    rather than comparing independently generated embeddings.

    Args:
        query: Original user query.
        documents: Candidate documents returned by vector retrieval.

    Returns:
        Documents sorted by reranker score in descending order.
    """

    if not documents:
        return []

    reranker = get_reranker()

    # Create query-document pairs for Cross-Encoder evaluation.
    query_document_pairs = [
        (query, document.page_content)
        for document in documents
    ]

    # Predict relevance scores for all candidate documents.
    scores = reranker.predict(
        query_document_pairs,
        show_progress_bar=False
    )

    # Attach reranker scores and preserve original retrieval ranks.
    ranked_documents = []

    for document, score in zip(documents, scores):

        metadata = document.metadata.copy()

        metadata["reranker_score"] = float(score)

        ranked_document = Document(
            page_content=document.page_content,
            metadata=metadata
        )

        ranked_documents.append(ranked_document)

    # Sort by Cross-Encoder score, highest first.
    ranked_documents.sort(
        key=lambda doc: doc.metadata["reranker_score"],
        reverse=True
    )

    # Assign final reranking positions.
    for rank, document in enumerate(
        ranked_documents,
        start=1
    ):
        document.metadata["rerank_rank"] = rank

    return ranked_documents
