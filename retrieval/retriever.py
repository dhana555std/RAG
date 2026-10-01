
"""
Document retrieval and reranking.

Retrieval pipeline:
1. Accept a user query.
2. Retrieve candidate chunks using ChromaDB cosine distance.
3. Rerank candidates using a Cross-Encoder.
4. Return the final Top-K documents.

No LLM-based query processing or generation is performed.
"""

import os
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document

from ingestion.vector_store import get_vector_store
from retrieval.reranker import rerank_documents


# ---------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

DEFAULT_TOP_K = int(
    os.getenv("TOP_K", "5")
)

DEFAULT_CANDIDATE_K = int(
    os.getenv("RERANK_CANDIDATES", "20")
)


# ---------------------------------------------------------------------
# Query processing
# ---------------------------------------------------------------------

def prepare_query(query: str) -> str:
    """
    Validate and normalize the user query.

    Performs basic whitespace cleanup only.
    """

    if not isinstance(query, str):
        raise TypeError(
            "Query must be a string."
        )

    processed_query = query.strip()

    if not processed_query:
        raise ValueError(
            "Query cannot be empty."
        )

    return processed_query


# ---------------------------------------------------------------------
# Document retrieval and reranking
# ---------------------------------------------------------------------

def retrieve_documents(
    query: str,
    top_k: Optional[int] = None,
    candidate_top_k: Optional[int] = None
) -> List[Document]:
    """
    Retrieve and rerank relevant document chunks.

    Stage 1:
        Retrieve candidate documents using ChromaDB cosine distance.

    Stage 2:
        Rerank candidate documents using a Cross-Encoder.

    Args:
        query: User's search query.
        top_k: Number of final documents to return.
        candidate_top_k: Number of initial candidates to retrieve
                         before reranking.

    Returns:
        Final Top-K documents sorted by reranker relevance score.
    """

    processed_query = prepare_query(query)

    final_count = (
        DEFAULT_TOP_K if top_k is None else top_k
    )

    candidate_count = (
        DEFAULT_CANDIDATE_K
        if candidate_top_k is None
        else candidate_top_k
    )

    if not isinstance(final_count, int) or isinstance(final_count, bool):
        raise TypeError("top_k must be an integer.")

    if not isinstance(candidate_count, int) or isinstance(candidate_count, bool):
        raise TypeError("candidate_top_k must be an integer.")

    if final_count <= 0:
        raise ValueError("top_k must be greater than zero.")

    if candidate_count <= 0:
        raise ValueError(
            "candidate_top_k must be greater than zero."
        )

    if candidate_count < final_count:
        raise ValueError(
            "candidate_top_k must be greater than or equal to top_k."
        )

    vector_store = get_vector_store()

    # -------------------------------------------------------------
    # Stage 1: Initial vector retrieval using cosine distance.
    # -------------------------------------------------------------

    search_results = vector_store.similarity_search_with_score(
        query=processed_query,
        k=candidate_count
    )

    candidate_documents = []

    for rank, (document, cosine_distance) in enumerate(
        search_results,
        start=1
    ):

        cosine_distance = float(cosine_distance)

        # ChromaDB cosine distance = 1 - cosine similarity.
        cosine_similarity = 1.0 - cosine_distance

        metadata = document.metadata.copy()

        metadata.update({
            "retrieval_rank": rank,
            "cosine_distance": cosine_distance,
            "cosine_similarity": cosine_similarity
        })

        candidate_document = Document(
            page_content=document.page_content,
            metadata=metadata
        )

        candidate_documents.append(
            candidate_document
        )

    if not candidate_documents:
        return []

    # -------------------------------------------------------------
    # Stage 2: Cross-Encoder reranking.
    # -------------------------------------------------------------

    reranked_documents = rerank_documents(
        query=processed_query,
        documents=candidate_documents
    )

    # Return only the final requested number of documents.
    return reranked_documents[:final_count]


# ---------------------------------------------------------------------
# Display retrieved and reranked documents
# ---------------------------------------------------------------------

def display_retrieved_documents(
    documents: List[Document]
) -> None:
    """
    Display the final reranked documents and their scores.
    """

    if not documents:
        print("\nNo relevant documents found.")
        return

    print("\n" + "=" * 80)
    print("FINAL RERANKED DOCUMENTS")
    print("=" * 80)

    for document in documents:

        metadata = document.metadata

        print(
            f"\nFinal Rank       : "
            f"{metadata.get('rerank_rank', 'N/A')}"
        )

        print(
            f"Initial Rank     : "
            f"{metadata.get('retrieval_rank', 'N/A')}"
        )

        print(
            f"Source           : "
            f"{metadata.get('source', 'Unknown')}"
        )

        print(
            f"Page Number      : "
            f"{metadata.get('page_number', 'N/A')}"
        )

        print(
            f"Cosine Distance  : "
            f"{metadata.get('cosine_distance', 'N/A')}"
        )

        print(
            f"Cosine Similarity: "
            f"{metadata.get('cosine_similarity', 'N/A')}"
        )

        print(
            f"Reranker Score   : "
            f"{metadata.get('reranker_score', 'N/A')}"
        )

        print("\nContent:")
        print("-" * 80)
        print(document.page_content)
        print("-" * 80)
