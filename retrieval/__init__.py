
"""
Retrieval Package
=================

Exposes the public retrieval functions.
"""

from retrieval.retriever import (
    retrieve_documents,
    display_retrieved_documents
)

from retrieval.query_processor import prepare_query


__all__ = [
    "retrieve_documents",
    "display_retrieved_documents",
    "prepare_query"
]

