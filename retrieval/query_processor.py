
"""
Query Processor
===============

Prepares a user-provided question before it is passed to the
retrieval component.

Responsibilities:
    1. Validate that the query is a non-empty string.
    2. Remove leading and trailing whitespace.
    3. Return a clean query for embedding and vector search.

Note:
    This module performs basic text normalization only.
    It does not use an LLM for query rewriting or expansion.
"""


def prepare_query(query: str) -> str:
    """
    Validate and normalize a user query.

    Args:
        query (str): The question submitted by the user.

    Returns:
        str: A normalized query string.

    Raises:
        TypeError: If the query is not a string.
        ValueError: If the query is empty or contains only whitespace.
    """

    # Ensure the input is a string.
    if not isinstance(query, str):
        raise TypeError("Query must be a string.")

    # Remove leading and trailing whitespace.
    normalized_query = query.strip()

    # Reject empty queries.
    if not normalized_query:
        raise ValueError("Query cannot be empty.")

    return normalized_query
