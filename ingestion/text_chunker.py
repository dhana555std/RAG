
"""
Module: text_chunker.py

Purpose:
    Split loaded documents into smaller text chunks before
    generating embeddings.

Responsibilities:
    1. Read chunk configuration from .env.
    2. Initialize the tiktoken tokenizer.
    3. Split LangChain Documents recursively.
    4. Maintain overlap between adjacent chunks.
    5. Preserve source document metadata.
    6. Add chunk-specific metadata.

Chunking strategy:
    RecursiveCharacterTextSplitter with tiktoken support.

Tokenizer:
    cl100k_base

Default configuration:
    CHUNK_SIZE = 250 tokens
    CHUNK_OVERLAP = 50 tokens

Character approximation:
    250 tokens is approximately 1000 characters for typical
    English text. The actual character count will vary.

Why overlap?
    Overlap helps retain contextual information when a
    sentence or related information spans two chunks.

Metadata added:
    - chunk_id
    - start_index
    - character_count
    - token_count

Input:
    list[Document] returned by document_loader.py

Output:
    list[Document] containing smaller chunks.
"""

from pathlib import Path

from dotenv import load_dotenv
import os

import tiktoken

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

# Chunk configuration is expressed in tokens.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 250))

CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", 50)
)

# Initialize the tokenizer.
ENCODING = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------
# TOKEN COUNTING
# ---------------------------------------------------------

def count_tokens(text: str) -> int:
    """
    Calculate the number of tokens in a text string.

    Args:
        text (str):
            Input text.

    Returns:
        int:
            Number of tokens according to cl100k_base.

    Note:
        This function is used for chunk metadata and
        validation. It does not perform text splitting.
    """

    return len(ENCODING.encode(text))


# ---------------------------------------------------------
# DOCUMENT CHUNKING
# ---------------------------------------------------------

def chunk_documents(documents):
    """
    Split loaded documents into overlapping chunks.

    Uses LangChain's RecursiveCharacterTextSplitter with
    tiktoken encoding support.

    Recursive splitting attempts to preserve natural text
    boundaries using the configured separator hierarchy.

    Separators are evaluated in this order:
        1. Paragraph boundaries.
        2. Line boundaries.
        3. Sentence boundaries.
        4. Word boundaries.
        5. Individual characters.

    Original metadata is inherited by every generated chunk.

    Args:
        documents (list[Document]):
            Documents returned by document_loader.py.

    Returns:
        list[Document]:
            Chunked documents with source and chunk metadata.

    Raises:
        ValueError:
            If chunk configuration is invalid.
    """

    if CHUNK_SIZE <= 0:
        raise ValueError("CHUNK_SIZE must be greater than zero.")

    if CHUNK_OVERLAP < 0:
        raise ValueError("CHUNK_OVERLAP cannot be negative.")

    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError(
            "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
        )

    if not documents:
        return []

    # Initialize the recursive splitter using tiktoken.
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",

        chunk_size=CHUNK_SIZE,

        chunk_overlap=CHUNK_OVERLAP,

        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ],

        add_start_index=True
    )

    # Split documents while retaining original metadata.
    chunks = splitter.split_documents(documents)

    # Add additional metadata to every chunk.
    for index, chunk in enumerate(chunks):

        chunk.metadata["chunk_id"] = index

        chunk.metadata["character_count"] = len(
            chunk.page_content
        )

        chunk.metadata["token_count"] = count_tokens(
            chunk.page_content
        )

    print("\n========== CHUNKING SUMMARY ==========")

    print(f"Original documents/pages : {len(documents)}")
    print(f"Generated chunks         : {len(chunks)}")
    print(f"Configured chunk size    : {CHUNK_SIZE} tokens")
    print(f"Configured overlap       : {CHUNK_OVERLAP} tokens")

    return chunks
