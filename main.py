"""
Main application entry point for the RAG application.

Application workflow:
1. Load source documents.
2. Split documents into chunks.
3. Generate embeddings and store chunks in ChromaDB.
4. Retrieve relevant chunks using cosine distance.
5. Rerank retrieved chunks using a Cross-Encoder.
6. Generate answers using Google Gemini.
7. Display the generated answer with source citations.

Note:
The application implements ingestion, retrieval, reranking
and LLM-based answer generation.
"""

from dotenv import load_dotenv


# ---------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------

# Load environment variables before importing application modules.
load_dotenv()


# ---------------------------------------------------------------------
# Application imports
# ---------------------------------------------------------------------

from ingestion.document_loader import load_documents
from ingestion.text_chunker import chunk_documents
from ingestion.vector_store import store_chunks

from retrieval import (
    retrieve_documents,
    display_retrieved_documents
)

from generation import generate_answer


# ---------------------------------------------------------------------
# Document ingestion
# ---------------------------------------------------------------------

def run_ingestion() -> bool:
    """
    Execute the complete document ingestion pipeline.

    Steps:
    1. Load source documents.
    2. Split documents into chunks.
    3. Generate embeddings and store chunks in ChromaDB.

    Returns:
        bool: True if ingestion completes successfully,
              otherwise False.
    """

    print("\n" + "=" * 80)
    print("RAG APPLICATION - DOCUMENT INGESTION")
    print("=" * 80)

    # Step 1: Load source documents.
    print("\n[STEP 1] Loading source documents...")

    documents = load_documents()

    if not documents:
        print("No supported documents found.")
        return False

    print(
        f"Successfully loaded {len(documents)} documents."
    )

    # Step 2: Split documents into chunks.
    print("\n[STEP 2] Splitting documents into chunks...")

    chunks = chunk_documents(documents)

    if not chunks:
        print("No document chunks generated.")
        return False

    print(
        f"Successfully generated {len(chunks)} chunks."
    )

    # Step 3: Generate embeddings and store chunks.
    print(
        "\n[STEP 3] Generating embeddings "
        "and storing in ChromaDB..."
    )

    vector_store = store_chunks(chunks)

    if vector_store is None:
        print("Failed to store document chunks.")
        return False

    print("\n" + "-" * 80)
    print("DOCUMENT INGESTION COMPLETED SUCCESSFULLY")
    print("-" * 80)

    print(f"Total Documents : {len(documents)}")
    print(f"Total Chunks    : {len(chunks)}")

    return True


# ---------------------------------------------------------------------
# Document retrieval, reranking and generation
# ---------------------------------------------------------------------

def run_rag_pipeline() -> None:
    """
    Execute the interactive RAG pipeline.

    Steps:
    1. Accept a user question.
    2. Retrieve relevant chunks using cosine similarity.
    3. Rerank retrieved chunks using a Cross-Encoder.
    4. Generate an answer using Gemini.
    5. Display the generated answer with citations.
    """

    print("\n" + "=" * 80)
    print("RAG APPLICATION - QUESTION ANSWERING")
    print("=" * 80)

    print("\nEnter your question to search the indexed documents.")
    print("Type 'exit' to stop the application.")

    while True:

        print("\n" + "-" * 80)

        query = input("\nEnter your question: ").strip()

        if query.lower() == "exit":
            print("\nRAG application stopped.")
            break

        if not query:
            print("Please enter a valid question.")
            continue

        try:

            # ---------------------------------------------------------
            # Step 1: Retrieval + Reranking
            # ---------------------------------------------------------

            print("\n[STEP 1] Retrieving and reranking documents...")

            retrieved_documents = retrieve_documents(
                query=query
            )

            if not retrieved_documents:
                print(
                    "\nNo relevant documents found. "
                    "Answer generation skipped."
                )
                continue

            # Display the final reranked documents.
            display_retrieved_documents(
                retrieved_documents
            )

            # ---------------------------------------------------------
            # Step 2: Answer Generation
            # ---------------------------------------------------------

            print("\n[STEP 2] Generating answer using Gemini...")

            answer = generate_answer(
                query=query,
                documents=retrieved_documents
            )

            # ---------------------------------------------------------
            # Step 3: Display Generated Answer
            # ---------------------------------------------------------

            print("\n" + "=" * 80)
            print("GENERATED ANSWER")
            print("=" * 80)

            print("\n" + answer)

            print("\n" + "=" * 80)

        except Exception as error:

            print(
                f"\nRAG pipeline failed: {error}"
            )


# ---------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------

def main() -> None:
    """
    Start interactive question answering.

    Documents are no longer loaded here. Use POST /ingest/initial (app.py).
    """

    run_rag_pipeline()


if __name__ == "__main__":
    main()