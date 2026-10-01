
"""
Module: embedding.py

Purpose:
    Initialize and provide the local embedding model used
    to convert document chunks into numerical vectors.

Embedding model:
    sentence-transformers/all-MiniLM-L6-v2

Model characteristics:
    - Open-source.
    - Free to use.
    - Runs locally.
    - Produces 384-dimensional embeddings.
    - Does not require an external embedding API.

Responsibilities:
    1. Read the embedding model name from .env.
    2. Initialize HuggingFaceEmbeddings.
    3. Configure the execution device.
    4. Normalize generated embeddings.
    5. Return the embedding object to vector_store.py.

Important:
    The model is generally downloaded from Hugging Face
    during its first execution and cached locally.

    Subsequent executions can reuse the cached model.

Input:
    Text supplied by LangChain's embedding interface.

Output:
    Numerical embedding vectors.

The same embedding model must be used during ingestion
and retrieval to maintain vector compatibility.
"""

from pathlib import Path

from dotenv import load_dotenv
import os

from langchain_huggingface import HuggingFaceEmbeddings


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

# HuggingFace embedding model.
MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

# CPU is the default execution device.
# CUDA can be configured when supported by the environment.
EMBEDDING_DEVICE = os.getenv(
    "EMBEDDING_DEVICE",
    "cpu"
)


# ---------------------------------------------------------
# EMBEDDING MODEL INITIALIZATION
# ---------------------------------------------------------

def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initialize the HuggingFace embedding model.

    Returns:
        HuggingFaceEmbeddings:
            LangChain-compatible embedding implementation.

    Configuration:
        model_name:
            HuggingFace model identifier.

        model_kwargs:
            Specifies the device used for inference.

        encode_kwargs:
            Normalizes embeddings to unit length.

    Usage:
        The returned object is supplied to ChromaDB
        so that document embeddings are generated locally.
    """

    print("\n========== EMBEDDING MODEL ==========")

    print(f"Model name : {MODEL_NAME}")
    print(f"Device     : {EMBEDDING_DEVICE}")

    embeddings = HuggingFaceEmbeddings(

        model_name=MODEL_NAME,

        model_kwargs={
            "device": EMBEDDING_DEVICE
        },

        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    print("Embedding model initialized successfully.")

    return embeddings
