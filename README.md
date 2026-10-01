# RAG Application

A Retrieval-Augmented Generation (RAG) application built with Python, LangChain, ChromaDB, local Hugging Face embeddings, Cross-Encoder reranking, and Google Gemini.

The application ingests documents, splits them into token-aware chunks, creates embeddings, stores the chunks in persistent ChromaDB, retrieves relevant chunks using cosine similarity, reranks the candidates, and generates an answer grounded in the retrieved context.

## Features

- Recursively loads PDF (`.pdf`), Word (`.docx`), and legacy Word (`.doc`) documents.
- Splits content using `tiktoken`-based token counting.
- Generates embeddings locally with Sentence Transformers.
- Stores embeddings and metadata in persistent ChromaDB.
- Uses cosine distance for vector retrieval.
- Reranks retrieved candidates with a Sentence Transformers Cross-Encoder.
- Generates context-grounded answers using Google Gemini.
- Includes source labels in generated answers.

## Project Structure

```text
rag-application/
├── .env
├── .gitignore
├── README.md
├── requirements.txt
├── main.py
├── data/
│   ├── source/                 # Source documents to ingest
│   └── chroma_db/              # Persistent ChromaDB data (generated)
├── ingestion/
│   ├── __init__.py
│   ├── document_loader.py
│   ├── text_chunker.py
│   ├── embedding.py
│   └── vector_store.py
├── retrieval/
│   ├── __init__.py
│   ├── query_processor.py
│   ├── retriever.py
│   └── reranker.py
└── generation/
    ├── __init__.py
    └── generator.py
```

## Requirements

- Python 3.10 or later
- LibreOffice installed if you need to load legacy `.doc` files
- A Google Gemini API key

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

If PDF parsing reports that `fontTools` is required, install it and add it to `requirements.txt`:

```bash
python -m pip install fonttools
```

## Configuration

Create a `.env` file in the project root. Do not commit this file.

```env
# Google Gemini
GOOGLE_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
GEMINI_TEMPERATURE=0

# Document ingestion
SOURCE_DIR=data/source
CHUNK_SIZE=250
CHUNK_OVERLAP=50

# ChromaDB and embeddings
CHROMA_DIR=data/chroma_db
COLLECTION_NAME=rag_documents
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Retrieval and reranking
RERANK_CANDIDATES=20
TOP_K=5
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_DEVICE=cpu

# Optional: use this if soffice is not available on PATH
# LIBREOFFICE_PATH=/Applications/LibreOffice.app/Contents/MacOS/soffice
```

Use the exact environment variable names expected by your implementation. Remove settings that your code does not read.

## Add Documents

Place source files under the configured `SOURCE_DIR`, for example:

```text
data/source/
├── handbook.pdf
├── policy.docx
└── legacy-document.doc
```

The loader scans the directory recursively and processes supported file types.

## Run the Application

From the project root:

```bash
python main.py
```

The current application flow:

1. Loads supported source documents.
2. Splits extracted content into chunks.
3. Generates local embeddings.
4. Stores chunks in the persistent ChromaDB collection.
5. Accepts a question from the command line.
6. Retrieves candidate chunks using cosine similarity.
7. Reranks candidates with the Cross-Encoder.
8. Sends the top-ranked context to Gemini and prints the answer.

Enter `exit` or `quit` at the question prompt to stop the interactive loop.

**Note:** The current `main.py` flow runs ingestion when the application starts, so documents are reprocessed at startup.

## Retrieval and Reranking

- `RERANK_CANDIDATES`: Number of vector-search candidates passed to the reranker (example: `20`).
- `TOP_K`: Number of reranked documents retained for generation (example: `5`).
- `RERANKER_MODEL`: Cross-Encoder model used to score query/document pairs.
- `RERANKER_DEVICE`: Device used by the Cross-Encoder, such as `cpu`.

ChromaDB cosine distance and the Cross-Encoder reranker score are different measures. The reranker score is model-specific; do not interpret it as a probability or percentage.

## Important Notes

- Use the same embedding model for ingestion and querying.
- ChromaDB's distance metric is configured when the collection is created. If an existing collection uses a different metric, recreate it and re-ingest the documents to use cosine.
- Back up `data/chroma_db/` before deleting or recreating the collection.
- Keep API keys in `.env`; never hard-code secrets.
- `[S1]`, `[S2]`, etc. identify retrieved chunks supplied to the generator. Check the corresponding source content when using answers in critical workflows.

## Troubleshooting

### LibreOffice cannot be found

Install LibreOffice and either make `soffice` available on your `PATH` or set `LIBREOFFICE_PATH` in `.env`.

### Gemini API key is missing

Check that `.env` exists in the project root and that `GOOGLE_API_KEY` contains a valid key.

### ChromaDB uses the wrong distance metric

Back up the persisted database, recreate the collection or persistence directory as appropriate, and run ingestion again.

### Model download fails

Check network access on the first run. Hugging Face models are downloaded and cached locally.

## License

MIT
