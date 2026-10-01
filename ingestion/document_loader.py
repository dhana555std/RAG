
"""
Module: document_loader.py

Purpose:
    Discover and load source documents from the configured SOURCE_DIR.

Supported formats:
    - PDF  (.pdf)
    - Word (.docx)
    - Word legacy format (.doc)

Responsibilities:
    1. Read SOURCE_DIR from the project .env file.
    2. Recursively discover supported files.
    3. Extract textual content from each file.
    4. Preserve document-level metadata.
    5. Return content as LangChain Document objects.

Metadata captured:
    - source: Absolute path of the source document.
    - file_name: Original filename.
    - file_type: File extension.
    - file_size_bytes: Original file size.
    - last_modified: Last modification timestamp.
    - page_number: PDF page number (PDF only).

Important:
    PDF files are loaded page by page to preserve page-level
    traceability for future retrieval and citation.

    DOCX files are loaded as individual documents.

    Legacy DOC files are converted to TXT using LibreOffice
    in headless mode.

    LibreOffice must be installed separately on the operating
    system. It is not a Python package.
"""

import os
import shutil
import subprocess
import tempfile

from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
from pypdf import PdfReader
import docx2txt

from langchain_core.documents import Document


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

# Resolve project root independently of the execution directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load environment variables from the project root.
load_dotenv(PROJECT_ROOT / ".env")

# Read the source directory from .env.
SOURCE_DIR = Path(
    os.getenv("SOURCE_DIR", "data/source")
)

# Convert relative paths into absolute paths.
if not SOURCE_DIR.is_absolute():
    SOURCE_DIR = PROJECT_ROOT / SOURCE_DIR

# Only these file formats will be processed.
SUPPORTED_EXTENSIONS = {
    ".doc",
    ".docx",
    ".pdf"
}

# LibreOffice executable configuration.
# The environment variable takes precedence.
LIBREOFFICE_PATH = os.getenv("LIBREOFFICE_PATH")

# ---------------------------------------------------------
# COMMON METADATA EXTRACTION
# ---------------------------------------------------------

def get_file_metadata(file_path: Path) -> dict:
    """
    Extract common metadata associated with a source file.

    Args:
        file_path (Path):
            Absolute path of the source document.

    Returns:
        dict:
            Metadata that will be attached to LangChain
            Document objects and eventually stored in ChromaDB.
    """

    stat = file_path.stat()

    return {
        "source": str(file_path.resolve()),
        "file_name": file_path.name,
        "file_type": file_path.suffix.lower().replace(".", ""),
        "file_size_bytes": stat.st_size,
        "last_modified": datetime.fromtimestamp(
            stat.st_mtime,
            tz=timezone.utc
        ).isoformat()
    }


# ---------------------------------------------------------
# PDF LOADER
# ---------------------------------------------------------

def load_pdf(file_path: Path) -> list[Document]:
    """
    Extract text from a PDF file page by page.

    Each PDF page becomes a separate LangChain Document.

    This approach preserves page-level metadata, which is
    useful when retrieving information and identifying
    the original source page.

    Args:
        file_path (Path):
            PDF file path.

    Returns:
        list[Document]:
            One LangChain Document per non-empty PDF page.
    """

    documents = []

    reader = PdfReader(str(file_path))

    # Extract common metadata once per source file.
    metadata = get_file_metadata(file_path)

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        # Extract text from the current PDF page.
        text = page.extract_text() or ""

        # Ignore empty pages.
        if not text.strip():
            continue

        # Add page-specific metadata.
        page_metadata = {
            **metadata,
            "page_number": page_number
        }

        documents.append(
            Document(
                page_content=text,
                metadata=page_metadata
            )
        )

    return documents


# ---------------------------------------------------------
# DOCX LOADER
# ---------------------------------------------------------

def load_docx(file_path: Path) -> list[Document]:
    """
    Extract text from a Microsoft Word DOCX document.

    The complete document is represented as one
    LangChain Document. It will be divided into smaller
    chunks later by the text_chunker module.

    Args:
        file_path (Path):
            DOCX file path.

    Returns:
        list[Document]:
            A list containing one Document if text exists.
    """

    text = docx2txt.process(str(file_path))

    if not text.strip():
        return []

    return [
        Document(
            page_content=text,
            metadata=get_file_metadata(file_path)
        )
    ]


# ---------------------------------------------------------
# LEGACY DOC LOADER
# ---------------------------------------------------------

def get_libreoffice_executable() -> str:
    """
    Locate the LibreOffice executable.

    Lookup order:
        1. LIBREOFFICE_PATH environment variable.
        2. Executable available in system PATH.
        3. Default macOS installation path.

    Returns:
        str:
            Absolute path to the LibreOffice executable.

    Raises:
        FileNotFoundError:
            If LibreOffice cannot be located.
    """

    # First priority: explicitly configured executable.
    if LIBREOFFICE_PATH:
        executable = Path(LIBREOFFICE_PATH).expanduser()

        if executable.is_file():
            return str(executable)

        raise FileNotFoundError(
            f"Configured LibreOffice executable not found: "
            f"{executable}"
        )

    # Second priority: search system PATH.
    executable = shutil.which("soffice")

    if executable:
        return executable

    # Third priority: default macOS installation path.
    macos_path = Path(
        "/Applications/LibreOffice.app/Contents/MacOS/soffice"
    )

    if macos_path.is_file():
        return str(macos_path)

    raise FileNotFoundError(
        "LibreOffice executable not found. "
        "Install LibreOffice or configure LIBREOFFICE_PATH "
        "in the project .env file."
    )


def load_doc(file_path: Path) -> list[Document]:
    """
    Extract text from legacy Microsoft Word DOC files.

    Uses LibreOffice in headless mode to convert DOC files
    into TXT files, then reads the extracted text.

    LibreOffice must be installed separately on the operating
    system. No additional Python package is required.

    Args:
        file_path (Path):
            Legacy DOC file path.

    Returns:
        list[Document]:
            A list containing one Document if text exists.
    """

    try:

        # Locate LibreOffice executable.
        soffice_path = get_libreoffice_executable()

        # Create a temporary directory for converted TXT files.
        with tempfile.TemporaryDirectory() as temp_dir:

            # Convert DOC to TXT using LibreOffice.
            result = subprocess.run(
                [
                    soffice_path,
                    "--headless",
                    "--convert-to",
                    "txt:Text",
                    "--outdir",
                    temp_dir,
                    str(file_path.resolve())
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=120
            )

            # LibreOffice generates a TXT file with the
            # same filename as the original DOC file.
            output_file = (
                Path(temp_dir) / f"{file_path.stem}.txt"
            )

            # Validate conversion output.
            if not output_file.exists():

                raise RuntimeError(
                    f"LibreOffice conversion failed for "
                    f"{file_path}. "
                    f"Output: {result.stdout} {result.stderr}"
                )

            # Read the extracted text.
            text = output_file.read_text(
                encoding="utf-8",
                errors="replace"
            )

    except FileNotFoundError as exc:

        raise RuntimeError(
            f"LibreOffice executable not found while "
            f"processing DOC file: {file_path}"
        ) from exc

    except subprocess.TimeoutExpired as exc:

        raise RuntimeError(
            f"LibreOffice conversion timed out for: {file_path}"
        ) from exc

    except subprocess.CalledProcessError as exc:

        raise RuntimeError(
            f"Unable to convert DOC file: {file_path}. "
            f"Error: {exc.stderr}"
        ) from exc

    if not text.strip():
        return []

    return [
        Document(
            page_content=text,
            metadata=get_file_metadata(file_path)
        )
    ]


# ---------------------------------------------------------
# DOCUMENT DISCOVERY AND LOADING
# ---------------------------------------------------------

def load_documents() -> list[Document]:
    """
    Discover and load all supported source documents.

    Processing workflow:
        1. Validate SOURCE_DIR.
        2. Recursively discover files.
        3. Filter by supported extensions.
        4. Select the appropriate document loader.
        5. Collect extracted LangChain Documents.

    Returns:
        list[Document]:
            Collection of loaded documents/pages with metadata.

    Raises:
        FileNotFoundError:
            If SOURCE_DIR does not exist.

        RuntimeError:
            If a supported document cannot be loaded.
    """

    if not SOURCE_DIR.exists():

        raise FileNotFoundError(
            f"Source directory does not exist: {SOURCE_DIR}"
        )

    # Discover supported files recursively.
    files = sorted(
        file_path
        for file_path in SOURCE_DIR.rglob("*")
        if file_path.is_file()
        and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    documents = []

    print(f"Source directory: {SOURCE_DIR}")
    print(f"Supported files found: {len(files)}")

    for file_path in files:

        extension = file_path.suffix.lower()

        print(f"Loading: {file_path.name}")

        try:

            if extension == ".pdf":

                loaded = load_pdf(file_path)

            elif extension == ".docx":

                loaded = load_docx(file_path)

            elif extension == ".doc":

                loaded = load_doc(file_path)

            else:

                continue

            documents.extend(loaded)

            print(
                f"  Extracted documents/pages: {len(loaded)}"
            )

        except Exception as exc:

            # Fail explicitly rather than silently skipping
            # documents that could not be processed.
            raise RuntimeError(
                f"Failed to load {file_path}: {exc}"
            ) from exc

    print(
        f"\nTotal documents/pages loaded: {len(documents)}"
    )

    return documents
