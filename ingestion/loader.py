"""Document loaders for PDF, DOCX, and TXT files."""

from pathlib import Path

from ingestion.models import Document, normalize_source


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def load_document(file_path: str | Path) -> Document:
    """Load a supported document and return extracted raw text."""
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type '{extension}'. Supported types: {supported}")

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    if extension == ".txt":
        text = _load_txt(path)
        metadata = {"page_count": None}
    elif extension == ".pdf":
        text, page_count = _load_pdf(path)
        metadata = {"page_count": page_count}
    else:
        text = _load_docx(path)
        metadata = {"page_count": None}

    return Document(
        text=text.strip(),
        source=normalize_source(path),
        file_type=extension.lstrip("."),
        metadata=metadata,
    )


def _load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_pdf(path: Path) -> tuple[str, int]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("Install pypdf to load PDF files: pip install pypdf") from exc

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages), len(reader.pages)


def _load_docx(path: Path) -> str:
    try:
        from docx import Document as DocxDocument
    except ImportError as exc:
        raise ImportError("Install python-docx to load DOCX files: pip install python-docx") from exc

    document = DocxDocument(str(path))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)
