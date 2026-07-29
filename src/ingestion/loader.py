from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Document:
    """
    Represents a loaded document.
    """

    source: str
    text: str


def load_text_file(file_path: Path) -> Document:
    """
    Load a .txt file.
    """

    text = file_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    return Document(
        source=file_path.name,
        text=text
    )


def load_pdf_file(file_path: Path) -> Document:
    """
    Load a PDF file.
    """

    reader = PdfReader(str(file_path))

    pages = []

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            pages.append(page_text)

    return Document(
        source=file_path.name,
        text="\n".join(pages)
    )


def load_documents(data_dir: Path) -> list[Document]:
    """
    Load all supported documents from a directory.
    """

    documents = []

    for file_path in data_dir.iterdir():

        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        if suffix == ".txt":
            documents.append(
                load_text_file(file_path)
            )

        elif suffix == ".pdf":
            documents.append(
                load_pdf_file(file_path)
            )

    return documents