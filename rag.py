"""Minimal retrieval layer: load documents, chunk them, and rank chunks with TF-IDF."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass
class Chunk:
    source: str
    text: str


def read_file(name: str, data: bytes) -> str:
    """Extract text from a .txt, .md or .pdf file's bytes."""
    if name.lower().endswith(".pdf"):
        reader = PdfReader(BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return data.decode("utf-8", errors="ignore")


def load_folder(folder: Path) -> list[tuple[str, str]]:
    """Return (filename, text) for every supported file in a folder."""
    docs = []
    for path in sorted(folder.glob("**/*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            docs.append((path.name, read_file(path.name, path.read_bytes())))
    return docs


def chunk_text(source: str, text: str, size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Split text into overlapping character windows."""
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(Chunk(source, text[start : start + size]))
        start += size - overlap
    return chunks


class Retriever:
    def __init__(self, docs: list[tuple[str, str]]):
        self.chunks = [c for name, text in docs for c in chunk_text(name, text)]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = (
            self.vectorizer.fit_transform([c.text for c in self.chunks]) if self.chunks else None
        )

    def search(self, query: str, k: int = 4) -> list[Chunk]:
        if self.matrix is None:
            return []
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        ranked = scores.argsort()[::-1][:k]
        return [self.chunks[i] for i in ranked if scores[i] > 0]
