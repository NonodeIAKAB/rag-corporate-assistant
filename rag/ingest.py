"""
Phase A - Ingestion
Lit tous les fichiers de data/raw/ (PDF, DOCX, TXT), les decoupe en chunks
et renvoie une liste de langchain_core.documents.Document prets a etre
vectorises.

Formats geres : .pdf, .docx, .txt
Pour ajouter un format (.xlsx, .pptx...), ajoute une fonction load_xxx()
et branche-la dans EXTENSION_LOADERS.
"""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import docx

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Taille de chunk en caracteres (~ 150-200 tokens) et chevauchement.
# Documents tres structures (rapports, manuels) -> commence ici, ajuste
# ensuite en regardant les reponses obtenues.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def load_pdf(path: Path) -> list[Document]:
    """Un Document par page, avec le numero de page en metadata.
    C'est ce qui permet ensuite de citer 'page 34' dans les reponses."""
    reader = PdfReader(str(path))
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": path.name, "page": i + 1},
                )
            )
    return docs


def load_docx(path: Path) -> list[Document]:
    d = docx.Document(str(path))
    text = "\n".join(p.text for p in d.paragraphs if p.text.strip())
    if not text.strip():
        return []
    return [Document(page_content=text, metadata={"source": path.name, "page": None})]


def load_txt(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return []
    return [Document(page_content=text, metadata={"source": path.name, "page": None})]


EXTENSION_LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".txt": load_txt,
}


def load_raw_documents(raw_dir: Path = RAW_DATA_DIR) -> list[Document]:
    """Parcourt data/raw/ recursivement et charge tous les fichiers reconnus."""
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Dossier introuvable : {raw_dir}\n"
            "Decompresse ton zip dedans avant de lancer l'ingestion."
        )

    documents: list[Document] = []
    skipped: list[str] = []

    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue
        loader = EXTENSION_LOADERS.get(path.suffix.lower())
        if loader is None:
            skipped.append(path.name)
            continue
        documents.extend(loader(path))

    if skipped:
        print(f"[ingest] {len(skipped)} fichier(s) ignore(s) (format non gere) : {skipped}")
    if not documents:
        raise ValueError(
            f"Aucun document exploitable trouve dans {raw_dir}. "
            "Verifie que le zip est bien decompresse et contient des .pdf/.docx/.txt."
        )
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Decoupe chaque Document (page ou fichier) en chunks avec chevauchement.
    On garde les metadata (source, page) sur chaque chunk : c'est ce qui
    permet de citer precisement la source dans la reponse finale."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks


def run_ingestion(raw_dir: Path = RAW_DATA_DIR) -> list[Document]:
    raw_docs = load_raw_documents(raw_dir)
    chunks = split_documents(raw_docs)
    print(f"[ingest] {len(raw_docs)} page(s)/fichier(s) -> {len(chunks)} chunks")
    return chunks


if __name__ == "__main__":
    # Test rapide en ligne de commande : python -m rag.ingest
    chunks = run_ingestion()
    print("\nExemple de chunk :\n")
    print(chunks[0].metadata)
    print(chunks[0].page_content[:300])
