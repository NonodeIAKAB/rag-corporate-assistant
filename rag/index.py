"""
Phase B - Indexation vectorielle
Construit (ou recharge) un index FAISS local a partir des chunks.

Embeddings : modele local et gratuit (aucune cle API requise pour cette
etape), multilingue -> fonctionne bien sur des documents en francais.
"""

import pickle
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "index"
CHUNKS_PATH_NAME = "chunks.pkl"

# Modele leger (~470 Mo), tourne bien sur CPU. Pour plus de qualite (et si
# ta machine suit), remplace par "BAAI/bge-m3" (plus lourd, plus precis).
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )


def build_index(chunks: list[Document], index_dir: Path = INDEX_DIR) -> FAISS:
    """Embed tous les chunks et sauvegarde l'index FAISS sur disque.
    Sauvegarde aussi les chunks bruts (chunks.pkl) : necessaires pour
    reconstruire le retriever BM25 (recherche par mots-cles) sans avoir a
    re-parser les documents a chaque lancement de l'app.
    A relancer chaque fois que le contenu de data/raw/ change."""
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    with open(index_dir / CHUNKS_PATH_NAME, "wb") as f:
        pickle.dump(chunks, f)
    print(f"[index] {len(chunks)} chunks indexes -> {index_dir}")
    return vectorstore


def load_chunks(index_dir: Path = INDEX_DIR) -> list[Document]:
    """Recharge les chunks bruts sauvegardes par build_index() - utilise
    par le retriever BM25 (rag/retriever.py)."""
    chunks_path = index_dir / CHUNKS_PATH_NAME
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"{chunks_path} introuvable. Reconstruis l'index : "
            "python -m rag.ingest_and_index"
        )
    with open(chunks_path, "rb") as f:
        return pickle.load(f)


def load_index(index_dir: Path = INDEX_DIR) -> FAISS:
    """Recharge un index deja construit (rapide, pas de re-embedding)."""
    if not (index_dir / "index.faiss").exists():
        raise FileNotFoundError(
            f"Aucun index trouve dans {index_dir}. "
            "Lance d'abord : python -m rag.ingest_and_index"
        )
    embeddings = get_embeddings()
    return FAISS.load_local(
        str(index_dir), embeddings, allow_dangerous_deserialization=True
    )


def index_exists(index_dir: Path = INDEX_DIR) -> bool:
    return (index_dir / "index.faiss").exists()
