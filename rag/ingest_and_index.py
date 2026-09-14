"""
Point d'entree unique pour (re)construire l'index a partir de data/raw/.

Usage :
    python -m rag.ingest_and_index
"""

from rag.ingest import run_ingestion
from rag.index import build_index


def main():
    chunks = run_ingestion()
    build_index(chunks)
    print("Index pret. Lance maintenant : streamlit run app.py")


if __name__ == "__main__":
    main()
