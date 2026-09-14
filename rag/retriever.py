"""
Phase B (suite) - Recherche hybride + reranking.

Pourquoi ce fichier existe : sur un document dense et tres repetitif (ex.
des dizaines de fiches biographiques de dirigeants qui partagent la meme
structure "MAIN POSITION HELD / CURRENT OFFICES AND POSITIONS..."), la
seule similarite vectorielle confond facilement deux passages proches
lexicalement (le CEO et le President du Conseil, par exemple) parce que
le texte generique autour (comites, dates, filiales) domine numeriquement
le chunk et noie le signal utile (le nom + le titre exact).

Deux ameliorations la-dessus :
  1. Recherche hybride : on combine la recherche vectorielle (FAISS) avec
     une recherche par mots-cles (BM25). BM25 est tres bon pour retrouver
     un terme exact ("Chairman of the Board of Directors", un nom propre)
     que l'embedding peut diluer.
  2. Reranking : les candidats des deux recherches sont relus un par un par
     un cross-encoder, qui note (question, passage) ensemble - beaucoup
     plus precis qu'une comparaison de vecteurs precalcules, mais plus
     couteux, d'ou son usage seulement sur les ~20 meilleurs candidats et
     pas sur toute la base.
"""

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from rag.index import load_chunks, load_index

FETCH_K = 20   # candidats remontes par la recherche hybride avant reranking
FINAL_K = 6    # chunks gardes apres reranking, envoyes au LLM

# Cross-encoder multilingue (mMARCO) : comprend une question en francais
# posee sur un document en anglais. Plus lourd a charger que les
# embeddings mais ne tourne que sur FETCH_K passages par question, pas sur
# toute la base.
RERANKER_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

_cross_encoder: CrossEncoder | None = None
_bm25_retriever: BM25Retriever | None = None


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(RERANKER_MODEL_NAME)
    return _cross_encoder


def _get_bm25_retriever() -> BM25Retriever:
    """Construit le retriever BM25 une seule fois (garde en memoire pour
    les questions suivantes) - reconstruire l'index BM25 a chaque question
    serait inutilement lent."""
    global _bm25_retriever
    if _bm25_retriever is None:
        chunks = load_chunks()
        _bm25_retriever = BM25Retriever.from_documents(chunks)
        _bm25_retriever.k = FETCH_K
    return _bm25_retriever


def retrieve(question: str, final_k: int = FINAL_K) -> list[Document]:
    """Recherche hybride (BM25 + FAISS) puis reranking cross-encoder.
    Retourne les final_k chunks les plus pertinents pour la question."""
    vectorstore = load_index()
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": FETCH_K})
    bm25_retriever = _get_bm25_retriever()

    # weights : un peu plus de poids au vectoriel (comprend le sens), le
    # mot-cle sert surtout a rattraper les noms propres/termes exacts.
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6],
    )
    candidates = ensemble.invoke(question)
    if not candidates:
        return []

    encoder = _get_cross_encoder()
    pairs = [(question, doc.page_content) for doc in candidates]
    scores = encoder.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [doc for doc, _score in ranked[:final_k]]
