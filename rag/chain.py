"""
Phase C - Recherche + generation
Assemble : retriever FAISS (top-k) -> prompt strict -> LLM Groq.

Le prompt impose deux garde-fous :
  1. Ne repondre qu'a partir du contexte fourni (pas de connaissances generales).
  2. Dire explicitement "information absente du document" si ce n'est pas dedans.
C'est ce qui evite les reponses inventees ("hallucinations").
"""

import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from rag.retriever import retrieve

TOP_K = 6

# Le catalogue Groq change souvent (modeles ajoutes/retires). Configurable
# via .env pour ne pas avoir a toucher au code si un modele n'est plus
# disponible sur ton compte. Verifie la liste a jour sur ton compte ici :
# https://console.groq.com/docs/models (section "vos modeles disponibles").
# llama-3.1-8b-instant par defaut : petit, rapide, tres largement disponible.
# Pour plus de qualite, essaie llama-3.3-70b-versatile ou openai/gpt-oss-120b
# une fois que tu as confirme qu'il apparait dans TON compte.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = """Tu es l'assistant documentaire d'une entreprise industrielle.
Tu reponds UNIQUEMENT a partir des extraits de document fournis dans le contexte
ci-dessous. Tu ne dois jamais utiliser de connaissances generales ni inventer
une information absente du contexte.

Regles :
- Si la reponse n'est pas dans le contexte, dis clairement :
  "Je ne trouve pas cette information dans le document fourni."
- Cite systematiquement la source (nom de fichier, et page si disponible)
  entre parentheses a la fin de chaque affirmation.
- Reponds en francais, de maniere concise et factuelle.

Contexte :
{context}
"""
# {context} ci-dessus n'est PAS a remplir a la main : c'est une variable de
# template LangChain, injectee automatiquement a chaque question dans
# answer_question() (voir chain.invoke(...) plus bas), avec le texte des
# chunks retrouves par le retriever (voir format_context()). Le contexte
# change donc a chaque question posee - il ne peut pas etre fige ici.


def format_context(docs: list[Document]) -> str:
    parts = []
    for d in docs:
        page = d.metadata.get("page")
        ref = f"{d.metadata.get('source', 'inconnu')}" + (f", page {page}" if page else "")
        parts.append(f"[Source: {ref}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def get_llm() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY manquante. Cree un fichier .env a partir de "
            ".env.example et colle ta cle (https://console.groq.com/keys)."
        )
    return ChatGroq(model=GROQ_MODEL, temperature=0, api_key=api_key)


def answer_question(question: str, top_k: int = TOP_K) -> dict:
    """Retourne {"answer": str, "sources": list[Document]}"""
    # Recherche hybride (BM25 + FAISS) + reranking cross-encoder - voir
    # rag/retriever.py pour le detail et le pourquoi.
    retrieved_docs = retrieve(question, final_k=top_k)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )
    llm = get_llm()
    chain = prompt | llm

    # C'est ICI que {context} et {question} sont remplis dans le prompt -
    # format_context(retrieved_docs) devient le contenu de {context} defini
    # dans SYSTEM_PROMPT, pour CETTE question precise.
    response = chain.invoke(
        {"context": format_context(retrieved_docs), "question": question}
    )
    return {"answer": response.content, "sources": retrieved_docs}
