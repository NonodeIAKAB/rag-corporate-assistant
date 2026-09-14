"""
Interface Streamlit - Assistant documentaire RAG.

Lancement : streamlit run app.py
"""

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag.index import index_exists, INDEX_DIR
from rag.ingest import RAW_DATA_DIR
from rag.ingest_and_index import main as build_the_index
from rag.chain import answer_question

load_dotenv()

st.set_page_config(page_title="Assistant Documentaire RAG", page_icon="📄", layout="centered")

st.title("📄 Assistant documentaire")
st.caption(
    "Pose une question sur le document indexé. Les réponses sont générées "
    "uniquement à partir du contenu du document, avec citation des sources."
)

# --- Sidebar : etat de l'index -------------------------------------------
with st.sidebar:
    st.header("Index")
    raw_files = [p.name for p in RAW_DATA_DIR.glob("*") if p.is_file()] if RAW_DATA_DIR.exists() else []
    st.write(f"**{len(raw_files)}** fichier(s) dans `data/raw/`")
    if raw_files:
        with st.expander("Voir les fichiers"):
            for f in raw_files:
                st.write(f"- {f}")

    if index_exists():
        st.success("Index vectoriel prêt.")
    else:
        st.warning("Aucun index trouvé.")

    if st.button("🔄 (Re)construire l'index", use_container_width=True):
        if not raw_files:
            st.error("data/raw/ est vide — place tes fichiers dedans d'abord.")
        else:
            with st.spinner("Ingestion + embeddings en cours (peut prendre 1-2 min)..."):
                build_the_index()
            st.success("Index reconstruit.")
            st.rerun()

# --- Chat -------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources utilisées"):
                for s in msg["sources"]:
                    page = s.metadata.get("page")
                    ref = s.metadata.get("source", "inconnu") + (f" — page {page}" if page else "")
                    st.markdown(f"**{ref}**")
                    st.caption(s.page_content[:400] + "…")

question = st.chat_input("Ta question sur le document...")

if question:
    if not index_exists():
        st.error("Construis d'abord l'index (bouton dans la barre latérale).")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Recherche dans le document..."):
                try:
                    result = answer_question(question)
                    st.markdown(result["answer"])
                    with st.expander("Sources utilisées"):
                        for s in result["sources"]:
                            page = s.metadata.get("page")
                            ref = s.metadata.get("source", "inconnu") + (f" — page {page}" if page else "")
                            st.markdown(f"**{ref}**")
                            st.caption(s.page_content[:400] + "…")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
                    )
                except RuntimeError as e:
                    st.error(str(e))
