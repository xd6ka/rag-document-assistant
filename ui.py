"""Streamlit front end: upload PDFs, index them, ask questions, inspect citations."""

from __future__ import annotations

import streamlit as st

from src import config
from src.generation import answer_question
from src.ingest import ingest_pdf
from src.retrieval import Retriever
from src.store import DocumentStore, RetrievedChunk

st.set_page_config(page_title="RAG Document Assistant", layout="wide", page_icon="📄")


@st.cache_resource(show_spinner=False)
def get_store() -> DocumentStore:
    return DocumentStore()


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    from src.embeddings import get_model

    return get_model()


def render_sources(contexts: list[RetrievedChunk]) -> None:
    with st.expander(f"Sources ({len(contexts)} chunks)"):
        for number, chunk in enumerate(contexts, start=1):
            st.markdown(
                f"**[{number}]** `{chunk.source}` · page {chunk.page} · "
                f"similarity {chunk.score:.3f}"
            )
            snippet = chunk.text if len(chunk.text) <= 600 else chunk.text[:600] + " ..."
            st.text(snippet)


store = get_store()

with st.spinner("Loading embedding model..."):
    load_embedding_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("RAG Document Assistant")
st.caption(
    f"Local embeddings ({config.EMBEDDING_MODEL}) · {config.CHUNK_SIZE}-char chunks "
    f"with {config.CHUNK_OVERLAP} overlap · MMR reranking · answers by {config.QWEN_MODEL}"
)

documents = store.documents()

with st.sidebar:
    st.header("Documents")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
    )

    if st.button("Process documents", type="primary", disabled=not uploaded_files):
        for uploaded in uploaded_files or []:
            with st.spinner(f"Indexing {uploaded.name}..."):
                try:
                    result = ingest_pdf(uploaded.name, uploaded, store=store)
                    if result["skipped"]:
                        st.warning(f"{uploaded.name}: no extractable text (scanned PDF?).")
                    else:
                        st.success(
                            f"{uploaded.name}: {result['chunks']} chunks "
                            f"from {result['pages']} pages."
                        )
                except Exception as exc:  # noqa: BLE001 - surface the reason in the UI
                    st.error(f"{uploaded.name}: {exc}")
        st.rerun()

    if documents:
        st.subheader("Indexed")
        for name, info in documents.items():
            row = st.columns([4, 1])
            row[0].markdown(f"**{name}**  \n{info['chunks']} chunks · {info['pages']} pages")
            if row[1].button("Delete", key=f"delete_{name}", help=f"Remove {name}"):
                store.delete_document(name)
                st.rerun()
    else:
        st.info("Nothing indexed yet. Upload a PDF above.")

    st.divider()
    st.header("Retrieval")

    selected_sources = st.multiselect(
        "Restrict to documents",
        options=list(documents.keys()),
        default=[],
        help="Leave empty to search everything in the collection.",
    )
    context_k = st.slider(
        "Chunks sent to the model",
        min_value=1,
        max_value=8,
        value=config.CONTEXT_K,
        help="More chunks means more context, larger prompt and slower answers.",
    )
    use_mmr = st.checkbox("MMR reranking", value=True, help="Diversify retrieved chunks.")

    st.divider()
    with st.expander("Advanced"):
        st.write(
            f"- embeddings: `{config.EMBEDDING_MODEL}`\n"
            f"- store: `{config.CHROMA_PATH}`\n"
            f"- collection: `{config.COLLECTION}`\n"
            f"- retrieval: top {config.RETRIEVE_K} → MMR λ={config.MMR_LAMBDA} → {config.CONTEXT_K}\n"
            f"- chunking: {config.CHUNK_SIZE} chars / {config.CHUNK_OVERLAP} overlap"
        )
        if st.button("Clear database"):
            store.clear()
            st.session_state.messages = []
            st.rerun()

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])

prompt = st.chat_input("Ask a question about your documents")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    contexts: list[RetrievedChunk] = []
    with st.chat_message("assistant"):
        if store.count() == 0:
            answer = "No documents indexed yet. Upload a PDF in the sidebar first."
            st.warning(answer)
        else:
            try:
                with st.spinner("Retrieving..."):
                    contexts = Retriever(store).retrieve(
                        prompt,
                        k=context_k,
                        sources=selected_sources or None,
                        rerank=use_mmr,
                    )
                if not contexts:
                    answer = "Nothing relevant found for that question in the indexed documents."
                    st.warning(answer)
                else:
                    with st.spinner("Generating answer..."):
                        answer = answer_question(prompt, contexts)
                    st.markdown(answer)
                    render_sources(contexts)
            except Exception as exc:  # noqa: BLE001 - show API/configuration errors in the UI
                answer = f"Error: {exc}"
                st.error(answer)
                contexts = []

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": contexts,
        }
    )
