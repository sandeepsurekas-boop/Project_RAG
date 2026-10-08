"""Streamlit interface for the Research Paper RAG API."""

import os
from typing import Any

import requests
import streamlit as st

API_URL = os.getenv("RAG_API_URL", "http://localhost:8000").rstrip("/")
st.set_page_config(page_title="Research Paper RAG", page_icon="📚", layout="wide")


def api_request(method: str, path: str, **kwargs: Any) -> Any:
    """Call the backend and render a readable error on failure."""
    try:
        response = requests.request(
            method, f"{API_URL}{path}", timeout=180, **kwargs
        )
        if not response.ok:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            st.error(f"API error ({response.status_code}): {detail}")
            return None
        return response.json() if response.content else {}
    except requests.RequestException as exc:
        st.error(f"Cannot reach the RAG API at {API_URL}: {exc}")
        return None


st.title("Research Paper RAG")
st.caption("Ask questions across AI research papers and inspect the evidence behind each answer.")

with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "Upload research papers (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Add up to four PDF papers (three or four recommended). Each file can be up to the configured upload limit.",
    )
    if st.button("Upload PDFs", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Choose at least one PDF first.")
        else:
            files = [
                ("files", (file.name, file.getvalue(), "application/pdf"))
                for file in uploaded_files
            ]
            uploaded = api_request("POST", "/upload", files=files)
            if uploaded:
                st.success(uploaded["message"])
    if st.button("Process Documents", use_container_width=True):
        with st.spinner("Extracting, embedding, and indexing documents..."):
            processed = api_request("POST", "/process")
        if processed:
            if processed["documents"]:
                st.success(
                    f"Indexed {len(processed['documents'])} document(s), "
                    f"{processed['pages']} pages, and {processed['chunks']} chunks."
                )
            else:
                st.info("There are no new uploaded documents to process.")

    top_k = st.slider("Retrieved chunks (top-k)", min_value=1, max_value=20, value=5)
    threshold = st.slider(
        "Similarity threshold",
        min_value=-1.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Only chunks whose cosine similarity meets this threshold are used.",
    )
    if st.button("Clear vector database", use_container_width=True):
        cleared = api_request("DELETE", "/documents")
        if cleared:
            st.success(cleared["message"])

st.subheader("Processed papers")
documents_result = api_request("GET", "/documents")
documents = documents_result.get("documents", []) if documents_result else []
if documents:
    for document in documents:
        st.write(
            f"**{document['filename']}** — {document['pages']} pages, "
            f"{document['chunks']} chunks"
        )
else:
    st.info("No processed papers yet. Upload PDFs in the sidebar to get started.")

st.divider()
st.subheader("Ask a question")
with st.form("question_form"):
    question = st.text_area(
        "Question",
        placeholder="For example: Why is multi-head attention beneficial?",
        height=90,
    )
    submitted = st.form_submit_button("Get answer", type="primary")

if submitted:
    if not question.strip():
        st.warning("Enter a question before submitting.")
    elif not documents:
        st.warning("Process at least one paper before asking a question.")
    else:
        with st.spinner("Backend is searching the papers and generating a cited answer..."):
            result = api_request(
                "POST",
                "/query",
                json={
                    "question": question.strip(),
                    "top_k": top_k,
                    "similarity_threshold": threshold,
                },
            )
        if result:
            st.markdown("### Answer")
            st.write(result["answer"])
            with st.expander(f"Sources / Retrieved Context ({len(result['sources'])})"):
                if not result["sources"]:
                    st.info("No relevant source chunks met the similarity threshold.")
                for index, source in enumerate(result["sources"], start=1):
                    st.markdown(
                        f"**Source {index}: {source['document']} — page {source['page']}**  \n"
                        f"Similarity: `{source['score']:.3f}` · "
                        f"Chunk ID: `{source['chunk_id']}`"
                    )
                    st.write(source["content"])
                    if index < len(result["sources"]):
                        st.divider()
