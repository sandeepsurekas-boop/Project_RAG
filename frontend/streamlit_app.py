"""Simple chat UI; all document and AI work runs in the backend API."""

import os
from typing import Any

import requests
import streamlit as st

API_URL = os.getenv("RAG_API_URL", "http://localhost:8000").rstrip("/")
st.set_page_config(page_title="Research Paper Q&A", page_icon="📚", layout="centered")


def api_request(method: str, path: str, **kwargs: Any) -> Any:
    """Call the backend and show readable errors."""
    try:
        response = requests.request(
            method,
            f"{API_URL}{path}",
            timeout=180,
            **kwargs,
        )
        if not response.ok:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            st.error(f"{detail} (HTTP {response.status_code})")
            return None
        return response.json() if response.content else {}
    except requests.RequestException as exc:
        st.error(f"Cannot connect to the backend at {API_URL}: {exc}")
        return None


st.title("Research Paper Q&A")
st.caption("Ask questions about your papers. Answers are based on the papers and include page references.")

if "messages" not in st.session_state:
    st.session_state.messages = []

papers_result = api_request("GET", "/documents")
papers = papers_result.get("documents", []) if papers_result else []

with st.expander("Add or manage papers", expanded=not bool(papers)):
    uploads = st.file_uploader(
        "Choose PDF papers",
        type=["pdf"],
        accept_multiple_files=True,
        help="Up to four papers. The backend saves and processes these files.",
    )
    if st.button("Upload and prepare papers", type="primary"):
        if not uploads:
            st.warning("Choose at least one PDF first.")
        else:
            files = [
                ("files", (item.name, item.getvalue(), "application/pdf"))
                for item in uploads
            ]
            uploaded = api_request("POST", "/upload", files=files)
            if uploaded:
                with st.spinner("Preparing papers for questions..."):
                    processed = api_request("POST", "/process")
                if processed:
                    st.success(
                        f"Ready: {len(processed['documents'])} papers, "
                        f"{processed['pages']} pages."
                    )
                    st.rerun()

    if st.button("Prepare papers already uploaded"):
        with st.spinner("Preparing papers for questions..."):
            processed = api_request("POST", "/process")
        if processed:
            if processed["documents"]:
                st.success(
                    f"Ready: {len(processed['documents'])} papers, "
                    f"{processed['pages']} pages."
                )
                st.rerun()
            st.info("No new uploaded papers need preparing.")

    if papers:
        st.write("**Papers in your knowledge base**")
        for paper in papers:
            st.write(f"- {paper['filename']} ({paper['pages']} pages)")
        confirm_clear = st.checkbox("I want to clear the prepared paper index.")
        if st.button("Clear prepared papers", disabled=not confirm_clear):
            cleared = api_request("DELETE", "/documents")
            if cleared:
                st.session_state.messages = []
                st.success("Paper index cleared. Uploaded PDF files were kept.")
                st.rerun()

if papers:
    st.caption(f"Ready to answer questions from {len(papers)} paper(s).")
else:
    st.info("Add and prepare PDF papers above to start a conversation.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        sources = message.get("sources", [])
        if sources:
            with st.expander(f"Paper references ({len(sources)})"):
                for index, source in enumerate(sources, start=1):
                    st.markdown(
                        f"**{index}. {source['document']} — page {source['page']}**  \n"
                        f"Text match: **{source['match_percent']}%**"
                    )
                    st.write(source["content"])

question = st.chat_input("Ask a question about your papers")
if question:
    if not papers:
        st.warning("Add and prepare at least one paper before asking a question.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        history = [
            {"role": turn["role"], "content": turn["content"]}
            for turn in st.session_state.messages[:-1][-12:]
        ]
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Reading your papers..."):
                result = api_request(
                    "POST",
                    "/query",
                    json={"question": question, "history": history},
                )
            if result:
                st.markdown(result["answer"])
                sources = result.get("sources", [])
                if sources:
                    with st.expander(f"Paper references ({len(sources)})"):
                        for index, source in enumerate(sources, start=1):
                            st.markdown(
                                f"**{index}. {source['document']} — page {source['page']}**  \n"
                                f"Text match: **{source['match_percent']}%**"
                            )
                            st.write(source["content"])
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": sources,
                    }
                )
