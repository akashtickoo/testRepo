"""Streamlit RAG chatbot: answers questions from your documents using Claude."""

import os
from pathlib import Path

import anthropic
import streamlit as st

from rag import Retriever, load_folder, read_file

MODEL = "claude-opus-5-5"
DATA_DIR = Path(__file__).parent / "data"
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using the provided context "
    "excerpts. Base your answer on the context and cite the source file names you used. "
    "If the context does not contain the answer, say so plainly rather than guessing."
)

st.set_page_config(page_title="RAG Chatbot", page_icon="💬")
st.title("💬 RAG Chatbot")

# On Streamlit Cloud the key comes from the app's Secrets; locally from
# .streamlit/secrets.toml or the ANTHROPIC_API_KEY environment variable.
try:
    api_key = st.secrets.get("ANTHROPIC_API_KEY")
except FileNotFoundError:  # no secrets.toml present
    api_key = None
api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    st.error("Set ANTHROPIC_API_KEY in Streamlit secrets to use this app.")
    st.stop()


@st.cache_resource
def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=api_key)


@st.cache_resource(show_spinner="Indexing documents...")
def build_retriever(uploaded: tuple[tuple[str, bytes], ...]) -> Retriever:
    docs = load_folder(DATA_DIR)
    docs += [(name, read_file(name, data)) for name, data in uploaded]
    return Retriever(docs)


with st.sidebar:
    st.header("Documents")
    files = st.file_uploader(
        "Add .txt, .md or .pdf files", type=["txt", "md", "pdf"], accept_multiple_files=True
    )
    uploaded = tuple((f.name, f.getvalue()) for f in files or [])
    retriever = build_retriever(uploaded)
    st.caption(f"{len(retriever.chunks)} chunks indexed (bundled `data/` folder + uploads)")
    if st.button("Clear chat"):
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about your documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    hits = retriever.search(question)
    context = "\n\n".join(f"[Source: {c.source}]\n{c.text}" for c in hits) or "(no relevant context found)"

    # Earlier turns go in as plain text; only the latest question carries the retrieved context.
    api_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
    api_messages.append(
        {"role": "user", "content": f"<context>\n{context}\n</context>\n\nQuestion: {question}"}
    )

    with st.chat_message("assistant"):
        try:
            with get_client().beta.messages.stream(
                model=MODEL,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                messages=api_messages,
                output_config={"effort": "low"},  # fast, inexpensive answers for chat
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",  # re-run on a fallback model if a safety classifier declines
            ) as stream:
                answer = st.write_stream(stream.text_stream)
                final = stream.get_final_message()
            if final.stop_reason == "refusal":
                answer = "Sorry, I can't help with that request."
                st.warning(answer)
        except anthropic.AuthenticationError:
            answer = "Invalid API key. Check ANTHROPIC_API_KEY in your secrets."
            st.error(answer)
        except anthropic.RateLimitError:
            answer = "Rate limited by the API. Please wait a moment and try again."
            st.error(answer)
        except anthropic.APIStatusError as e:
            answer = f"API error ({e.status_code}): {e.message}"
            st.error(answer)
        except anthropic.APIConnectionError:
            answer = "Could not reach the Claude API. Check your network and try again."
            st.error(answer)

        if hits:
            with st.expander("Sources"):
                for c in hits:
                    st.markdown(f"**{c.source}**: {c.text[:300]}...")

    st.session_state.messages.append({"role": "assistant", "content": answer or ""})
