# RAG Chatbot (Streamlit + Claude)

Ask questions about your documents. Files in `data/` (plus any you upload in the sidebar)
are split into chunks, ranked with TF-IDF, and the top matches are sent to Claude
(`claude-opus-5-5`) as context for the answer.

## Run locally

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then add your key
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to https://share.streamlit.io -> **Create app** -> pick this repo, branch `main`, main file `app.py`.
3. Under **Advanced settings -> Secrets**, add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
4. Deploy.

## Your own documents

Put `.txt`, `.md` or `.pdf` files in `data/` and push, or upload them from the sidebar
(uploads last only for the session). `data/sample.md` is placeholder content you can delete.

## Files

- `app.py` - Streamlit chat UI and the Claude call
- `rag.py` - document loading, chunking and TF-IDF retrieval
