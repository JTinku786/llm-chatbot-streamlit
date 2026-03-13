# llm-chatbot-streamlit

A Streamlit chat assistant using OpenAI with multimodal file upload support.

## Optional: LangSmith tracing

To enable tracing, add these secrets in your Streamlit deployment:

- `LANGSMITH_API_KEY`
- `LANGSMITH_PROJECT` (optional, defaults to `llm-chatbot-streamlit`)
- `LANGSMITH_ENDPOINT` (optional, defaults to `https://api.smith.langchain.com`)

When `LANGSMITH_API_KEY` is set, the app enables LangSmith tracing and wraps OpenAI calls for run-level visibility.
