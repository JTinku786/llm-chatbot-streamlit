# llm-chatbot-streamlit

A Streamlit chat assistant using OpenAI with multimodal file upload support.

## Optional: LangSmith tracing

To enable tracing, add these secrets in your Streamlit deployment:

- `LANGSMITH_API_KEY`
- `LANGSMITH_PROJECT` (optional, defaults to `llm-chatbot-streamlit`)
- `LANGSMITH_ENDPOINT` (optional, defaults to `https://api.smith.langchain.com`)

When `LANGSMITH_API_KEY` is set, the app enables LangSmith tracing and wraps OpenAI calls for run-level visibility.


## Weather tool (OpenWeatherMap)

The chat supports on-demand weather lookup using the command format:

- `/weather chennai, vellore`
- `weather in bodhan`

Add this secret to Streamlit Cloud:

- `OPENWEATHERMAP_API_KEY`

This key is used to call the OpenWeatherMap current weather API and include live weather context in the prompt.
