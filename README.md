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


## Web search tools

The chat supports web search augmentation with these prompt formats:

- `/search latest AI news`
- `search NVIDIA quarterly results`
- `google best cafes in hyderabad`
- `look up python 3.13 release notes`

Supported providers in the sidebar: `Auto`, `Tavily`, `SerpAPI`.

Add secrets in Streamlit Cloud:

- `TAVILY_API_KEY`
- `SERPAPI_API_KEY`

In `Auto` mode, the app tries Tavily first, then SerpAPI.


## Tool routing strategy

The app now uses a lightweight tool router before each LLM call:

1. Inspect user prompt for weather or live-web intent.
2. Call weather or web-search tools when needed.
3. Attach fetched context to the user message.
4. Send enriched context to the LLM for final response.

This routing is traced in LangSmith with dedicated runs (`tool_router`, `weather_context`, `web_search_context`, `search_tavily`, `search_serpapi`).


## Pinecone conversation storage

Each completed chat turn (user + assistant) is persisted to Pinecone when `PINECONE_API_KEY` and an existing `PINECONE_INDEX_NAME` are configured.

Stored metadata includes: `chat_id`, `timestamp`, `user_message`, `assistant_message`, and `source`.

If persistence is skipped, the LangSmith `store_conversation_pinecone` run now returns a reason (for example missing API key, missing index, or upsert failure).


If `store_conversation_pinecone` fails, check the `reason` in LangSmith output and sidebar. This now includes detailed init/import/upsert errors (including Pinecone import/plugin issues).


### Troubleshooting Pinecone writes

If LangSmith still shows `output: false` or `success: false` for `store_conversation_pinecone`:

1. Ensure the deployed app is on the latest commit (older builds returned only `false`).
2. Verify `PINECONE_API_KEY` and `PINECONE_INDEX_NAME` are set exactly in Streamlit secrets.
3. Ensure the index exists and is reachable with the same API key/project.
4. Check `reason` and `error_type` fields in the run output/sidebar for exact failure cause.
5. If using a custom index dimension, the app now adapts embedding dimensions up to 1536 for `text-embedding-3-small`.
6. The Streamlit app writes to Pinecone via REST (no direct `pinecone` SDK import), which avoids deprecated plugin import crashes in some hosted environments.
