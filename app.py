"""
Modern ChatGPT/Perplexity-style LLM Application
Built with LangChain, OpenAI, and Pinecone
"""

import streamlit as st
from src.utils.config import Config
from src.llm.chat_engine import ChatEngine
from src.rag.vector_store import VectorStoreManager
from PIL import Image
import base64
from io import BytesIO
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for ChatGPT/Perplexity-style UI
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .main {
        background-color: #f7f7f8;
    }
    
    /* Chat container */
    .stChatMessage {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
    }
    
    /* User message */
    [data-testid="stChatMessageContent"][data-baseweb="chat-message-user"] {
        background-color: #f7f7f8;
    }
    
    /* Assistant message */
    [data-testid="stChatMessageContent"][data-baseweb="chat-message-assistant"] {
        background-color: white;
        border: 1px solid #ececec;
    }
    
    /* Input box */
    .stChatInputContainer {
        border-top: 1px solid #ececec;
        padding: 1rem 0;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #202123;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    /* Title styling */
    h1 {
        font-weight: 600;
        font-size: 2rem;
        color: #202123;
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    
    /* Remove padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
        max-width: 48rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize configuration
@st.cache_resource
def initialize_services():
    """Initialize all services (cached)."""
    secrets = Config.get_secrets()
    model_config = Config.get_model_config()
    
    # Initialize chat engine
    chat_engine = ChatEngine(
        api_key=secrets["openai_api_key"],
        model=model_config["default_model"],
        temperature=model_config["temperature"]
    )
    
    # Initialize vector store
    vector_store = VectorStoreManager(
        api_key=secrets["pinecone_api_key"],
        environment=secrets["pinecone_environment"],
        index_name=secrets["pinecone_index_name"],
        openai_api_key=secrets["openai_api_key"]
    )
    
    return chat_engine, vector_store, secrets

# Initialize services
try:
    chat_engine, vector_store, secrets = initialize_services()
    services_ready = True
except Exception as e:
    st.error(f"⚠️ Error initializing services: {str(e)}")
    st.stop()

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# Sidebar
with st.sidebar:
    st.title("🤖 AI Assistant")
    st.markdown("---")
    
    # Model selection
    selected_model = st.selectbox(
        "Model",
        ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        index=1
    )
    
    # Temperature
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    
    # RAG toggle
    use_rag = st.toggle("Use RAG (Memory)", value=True)
    
    st.markdown("---")
    
    # Stats
    st.subheader("📊 Stats")
    try:
        stats = vector_store.get_stats()
        st.metric("Conversations Stored", stats.get("total_vectors", 0))
    except:
        st.metric("Conversations Stored", "N/A")
    
    st.metric("Messages in Chat", len(st.session_state.messages))
    
    st.markdown("---")
    
    # Clear chat
    if st.button("🗑️ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.rerun()

# Main chat interface
st.title("💬 AI Chat Assistant")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Message AI Assistant..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get RAG context if enabled
    context_messages = []
    if use_rag and len(st.session_state.messages) > 1:
        try:
            similar_convos = vector_store.search_similar(prompt, k=2)
            if similar_convos:
                context = "\n\n".join([f"Previous context: {conv['content']}" for conv in similar_convos])
                context_messages = [{
                    "role": "system",
                    "content": f"Here is some relevant context from previous conversations:\n{context}"
                }]
        except Exception as e:
            st.error(f"RAG error: {e}")
    
    # Prepare messages for LLM
    system_message = {
        "role": "system",
        "content": "You are a helpful AI assistant. Provide clear, concise, and accurate responses."
    }
    
    llm_messages = [system_message] + context_messages + st.session_state.messages[-10:]  # Last 10 messages
    
    # Get AI response with streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Stream the response
            for chunk in chat_engine.get_response(llm_messages, stream=True):
                if chunk:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            # Add to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            # Store in vector database
            if use_rag:
                try:
                    vector_store.add_conversation(
                        user_message=prompt,
                        ai_message=full_response,
                        metadata={
                            "conversation_id": st.session_state.conversation_id,
                            "timestamp": datetime.now().isoformat(),
                            "model": selected_model
                        }
                    )
                except Exception as e:
                    st.error(f"Error storing conversation: {e}")
                    
        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")

# Footer
st.markdown("---")
st.caption("Powered by OpenAI
