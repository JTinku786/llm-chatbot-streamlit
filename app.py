"""
Modern ChatGPT/Perplexity-style LLM Application
Built with LangChain, OpenAI, and Pinecone
"""

import streamlit as st
import openai
from pinecone import Pinecone, ServerlessSpec
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

# Load configuration
@st.cache_resource
def load_config():
    """Load configuration from secrets."""
    try:
        return {
            "openai_api_key": st.secrets["OPENAI_API_KEY"],
            "pinecone_api_key": st.secrets["PINECONE_API_KEY"],
            "pinecone_environment": st.secrets.get("PINECONE_ENVIRONMENT", "gcp-starter"),
            "pinecone_index_name": st.secrets.get("PINECONE_INDEX_NAME", "chatbot-memory")
        }
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        return None

config = load_config()

if not config:
    st.stop()

# Initialize OpenAI
openai.api_key = config["openai_api_key"]

# Initialize Pinecone
@st.cache_resource
def init_pinecone():
    """Initialize Pinecone vector store."""
    try:
        pc = Pinecone(api_key=config["pinecone_api_key"])
        
        # Create index if it doesn't exist
        if config["pinecone_index_name"] not in [index.name for index in pc.list_indexes()]:
            pc.create_index(
                name=config["pinecone_index_name"],
                dimension=1536,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
        
        return pc.Index(config["pinecone_index_name"])
    except Exception as e:
        st.sidebar.error(f"Pinecone Error: {str(e)}")
        return None

pinecone_index = init_pinecone()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# Sidebar
with st.sidebar:
    st.title("🤖 AI Assistant")
    st.markdown("---")
    
    st.success("✅ Connected")
    st.info(f"🔧 Environment: {config['pinecone_environment']}")
    
    # Model selection
    selected_model = st.selectbox(
        "Model",
        ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        index=1
    )
    
    # Temperature
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    
    # RAG toggle
    use_rag = st.toggle("Use Memory (RAG)", value=True)
    
    st.markdown("---")
    
    # Stats
    st.subheader("📊 Stats")
    if pinecone_index:
        try:
            stats = pinecone_index.describe_index_stats()
            st.metric("Conversations", stats.total_vector_count)
        except:
            st.metric("Conversations", "N/A")
    
    st.metric("Messages", len(st.session_state.messages))
    
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
    
    # Get RAG context
    context = ""
    if use_rag and pinecone_index and len(st.session_state.messages) > 1:
        try:
            # Get embeddings
            embed_response = openai.embeddings.create(
                input=prompt,
                model="text-embedding-ada-002"
            )
            query_embedding = embed_response.data[0].embedding
            
            # Search similar conversations
            results = pinecone_index.query(
                vector=query_embedding,
                top_k=2,
                include_metadata=True
            )
            
            if results.matches:
                context = "\n\nRelevant context from previous conversations:\n"
                for match in results.matches:
                    context += f"- {match.metadata.get('content', '')}\n"
        except Exception as e:
            pass
    
    # Prepare messages
    system_msg = {
        "role": "system",
        "content": f"You are a helpful AI assistant. Provide clear and concise responses.{context}"
    }
    
    messages_for_api = [system_msg] + st.session_state.messages[-10:]
    
    # Get AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            stream = openai.chat.completions.create(
                model=selected_model,
                messages=messages_for_api,
                stream=True,
                temperature=temperature,
                max_tokens=2000
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
            # Add to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            # Store in Pinecone
            if use_rag and pinecone_index:
                try:
                    embed_response = openai.embeddings.create(
                        input=f"User: {prompt}\nAssistant: {full_response}",
                        model="text-embedding-ada-002"
                    )
                    embedding = embed_response.data[0].embedding
                    
                    pinecone_index.upsert([(
                        f"conv_{st.session_state.conversation_id}_{len(st.session_state.messages)}",
                        embedding,
                        {
                            "content": f"User: {prompt}\nAssistant: {full_response}",
                            "conversation_id": st.session_state.conversation_id,
                            "timestamp": datetime.now().isoformat()
                        }
                    )])
                except Exception as e:
                    pass
                    
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.caption("Powered by OpenAI & Pinecone | Built with Streamlit")
