"""
Modern ChatGPT/Perplexity-style LLM Application with Multi-Chat Support
Built with LangChain, OpenAI, and Pinecone
"""

import streamlit as st
import openai
from pinecone import Pinecone, ServerlessSpec
from PIL import Image
import base64
from io import BytesIO
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ChatGPT-style UI with chat history
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
        color: #ececec;
    }
    
    [data-testid="stSidebar"] .stButton button {
        background-color: transparent;
        color: #ececec;
        border: 1px solid #565869;
        width: 100%;
        text-align: left;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        margin: 0.25rem 0;
    }
    
    [data-testid="stSidebar"] .stButton button:hover {
        background-color: #2a2b32;
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
    
    /* Chat history item */
    .chat-item {
        background-color: transparent;
        padding: 0.75rem;
        margin: 0.25rem 0;
        border-radius: 0.5rem;
        cursor: pointer;
        border: 1px solid transparent;
        color: #ececec;
    }
    
    .chat-item:hover {
        background-color: #2a2b32;
        border-color: #565869;
    }
    
    .chat-item.active {
        background-color: #343541;
        border-color: #565869;
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

# Initialize session state for multi-chat
if "all_chats" not in st.session_state:
    st.session_state.all_chats = {}

if "current_chat_id" not in st.session_state:
    # Create first chat
    chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.current_chat_id = chat_id
    st.session_state.all_chats[chat_id] = {
        "title": "New Chat",
        "messages": [],
        "created_at": datetime.now().isoformat()
    }

# Helper functions
def create_new_chat():
    """Create a new chat session."""
    chat_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    st.session_state.all_chats[chat_id] = {
        "title": "New Chat",
        "messages": [],
        "created_at": datetime.now().isoformat()
    }
    st.session_state.current_chat_id = chat_id

def delete_chat(chat_id):
    """Delete a chat session."""
    if len(st.session_state.all_chats) > 1:
        del st.session_state.all_chats[chat_id]
        # Switch to most recent chat
        st.session_state.current_chat_id = list(st.session_state.all_chats.keys())[-1]
    else:
        # Don't delete last chat, just clear it
        st.session_state.all_chats[chat_id]["messages"] = []
        st.session_state.all_chats[chat_id]["title"] = "New Chat"

def switch_chat(chat_id):
    """Switch to a different chat."""
    st.session_state.current_chat_id = chat_id

def get_chat_title(messages):
    """Generate chat title from first user message."""
    if messages:
        first_user_msg = next((msg["content"] for msg in messages if msg["role"] == "user"), None)
        if first_user_msg:
            return first_user_msg[:50] + "..." if len(first_user_msg) > 50 else first_user_msg
    return "New Chat"

# Get current chat
current_chat = st.session_state.all_chats[st.session_state.current_chat_id]

# Sidebar with chat history
with st.sidebar:
    # New Chat button at top
    st.markdown("### 🤖 AI Chat Assistant")
    
    if st.button("➕ New Chat", use_container_width=True, key="new_chat_btn"):
        create_new_chat()
        st.rerun()
    
    st.markdown("---")
    
    # Chat history
    st.markdown("### 💬 Chat History")
    
    # Sort chats by creation time (most recent first)
    sorted_chats = sorted(
        st.session_state.all_chats.items(),
        key=lambda x: x[1]["created_at"],
        reverse=True
    )
    
    for chat_id, chat_data in sorted_chats:
        # Auto-update title based on messages
        if chat_data["messages"]:
            chat_data["title"] = get_chat_title(chat_data["messages"])
        
        # Create columns for chat item and delete button
        col1, col2 = st.columns([5, 1])
        
        with col1:
            is_active = chat_id == st.session_state.current_chat_id
            button_type = "primary" if is_active else "secondary"
            
            if st.button(
                f"{'📍 ' if is_active else '💬 '}{chat_data['title']}",
                key=f"chat_{chat_id}",
                use_container_width=True,
                type=button_type
            ):
                switch_chat(chat_id)
                st.rerun()
        
        with col2:
            if st.button("🗑️", key=f"del_{chat_id}", help="Delete chat"):
                delete_chat(chat_id)
                st.rerun()
    
    st.markdown("---")
    
    # Settings
    st.markdown("### ⚙️ Settings")
    
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
    st.markdown("### 📊 Stats")
    st.metric("Total Chats", len(st.session_state.all_chats))
    st.metric("Messages in Chat", len(current_chat["messages"]))
    
    if pinecone_index:
        try:
            stats = pinecone_index.describe_index_stats()
            st.metric("Stored Conversations", stats.total_vector_count)
        except:
            pass

# Main chat interface
st.title("💬 AI Chat Assistant")

# Display current chat messages
for message in current_chat["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Message AI Assistant..."):
    # Add user message to current chat
    current_chat["messages"].append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get RAG context
    context = ""
    if use_rag and pinecone_index and len(current_chat["messages"]) > 1:
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
    
    messages_for_api = [system_msg] + current_chat["messages"][-10:]
    
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
            
            # Add to current chat history
            current_chat["messages"].append({"role": "assistant", "content": full_response})
            
            # Store in Pinecone
            if use_rag and pinecone_index:
                try:
                    embed_response = openai.embeddings.create(
                        input=f"User: {prompt}\nAssistant: {full_response}",
                        model="text-embedding-ada-002"
                    )
                    embedding = embed_response.data[0].embedding
                    
                    pinecone_index.upsert([(
                        f"conv_{st.session_state.current_chat_id}_{len(current_chat['messages'])}",
                        embedding,
                        {
                            "content": f"User: {prompt}\nAssistant: {full_response}",
                            "chat_id": st.session_state.current_chat_id,
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
