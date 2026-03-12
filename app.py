import streamlit as st
import openai
from pinecone import Pinecone, ServerlessSpec
import base64
from io import BytesIO
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="LLM Chatbot with Vision",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 LLM Chatbot with OpenAI & Pinecone")
st.markdown("Chat with AI using text and images, powered by OpenAI and Pinecone vector database")

# Sidebar for API keys and configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # OpenAI API Key
    openai_api_key = st.text_input("OpenAI API Key", type="password", key="openai_key")
    
    # Pinecone configuration
    st.subheader("Pinecone Settings")
    pinecone_api_key = st.text_input("Pinecone API Key", type="password", key="pinecone_key")
    pinecone_env = st.text_input("Pinecone Environment", value="gcp-starter", key="pinecone_env")
    pinecone_index_name = st.text_input("Index Name", value="chatbot-memory", key="index_name")
    
    # Model selection
    model_choice = st.selectbox(
        "Select OpenAI Model",
        ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        index=1
    )
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Initialize Pinecone
def init_pinecone():
    if pinecone_api_key:
        try:
            pc = Pinecone(api_key=pinecone_api_key)
            
            # Create index if it doesn't exist
            if pinecone_index_name not in [index.name for index in pc.list_indexes()]:
                pc.create_index(
                    name=pinecone_index_name,
                    dimension=1536,  # OpenAI embedding dimension
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
            
            return pc.Index(pinecone_index_name)
        except Exception as e:
            st.sidebar.error(f"Pinecone Error: {str(e)}")
            return None
    return None

# Initialize OpenAI client
def init_openai():
    if openai_api_key:
        openai.api_key = openai_api_key
        return True
    return False

# Function to encode image to base64
def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Function to get embeddings
def get_embedding(text):
    try:
        response = openai.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Embedding Error: {str(e)}")
        return None

# Function to store message in Pinecone
def store_in_pinecone(index, message_id, text, metadata):
    if index:
        try:
            embedding = get_embedding(text)
            if embedding:
                index.upsert([(message_id, embedding, metadata)])
        except Exception as e:
            st.sidebar.warning(f"Storage Error: {str(e)}")

# Function to query Pinecone for similar messages
def query_pinecone(index, query_text, top_k=3):
    if index:
        try:
            query_embedding = get_embedding(query_text)
            if query_embedding:
                results = index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True
                )
                return results.matches
        except Exception as e:
            st.sidebar.warning(f"Query Error: {str(e)}")
    return []

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize clients
openai_ready = init_openai()
pinecone_index = init_pinecone()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "content" in message:
            st.markdown(message["content"])
        if "image" in message:
            st.image(message["image"], width=300)

# Image upload
uploaded_image = st.file_uploader("📷 Upload an image (optional)", type=["png", "jpg", "jpeg"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    if not openai_ready:
        st.error("⚠️ Please enter your OpenAI API key in the sidebar")
    else:
        # Add user message to chat
        user_message = {"role": "user", "content": prompt}
        
        # Handle image if uploaded
        image_data = None
        if uploaded_image:
            image = Image.open(uploaded_image)
            user_message["image"] = image
            image_data = encode_image(image)
        
        st.session_state.messages.append(user_message)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            if uploaded_image:
                st.image(image, width=300)
        
        # Store in Pinecone
        if pinecone_index:
            message_id = f"msg_{len(st.session_state.messages)}"
            store_in_pinecone(
                pinecone_index,
                message_id,
                prompt,
                {"role": "user", "content": prompt}
            )
        
        # Prepare messages for OpenAI
        messages_for_api = []
        
        # Add context from Pinecone if available
        if pinecone_index:
            similar_messages = query_pinecone(pinecone_index, prompt)
            if similar_messages:
                context = "Previous relevant conversations:\n"
                for match in similar_messages:
                    context += f"- {match.metadata.get('content', '')}\n"
                messages_for_api.append({"role": "system", "content": context})
        
        # Add conversation history
        for msg in st.session_state.messages[-5:]:  # Last 5 messages for context
            if "image" in msg and image_data:
                messages_for_api.append({
                    "role": msg["role"],
                    "content": [
                        {"type": "text", "text": msg["content"]},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                    ]
                })
            else:
                messages_for_api.append({"role": msg["role"], "content": msg.get("content", "")})
        
        # Get AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Stream response
                stream = openai.chat.completions.create(
                    model=model_choice,
                    messages=messages_for_api,
                    stream=True,
                    max_tokens=1000
                )
                
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                # Store in Pinecone
                if pinecone_index:
                    message_id = f"msg_{len(st.session_state.messages)}"
                    store_in_pinecone(
                        pinecone_index,
                        message_id,
                        full_response,
                        {"role": "assistant", "content": full_response}
                    )
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.error("Please check your API keys and try again.")

# Sidebar stats
if pinecone_index:
    with st.sidebar:
        st.divider()
        st.subheader("📊 Stats")
        try:
            stats = pinecone_index.describe_index_stats()
            st.metric("Vectors Stored", stats.total_vector_count)
        except:
            pass
