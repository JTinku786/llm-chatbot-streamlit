"""
Modern ChatGPT/Perplexity-style LLM Application with Multi-Chat & File Upload
Built with OpenAI, Pinecone, and Advanced Document Processing
"""

import os

import streamlit as st
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai  import OpenAI
from PIL import Image
import base64
from io import BytesIO
from datetime import datetime
import json
import requests
from pypdf import PdfReader
from pptx import Presentation
import docx

# Page configuration
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ChatGPT-style UI
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main {
        background-color: #f7f7f8;
    }
    
    .stChatMessage {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
    }
    
    .stChatInputContainer {
        border-top: 1px solid #ececec;
        padding: 1rem 0;
    }
    
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
    
    h1 {
        font-weight: 600;
        font-size: 2rem;
        color: #202123;
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
        max-width: 48rem;
    }
    
    .file-upload-area {
        border: 2px dashed #565869;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f7f7f8;
        text-align: center;
    }
    
    .uploaded-file {
        background-color: #e3f2fd;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Load configuration
@st.cache_resource
def load_config():
    try:
        return {
            "openai_api_key": st.secrets["OPENAI_API_KEY"],
            "pinecone_api_key": st.secrets.get("PINECONE_API_KEY", ""),
            "pinecone_environment": st.secrets.get("PINECONE_ENVIRONMENT", "gcp-starter"),
            "pinecone_index_name": st.secrets.get("PINECONE_INDEX_NAME", "chatbot-memory"),
            "langsmith_api_key": st.secrets.get("LANGSMITH_API_KEY", ""),
            "langsmith_project": st.secrets.get("LANGSMITH_PROJECT", "llm-chatbot-streamlit"),
            "langsmith_endpoint": st.secrets.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
            "openweathermap_api_key": st.secrets.get("OPENWEATHERMAP_API_KEY", "")
        }
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        return None

config = load_config()
if not config:
    st.stop()

if config["langsmith_api_key"]:
    os.environ["LANGSMITH_API_KEY"] = config["langsmith_api_key"]
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = config["langsmith_project"]
    os.environ["LANGSMITH_ENDPOINT"] = config["langsmith_endpoint"]

client = wrap_openai(OpenAI(api_key=config["openai_api_key"]))

@traceable(name="chat_completion_stream", run_type="llm")
def stream_chat_completion(selected_model, messages, temperature):
    return client.chat.completions.create(
        model=selected_model,
        messages=messages,
        temperature=temperature,
        stream=True
    )



def load_weather_context(cities):
    """Load weather context for the given cities using OpenWeatherMap."""
    if not config["openweathermap_api_key"]:
        return "", "OPENWEATHERMAP_API_KEY is not configured."

    clean_cities = [city.strip() for city in cities if city and city.strip()]
    if not clean_cities:
        return "", "No valid city names provided."

    try:
        weather_reports = []
        for city in clean_cities:
            response = requests.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={
                    "q": city,
                    "appid": config["openweathermap_api_key"],
                    "units": "metric",
                },
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            weather_reports.append(
                f"{data['name']}: {data['weather'][0]['description']}, "
                f"temperature {data['main']['temp']}°C, feels like {data['main']['feels_like']}°C, "
                f"humidity {data['main']['humidity']}%, wind {data['wind']['speed']} m/s"
            )

        return "\n".join(weather_reports), ""
    except requests.RequestException as exc:
        return "", f"Weather lookup failed: {exc}"

# Helper functions for file processing
def encode_image_to_base64(image):
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file."""
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"

def extract_text_from_pptx(pptx_file):
    """Extract text from PowerPoint file."""
    try:
        prs = Presentation(pptx_file)
        text = ""
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
        return text
    except Exception as e:
        return f"Error extracting PPTX: {str(e)}"

def extract_text_from_docx(docx_file):
    """Extract text from Word document."""
    try:
        doc = docx.Document(docx_file)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        return f"Error extracting DOCX: {str(e)}"

def process_uploaded_file(uploaded_file):
    """Process uploaded file and return content."""
    file_type = uploaded_file.type
    file_name = uploaded_file.name
    
    if file_type.startswith('image/'):
        # Image processing
        image = Image.open(uploaded_file)
        return {
            "type": "image",
            "name": file_name,
            "content": image,
            "base64": encode_image_to_base64(image)
        }
    elif file_type == 'application/pdf':
        # PDF processing
        text = extract_text_from_pdf(uploaded_file)
        return {
            "type": "pdf",
            "name": file_name,
            "content": text
        }
    elif file_type == 'application/vnd.openxmlformats-officedocument.presentationml.presentation':
        # PowerPoint processing
        text = extract_text_from_pptx(uploaded_file)
        return {
            "type": "pptx",
            "name": file_name,
            "content": text
        }
    elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        # Word document processing
        text = extract_text_from_docx(uploaded_file)
        return {
            "type": "docx",
            "name": file_name,
            "content": text
        }
    else:
        return {
            "type": "unknown",
            "name": file_name,
            "content": "Unsupported file type"
        }

# Initialize session state
if "all_chats" not in st.session_state:
    st.session_state.all_chats = {}

if "current_chat_id" not in st.session_state:
    chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.current_chat_id = chat_id
    st.session_state.all_chats[chat_id] = {
        "title": "New Chat",
        "messages": [],
        "created_at": datetime.now().isoformat()
    }

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Helper functions for chat management
def create_new_chat():
    chat_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    st.session_state.all_chats[chat_id] = {
        "title": "New Chat",
        "messages": [],
        "created_at": datetime.now().isoformat()
    }
    st.session_state.current_chat_id = chat_id
    st.session_state.uploaded_files = []

def delete_chat(chat_id):
    if len(st.session_state.all_chats) > 1:
        del st.session_state.all_chats[chat_id]
        st.session_state.current_chat_id = list(st.session_state.all_chats.keys())[-1]
    else:
        st.session_state.all_chats[chat_id]["messages"] = []
        st.session_state.all_chats[chat_id]["title"] = "New Chat"

def switch_chat(chat_id):
    st.session_state.current_chat_id = chat_id
    st.session_state.uploaded_files = []

def get_chat_title(messages):
    if messages:
        first_user_msg = next((msg["content"] for msg in messages if msg["role"] == "user"), None)
        if first_user_msg and isinstance(first_user_msg, str):
            return first_user_msg[:50] + "..." if len(first_user_msg) > 50 else first_user_msg
    return "New Chat"

current_chat = st.session_state.all_chats[st.session_state.current_chat_id]

# Sidebar
with st.sidebar:
    st.markdown("### 🤖 AI Chat Assistant")
    
    if st.button("➕ New Chat", use_container_width=True, key="new_chat_btn"):
        create_new_chat()
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 💬 Chat History")
    
    sorted_chats = sorted(
        st.session_state.all_chats.items(),
        key=lambda x: x[1]["created_at"],
        reverse=True
    )
    
    for chat_id, chat_data in sorted_chats:
        if chat_data["messages"]:
            chat_data["title"] = get_chat_title(chat_data["messages"])
        
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
    st.markdown("### ⚙️ Settings")
    
    selected_model = st.selectbox(
        "Model",
        ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        index=0  # gpt-4o for vision support
    )
    
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    use_rag = st.toggle("Use Memory (RAG)", value=True)
    
    st.markdown("---")
    st.markdown("### 📊 Stats")
    st.metric("Total Chats", len(st.session_state.all_chats))
    st.metric("Messages", len(current_chat["messages"]))
    st.metric("Files Attached", len(st.session_state.uploaded_files))

# Main chat interface
st.title("💬 AI Chat Assistant")

# File upload section
with st.expander("📎 Attach Files (Images, PDFs, PPT, Docs)", expanded=bool(st.session_state.uploaded_files)):
    uploaded_files = st.file_uploader(
        "Upload files to analyze",
        type=["png", "jpg", "jpeg", "pdf", "pptx", "docx"],
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if uploaded_files:
        st.session_state.uploaded_files = []
        for uploaded_file in uploaded_files:
            processed = process_uploaded_file(uploaded_file)
            st.session_state.uploaded_files.append(processed)
            
            # Display uploaded file
            if processed["type"] == "image":
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.image(processed["content"], width=100)
                with col2:
                    st.success(f"✅ {processed['name']}")
            else:
                st.success(f"✅ {processed['name']} ({processed['type'].upper()}) - {len(processed['content'])} characters extracted")

# Display chat messages
for message in current_chat["messages"]:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], list):
            # Handle multimodal content
            for content_part in message["content"]:
                if isinstance(content_part, dict):
                    if content_part.get("type") == "image_url":
                        st.image(content_part["image_url"]["url"])
                    elif content_part.get("type") == "text":
                        st.markdown(content_part["text"])
                else:
                    st.markdown(content_part)
        else:
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Message AI Assistant... (use /weather city1, city2)"):
    # Prepare user message content
    user_message_content = []

    weather_context = ""
    if prompt.lower().startswith("/weather "):
        city_input = prompt[len("/weather "):].strip()
        city_list = [city.strip() for city in city_input.split(",")]
        weather_context, weather_error = load_weather_context(city_list)
        if weather_error:
            st.warning(weather_error)
        elif weather_context:
            st.info(f"Fetched weather data for: {', '.join(city_list)}")

    # Add text
    user_message_content.append({"type": "text", "text": prompt})
    
    # Add uploaded files
    file_context = ""
    for file_data in st.session_state.uploaded_files:
        if file_data["type"] == "image":
            # Add image to message
            user_message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{file_data['base64']}"
                }
            })
        else:
            # Add document content as
            file_context += f"\n{file_data['name']}: {file_data['content'][:500]}"    
    if file_context:
        user_message_content.append({"type": "text", "text": f"\n\nFiles: {file_context}"})

    if weather_context:
        user_message_content.append({"type": "text", "text": f"\n\nLive weather context:\n{weather_context}"})

    current_chat["messages"].append({"role": "user", "content": user_message_content if len(user_message_content) > 1 else prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        try:
            response = stream_chat_completion(selected_model, current_chat["messages"], temperature)
            full_response = ""
            placeholder = st.empty()
            for chunk in response:
                    if chunk.choices[0].delta.content:                    full_response += chunk.choices[0].delta.content
                    placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
            current_chat["messages"].append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"Error: {str(e)}")
    st.rerun()
