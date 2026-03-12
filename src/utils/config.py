"""Configuration management for the application."""
import os
import streamlit as st
from typing import Dict, Any

class Config:
    """Application configuration loader."""
    
    @staticmethod
    def get_secrets() -> Dict[str, Any]:
        """Load secrets from Streamlit secrets or environment variables."""
        try:
            # Try Streamlit secrets first (for cloud deployment)
            return {
                "openai_api_key": st.secrets["OPENAI_API_KEY"],
                "pinecone_api_key": st.secrets["PINECONE_API_KEY"],
                "pinecone_environment": st.secrets.get("PINECONE_ENVIRONMENT", "gcp-starter"),
                "pinecone_index_name": st.secrets.get("PINECONE_INDEX_NAME", "chatbot-memory")
            }
        except:
            # Fallback to environment variables (for local development)
            return {
                "openai_api_key": os.getenv("OPENAI_API_KEY"),
                "pinecone_api_key": os.getenv("PINECONE_API_KEY"),
                "pinecone_environment": os.getenv("PINECONE_ENVIRONMENT", "gcp-starter"),
                "pinecone_index_name": os.getenv("PINECONE_INDEX_NAME", "chatbot-memory")
            }
    
    @staticmethod
    def get_model_config() -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "default_model": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 2000,
            "streaming": True
        }
