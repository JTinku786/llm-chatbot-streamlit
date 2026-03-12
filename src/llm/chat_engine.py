"""LangChain-based chat engine with OpenAI."""
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from typing import List, Generator
import streamlit as st

class ChatEngine:
    """Chat engine using LangChain and OpenAI."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.7):
        """Initialize the chat engine."""
        self.llm = ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=temperature,
            streaming=True
        )
        
    def get_response(self, messages: List[dict], stream: bool = True) -> Generator:
        """Get response from the chat model."""
        # Convert messages to LangChain format
        langchain_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
        
        if stream:
            # Streaming response
            for chunk in self.llm.stream(langchain_messages):
                yield chunk.content
        else:
            # Non-streaming response
            response = self.llm.invoke(langchain_messages)
            yield response.content
