"""Pinecone vector store integration."""
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from typing import List, Dict, Any

class VectorStoreManager:
    """Manage Pinecone vector store for RAG."""
    
    def __init__(self, api_key: str, environment: str, index_name: str, openai_api_key: str):
        """Initialize Pinecone and embeddings."""
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.environment = environment
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        
        # Create index if it doesn't exist
        self._create_index_if_not_exists()
        
        # Initialize vector store
        self.vector_store = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings,
            pinecone_api_key=api_key
        )
    
    def _create_index_if_not_exists(self):
        """Create Pinecone index if it doesn't exist."""
        try:
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,  # OpenAI embedding dimension
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
        except Exception as e:
            print(f"Error creating index: {e}")
    
    def add_conversation(self, user_message: str, ai_message: str, metadata: Dict[str, Any] = None):
        """Add conversation to vector store."""
        texts = [f"User: {user_message}\nAssistant: {ai_message}"]
        metadatas = [metadata or {}]
        
        self.vector_store.add_texts(texts=texts, metadatas=metadatas)
    
    def search_similar(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar conversations."""
        results = self.vector_store.similarity_search(query, k=k)
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        try:
            index = self.pc.Index(self.index_name)
            stats = index.describe_index_stats()
            return {"total_vectors": stats.total_vector_count}
        except:
            return {"total_vectors": 0}
