"""
This will handle the embedding generation using Azure OpenAI.
"""

from typing import List, Optional
from pathlib import Path
import sys

from langchain_openai import AzureOpenAIEmbeddings
from system_config import(AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, 
                          AZURE_OPENAI_API_VERSION, AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME)

sys.path.append(str(Path(__file__).parent.parent.parent))

class DocumentEmbeddingModule:
    """
    This will handle embedding generation using Azure OpenAI.
    """
    def __init__(self, api_key: Optional[str] = None, endpoint_url: Optional[str] = None, 
                 api_version: Optional[str] = None, deployement_name: Optional[str] = None):
        """
        Constructor to initialize the embedding manager with Azure OpenAI credentials.
        Args:
            OpenAI key, endpoint url, api_version, deployement_name
        """
        self.api_key = api_key if api_key else AZURE_OPENAI_API_KEY
        self.endpoint_url = endpoint_url if endpoint_url else AZURE_OPENAI_ENDPOINT
        self.api_version = api_version if api_version else AZURE_OPENAI_API_VERSION
        self.deployment_name = deployement_name if deployement_name else AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME

        self.validate_credentials()

        self.embeddings = AzureOpenAIEmbeddings(azure_deployment=self.deployment_name, openai_api_version = self.api_version,
                                                azure_endpoint=self.endpoint_url, api_key=self.api_key)
        
    def validate_credentials(self):
        if not self.api_key:
            raise ValueError("Azure OpenAI API Key is not configured. Please update the .env file with actual credentials")
        if not self.endpoint_url:
            raise ValueError("Azure OpenAI End point is not configured. Please update the .env file with actual credentials")
        if not self.deployment_name:
            raise ValueError("Azure OpenAI Embedding deployment name not configured. Please update the .env file with actual credentials")
    
    def embed_the_query(self, text:str) -> List[float]:
        """
        Generate embedding for a query text
        Args: text: Query text to embed
        Returns: List of embedding values
        """
        return self.embeddings.embed_query(text)
    
    def embed_the_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents
        Args: List of document texts to embed
        Returns: List of embedding vectors
        """
        return self.embeddings.embed_documents(texts)
    
    def get_openai_embeddings_object(self) -> AzureOpenAIEmbeddings:
        return self.embeddings
    
    def get_openai_embeddings_dimension(self) -> int:
        embeddings_ = self.embed_the_query("test")
        return len(embeddings_)
    