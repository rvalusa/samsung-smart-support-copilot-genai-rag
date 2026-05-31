"""
THis is responsible for handling FAISS vector store and provide relevant context
"""

import os, sys
from typing import List, Optional, Tuple
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from system_config import VECTOR_DB_DIR, TOP_K_RETRIEVAL
from src.rag_module.document_embeddings import DocumentEmbeddingModule

sys.path.append(str(Path(__file__).parent.parent.parent))

class FAISS_Vector_Storage:
    """
    This will managers FAISS vector store for document retreival
    """
    def __init__(self, embedding_mgr: Optional[DocumentEmbeddingModule] = None, persistent_directory: Optional[Path] = None):
        """
        This constructor initializes vector storage data
        Args: 
            embedding_mgr: Embedding module instance for generating embeddings
            persistent_directory: directory to persist the vectore store
        """
        self.embedding_mgr = embedding_mgr
        self.persistent_directory = persistent_directory
        self.faiss_vector_store: Optional[FAISS] = None
        self.document_sources: List[str] = []

    def initialize_embedding_module_manager(self, **kwargs):
        """
        Initializes the embedding module instance
        Args: **kwargs : Arguments the can be passed to embedding instance
        """
        if self.embedding_mgr is None:
            self.embedding_mgr = DocumentEmbeddingModule(**kwargs)

    def create_FAISS_vector_store(self, documents: List[Document]) -> FAISS:
        """
        This creates FAISS vector store from given documents
        Args: LIst of Document objects to index
        Returns: FAISS instance
        """
        if not documents:
            raise ValueError("Error!! Vector store cannot be created from empty document list !!!")
        
        self.initialize_embedding_module_manager()
        
        self.faiss_vector_store = FAISS.from_documents(documents=documents, 
                                                       embedding=self.embedding_mgr.get_openai_embeddings_object())
        
        self.document_sources = list(set(d.metadata.get("source", "Unknown") for d in documents))

        return self.faiss_vector_store
    
    def add_documents_to_vectore_store(self, documents: List[Document]) -> None:
        """
        This adds documents to existing vector store or creates new 
        Args: documents: List of Document objects for adding
        """
        if not documents:
            return
        if self.faiss_vector_store is None:
            self.create_FAISS_vector_store(documents)
        else:
            self.faiss_vector_store.add_documents(documents)

            new_sources = list(set(d.metadata.get("source", "Unknown") for d in documents))
            self.document_sources = list(set(self.document_sources + new_sources))

    def perform_similarity_search(self, query: str, k: int = TOP_K_RETRIEVAL) -> List[Document]:
        """
        Peroforms similarity search to find relevant documents
        Args: 
            query: Query string
            k: Number of docs to retrieve
        Returns: List of relevant Document objects 
        """
        if self.faiss_vector_store is None:
            raise ValueError("Error!! Vector store is not initialized. Please add some documents (PDF/TXT/DOCX) >>> ")
        
        return self.faiss_vector_store.similarity_search(query, k=k)
    
    def perform_similarity_search_with_score(self, query: str, k: int = TOP_K_RETRIEVAL) -> List[Tuple[Document, float]]:
        """
        Performs similarity search with relevance scores
        Args: 
            query: Query string
            k : Number of docs to retrieve
        Returns: List of tuples containing Document and score together
        """
        if self.faiss_vector_store is None:
            raise ValueError("Error!! Vector store is not initialized. Please add some documents (PDF/TXT/DOCX) !!!")
        
        results = self.faiss_vector_store.similarity_search_with_score(query, k=k)
        
        print("\n")
        print("-"*80)
        print("QUERY:", query)

        for doc, score in results:
            print("SCORE:", score)
            print(doc.page_content[:200])
        print("-"*80)
        
        return results


    def get_vector_store_retriever(self, search_kwargs: Optional[dict] = None):
        """
        To get retriever instance for the vectore store.
        Args: search_kwargs: search parameters for the retriever
        Returns: VectorStoreRetriever instance
        """
        if self.faiss_vector_store is None:
            raise ValueError("Error!! Vector store is not initialized. Please add some documents (PDF/TXT/DOCX) ...")
        
        if search_kwargs is None:
            search_kwargs = {"k": TOP_K_RETRIEVAL}

        return self.faiss_vector_store.as_retriever(search_kwargs=search_kwargs)
    
    def save_faiss_vector_store(self, name: str= "default") -> str:
        """
        save the vector store to disk
        Args: name: Name of the vector store to be saved
        Returns: Path where the vector store saved
        """
        if self.faiss_vector_store is None:
            raise ValueError("Error!! There is No vector store to Save !!!")

        saved_store_path = self.persistent_directory / name
        self.faiss_vector_store.save_local(str(saved_store_path))

        return str(saved_store_path)
    
    def load_faiss_vector_store(self, name: str = "default") -> FAISS:
        """
        Load the FAISS vector store 
        Args: name: Name of the saved vector store
        Returns: FAISS vectore store instance
        """
        self.initialize_embedding_module_manager()

        load_store_path = self.persistent_directory / name

        if not load_store_path.exists():
            raise ValueError(f"Error!! Vector store not found at the path {load_store_path}")
        
        self.faiss_vector_store = FAISS.load_local(folder_path=str(load_store_path),
                                                   embeddings=self.embedding_mgr.get_openai_embeddings_object(),
                                                   allow_dangerous_deserialization=True)
        
        return self.faiss_vector_store
    
    def get_context_for_the_query(self, query: str, k: int = TOP_K_RETRIEVAL) -> Tuple[str, List[dict]]:
        """
        Get the formatted context for a query with source information
        Args: query: query string, k : Number of docs to retrieve
        Return: Tuple formatted context string and list of source metada
        """
        result_with_scores = self.perform_similarity_search_with_score(query, k=k)

        context__ = []
        sources__ = []

        for i, (doc, score) in enumerate(result_with_scores):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")

            context__.append(f"[Document {i+1}] --> Source is : {source}, Page: {page}")
            context__.append(doc.page_content)
            context__.append("*"*60)

            sources__.append({
                "source": source,
                "page": page, 
                "relevance_score": float(score),
                "content_preview": doc.page_content[:150] + "..."
            })

        context = "\n".join(context__)

        return context, sources__
        
    def get_total_document_count_in_store(self) -> int:
        """
        Get the total number of documents in the vector store
        Returns : Number of documents
        """
        if self.faiss_vector_store is None:
            return 0
        return self.faiss_vector_store.index.ntotal
    
    def is_vector_store_initialized(self) -> bool:
        """
        Check whether the vector store initialized or not
        Returns: True if initialized, otherwise False
        """
        return self.faiss_vector_store is not None
    
    def clear_vectore_store(self):
        """ This will clear the vectore store content """
        self.faiss_vector_store = None
        self.document_sources = []

