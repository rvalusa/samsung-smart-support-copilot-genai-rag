import os, sys
from typing import List, Optional
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from system_config import FILET_YPE_PDF, FILE_TYPE_TXT, FILE_TYPE_DOCX, UPLOADED_DOC_DIR, ALLOWED_FILE_TYPES
from system_config import CHUNK_SIZE, CHUNK_OVERLAP

class DocumentLoaderAndParser:
    """
    This class will handle document loading, parsing and chunking for RAG pipeline
    This supports PDF, TXT, DOCX
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap : int = CHUNK_OVERLAP):
        """
        This method is a constructor which initializes the loader and parser data
        Args:
            chunk_size: Max size of each text chunk
            chunk_overlap: Overlap between two consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter( chunk_size=chunk_size, chunk_overlap=chunk_overlap, 
                                                            length_function = len, separators=["\n\n", "\n", "", ".", " ", ])
        
    def get_uploaded_file_extension(self, file_path:str) -> str:
        """
        Get the file extension in lowercase
        Args: 
            file_path: Path of the file.
        Returns:
            extension of the file in lowercase
        """
        return Path(file_path).suffix.lower()
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Loads a PDF file and extracts its text
        Args: 
            file_path: Path of the file
        Return:
            List of Document type objects
        """
        pdfloader = PyPDFLoader(file_path)
        docs = pdfloader.load()

        # adding filename to metadata
        filename = os.path.basename(file_path)
        for doc in docs:
            doc.metadata["source"] = filename
            doc.metadata["file_path"] = file_path
            doc.metadata["file_type"] = "pdf"

        return docs

    def load_txt(self, file_path: str) -> List[Document]:
        """
        Loads a TXT file and extracts its text
        Args: 
            file_path: Path of the file
        Return:
            List of Document type objects
        """
        txtloader = TextLoader(file_path, encoding="utf-8")
        docs = txtloader.load()
        
        # adding filename to metadata
        filename = os.path.basename(file_path)
        for doc in docs:
            doc.metadata["source"] = filename
            doc.metadata["file_path"] = file_path
            doc.metadata["file_type"] = "txt"
            doc.metadata["page"] = 0
        
        return docs


    def load_docx(self, file_path: str) -> List[Document]:
        """
        Loads a DOCX file and extracts its text
        Args: 
            file_path: Path of the file
        Return:
            List of Document type objects
        """
        docloader = Docx2txtLoader(file_path)
        docs = docloader.load()

        # adding filename to metadata
        filename = os.path.basename(file_path)
        for doc in docs:
            doc.metadata["source"] = filename
            doc.metadata["file_path"] = file_path
            doc.metadata["file_type"] = "docx"        
        
        return docs

    def load_uploaded_document(self, file_path:str) -> List[Document]:
        """
        This method will load the document and extract its text.
        Allowed document types are PDF, TXT, DOCX
        Args:  
            file_path: Path of the document
        Returns:
            List of Document objects
        Raises:
            ValueError: if the unspported file type is uploaded
        """
        ext = self.get_uploaded_file_extension(file_path)

        if ext == FILET_YPE_PDF:
            return self.load_pdf(file_path)
        elif ext == FILE_TYPE_TXT:
            return self.load_txt(file_path)
        elif ext == FILE_TYPE_DOCX:
            return self.load_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type .{ext} found...")
        
    def load_multiple_uploaded_documents(self, file_paths: List[str]) -> List[Document]:
        """
        This method will load multiple documents like PDF, TXT, DOCX
        Args:
            file_paths: list of file paths
        Returns:
            List of Document objects
        """
        all_docs = []
        for file_path in file_paths:
            try:
                docs = self.load_uploaded_document(file_path)
                all_docs.extend(docs)
            except Exception as e:
                print(f"Exception occurred in loading {file_path} => {str(e)}")
        
        return all_docs

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split the documents in to smaller chunks for creating embeddings
        Args:
            documents: LIst of Document objects to chunk
        Returns:
            List of chunked Document objects
        """
        chunks = self.text_splitter.split_documents(documents)

        # adding chunk index to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_index'] = i
        
        return chunks
    
    def process_uploaded_file(self, uploaded_file, save_dir: Optional[Path] = None) -> tuple[List[Document], str]:
        """
        Process an uploaded file from Streamlet file uploader GUI
        (PDF, TXT, DOCX)
        Args:
            Uploaded_file : Streamlit uploadedFile obj
            save_dir : Directory where uploaded file is saved
        Returns:
            Tuple of chunked docs, saved file path
        """
        if save_dir is None:
            save_dir = UPLOADED_DOC_DIR
        
        # Save the uploaded file
        save_path = save_dir / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # load and chunk docs
        documents = self.load_uploaded_document(str(save_path))
        if documents is None:
            raise ValueError(f"No documents loaded from file: {save_path}")
        
        chunked_docs = self.chunk_documents(documents)

        return chunked_docs, str(save_path)
    
    def process_directory_of_files(self, directory_path: str) -> List[Document]:
        """
        Process all supported document files in a directory
        Args:
            directory_path: folder path which contains documents
        Returns:
            List of chunked Document objects
        """
        directory = Path(directory_path)

        all_supported_files = []
        for extnesion in ALLOWED_FILE_TYPES:
            all_supported_files.extend(directory.glob(f"*{extnesion}"))

        if not all_supported_files:
            print(f"No Supported document files found in {directory_path}")
            return []
        
        all_supported_documents = self.load_multiple_uploaded_documents([str(f) for f in all_supported_files])
        chunked_documents = self.chunk_documents(all_supported_documents)

        print(f"Processed {len(all_supported_files)} document file(s) into {len(chunked_documents)} chunks")
        return chunked_documents
    
    def get_document_details(self, documents: List[Document]) -> dict:
        """
        Get the details of the processed document
        Args:
            documents: List of Document objects
        Returns:
            Dictionary with document details
        """
        if not documents:
            return {"total_chunks":0, "total_characters":0, "sources":[]}
        
        total_characters = sum(len(doc.page_content) for doc in documents)
        sources = list(set(doc.metadata.get("source", "Unknown") for doc in documents))
        file_types = list(set(doc.metadata.get("file_type", "Unknown") for doc in documents))

        return {
            "total_chunks": len(documents),
            "total_characters": total_characters,
            "avg_chunk_size": total_characters // len(documents) if documents else 0,
            "sources": sources,
            "file_types": file_types
        }
