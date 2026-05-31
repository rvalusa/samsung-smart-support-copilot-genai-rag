"""
Smart Samsung Support Co-pilot Main Application file, where the actual execution starts
"""

import os, sys, traceback
from pathlib import Path
from typing import Optional, List, Dict
import streamlit as st

# Import the system level configurations
from system_config import(
    AZURE_OPENAI_API_KEY, 
    AZURE_OPENAI_ENDPOINT, 
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT_NAME,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
    ALLOWED_FILE_TYPES, VECTOR_DB_DIR, UPLOADED_DOC_DIR, UPLOADED_PDF_DIR, 
    TOP_K_RETRIEVAL, APP_TITLE, PAGE_TITLE, APP_ICON)

from system_config import ROLE, ROLE_USER, ROLE_ASSISTANT, CONTENT

# Import other modules
from src.rag_module.document_processor import DocumentLoaderAndParser
from src.rag_module.document_embeddings import DocumentEmbeddingModule
from src.rag_module.document_vector_storage import FAISS_Vector_Storage
from src.query_classifier.classifier import SamsungQueryClassifier, QueryType
from src.prompt_templates.prompts import SupportPromptTemplates
from src.history_storage.chat_conversation_memory import ChatConversationHistory
from src.utils.utilities import (
    validate_file_type, get_clean_response, show_welcome_message,
    get_confidence_level_of_response, get_timestamp, format_sources_display
)

from langchain_openai import AzureChatOpenAI

# Adding project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------
# Page Configuration 
# ----------------------

st.set_page_config(
    page_title=PAGE_TITLE, 
    # page_icon="",
    layout='wide',
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .query-type-badge {
        display: inline-block;
        padding: 0.25rem 0.75 rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    .troubleshooting-badge {
        background-color: #ffebee;
        color: #c62828;
    }
    .comparison-badge {
        background-color: #e3f2fd;
        color: #1565c0;
    }
    .general-badge {
        background-color: #e8f5e9;
        color: #ef6c00;
    }
    .follow-up-badge {
        background-color: #fff3e0;
        color: #ef6c00;
    }
    .source-info {
        font-size: 0.85rem;
        color: #666;
        font-style: italic;
        margin-top: 0.5rem;
        background-color: #f5f5f5;
        border-radius: 0.25rem;
    }
    .confidence-high { color: #2e7d32; }
    .confidence-medium { color: #f57c00; }
    .confidence-low { color: #c62828; }
    .stChatMessage {
        padding: 1rem;
    }
    .sidebar-section {
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Session State Initialization 
# -----------------------------
def initialize_session_state():
    """ This method will initialize all session state variables """

    # llm client
    if "llm_client" not in st.session_state:
        st.session_state.llm_client = None

    # for Query Classifier
    if "query_classifier" not in st.session_state:
        st.session_state.query_classifier = SamsungQueryClassifier(llm_client=None, is_llm_fallback_needed=False)

    # for Document
    if "doc_processor" not in st.session_state:
        st.session_state.doc_processor = DocumentLoaderAndParser()

    # for memory, hisotyr
    if "chat_conv_memory" not in st.session_state:
        st.session_state.chat_conv_memory = ChatConversationHistory()
    
    # for vectore store
    if "vector_store_mngr" not in st.session_state:
        st.session_state.vector_store_mngr = None

    # for parsed document tracking
    if "processed_docs" not in st.session_state:
        st.session_state.processed_docs = []
    
    # for chat messages for GUI
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # for processing status in GUI
    if "is_process_running" not in st.session_state:
        st.session_state.is_process_running = False

# -------------------
# LLM Initialization 
# -------------------

def initialize_LLM():
    """ this will initializes Azure OpenAI LLM client """
    try:
        llm_client = AzureChatOpenAI(api_key=AZURE_OPENAI_API_KEY,
                              azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
                              azure_endpoint=AZURE_OPENAI_ENDPOINT,
                              openai_api_version = AZURE_OPENAI_API_VERSION,
                              temperature=0.7)
        return llm_client
    except Exception as e:
        st.error(f"Failed to initialize LLM client : {str(e)}")
        return None

# ----------------------------
# Vector Store Initialization 
# ----------------------------

def initialize_FAISS_vectore_store():
    """ this will initializes vectore store """
    try:
        embedding_mngr = DocumentEmbeddingModule()
        FAISS_vector_store_mngr = FAISS_Vector_Storage(embedding_mgr=embedding_mngr)
        return FAISS_vector_store_mngr
    except ValueError as e:
        st.warning(f"Embedding manager initialization failed ! {str(e)}")
        return None
    except Exception as e:
        st.error(f"FAISS vector store initialization failed! {str(e)}")
        return None

# ---------------------
# Process the Documents 
# ----------------------

def process_uploaded_documents(uploaded_docs: List) -> Dict:
    """ this will process uploaded documents 
    Args: uploaded_docs: List of file objects uploaded
    Returns: Dictionary with processed results 
    """
    results = {
        "success": False,
        "message": "",
        "chunks_created": 0,
        "files_processed": []
    }

    # Check if files are uploaded
    if not uploaded_docs:
        results["message"] = "No files are provided [or] Uploaded"
        return results
    
    # file type check
    for file in uploaded_docs:
        if not validate_file_type(file.name):
            results["message"] = f"Invalid file type found: {file.name}. \n Supported file types are: {", ".join(ALLOWED_FILE_TYPES)}"
            return results
    
    try:
        # check vector store initialization
        if st.session_state.vector_store_mngr is None:
            st.session_state.vector_store_mngr = initialize_FAISS_vectore_store()

            if st.session_state.vector_store_mngr is None:
                results["message"] = "Vector store initialization FAILED! Check Azure OpenAI Configuration and Try Again..."
                return results

        # process docs
        total_chunks = []
        for doc in uploaded_docs:
            st.write(f"📄 Processing file: {doc.name}")
            processed_result = (st.session_state.doc_processor.process_uploaded_file(doc,save_dir=UPLOADED_DOC_DIR))

            if processed_result is None:
                st.warning(f"⚠️ No result returned for file: {doc.name}")
                continue

            chunks, file_path = processed_result

            st.write(f"File Path: {file_path}")
            st.write(f"Chunk Type: {type(chunks)}")

            if chunks is None:
                st.warning(f"⚠️ No chunks generated for file: {doc.name}")
                continue

            if not isinstance(chunks, list):
                st.warning(
                    f"⚠️ Expected list of chunks but got {type(chunks)} "
                    f"for file: {doc.name}"
                )
                continue

            total_chunks.extend(chunks)
            results["files_processed"].append(doc.name)
        
                # Ensure chunks exist before indexing
        if not total_chunks:
            results["message"] = (
                "No document chunks were generated. "
                "Please check document content and processing logic."
            )
            return results

        # Adding documents to vector store
        st.session_state.vector_store_mngr.add_documents_to_vectore_store(total_chunks)


        results["success"] = True
        results["chunks_created"] = len(total_chunks)
        results["message"] = f" {len(results['files_processed'])} Document(s) Processed Successfully 🎉 into {len(total_chunks)} chunks🧮. "

        # Track processed files safely
        if "processed_docs" not in st.session_state:
            st.session_state.processed_docs = []

        # Track processed files
        st.session_state.processed_docs.extend(results["files_processed"])
        st.session_state.processed_docs = list(set(st.session_state.processed_docs))

    except Exception as e:
        traceback.print_exc()  # Full traceback in terminal
        st.exception(e)        # Full traceback in Streamlit UI
        results["message"] = f"Error occurred while processing documents: {str(e)}"

    return results

# ------------------------------
# Processing the Customer Query 
# ------------------------------

def process_customer_query(query: str) -> Dict:
    """ this will process customer query throught the full pipeline 
    Args: customer query
    Returns: Dictionary with response and metadata
    """
    result = {
        "query_type": QueryType.GENERAL,
        "response": "",
        "confidence": 0.0,
        "sources": [],
        "reasoning": ""
    }

    try:
        # 1. query classification
        conv_history = st.session_state.chat_conv_memory.get_all_chat_messages()
        q_type, confidence, reasoning = st.session_state.query_classifier.classify(query, conv_history)

        print("-"*80)
        print("QUERY :", query)
        print("QUERY TYPE:", q_type)
        print("CONFIDENCE:", confidence)
        print("REASON", reasoning)
        print("-"*80)

        result["query_type"] = q_type
        result["confidence"] = confidence
        result["reasoning"] = reasoning

        # 2. retrieving context from vector store
        context = ""
        sources = []

        if st.session_state.vector_store_mngr and st.session_state.vector_store_mngr.is_vector_store_initialized():
            context, sources = st.session_state.vector_store_mngr.get_context_for_the_query(query, k=TOP_K_RETRIEVAL)
            result["sources"] = sources

        # 3. Check LLM and initialize if needed
        if st.session_state.llm_client is None:
            st.session_state.llm_client = initialize_LLM()

            # update query classifier with LLM
            if st.session_state.llm_client is not None:
                st.session_state.query_classifier.llm_client = st.session_state.llm_client
                st.session_state.query_classifier.is_llm_fallback_needed = True
            
        if st.session_state.llm_client is None:
            result["response"] = "We are Sorry 🥺.  Unable to connect to the AI Service. Please check Azure OpenAI Configuration again..."
            return result
        
        # 4. Generate response using prompt template
        conv_history_str = st.session_state.chat_conv_memory.get_formatted_history()

        if context:
            prompt = SupportPromptTemplates.get_prompt_based_on_query_type(query_type=q_type, query=query, 
                                                                           context=context, conversation_history=conv_history_str)
        else:
            prompt = SupportPromptTemplates.get_prompt_when_no_context_available(query=query, query_type=q_type)
        
        # 5. Getting response from LLM
        response = st.session_state.llm_client.invoke(prompt)
        result["response"] = response.content if hasattr(response, "content") else str(response)

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        result["response"] = f"We are Sorry, An error occurred while processing your query:\n\n **Error** {str(e)}\n\n**Details:**\n```\n{error_detail}\n```"
    
    return result


# ---------------------
# GUI Components 
# ----------------------

def show_sidebar_in_gui():
    """ Rendering the sidebar with doc upload and other settings. """

    with st.sidebar:

        # LLM Config status
        st.markdown("### ⚙️ LLM Configuration Status")

        # Check OpenAI configuration
        configuration_issues = []
        if AZURE_OPENAI_API_KEY is None:
            configuration_issues.append("API Key is not configured.")
        if AZURE_OPENAI_ENDPOINT is None:
            configuration_issues.append("Endpoint is not configured.")
        if AZURE_OPENAI_DEPLOYMENT_NAME is None:
            configuration_issues.append("Deployment name is not configured.")
        if AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME is None:
            configuration_issues.append("Embedding deployment is not configured.")

        if configuration_issues:
            st.warning("⚠️ Configuration is incomplete!")
            for issue in configuration_issues:
                st.markdown(f"- {issue}")
            st.markdown("📝 Update `.env` file with Azure OpenAI configruation details.")
        else:
            st.success("✅ Configuration is successfully completed.")

        st.divider()

        st.markdown("### 📚 Document Parser and Uploader:")

        # file uploading logic - PDF/TXT/DOCX - UI
        uploaded_docs = st.file_uploader("Upload Documents (PDF/TXT/DOCX)",
                                         type=["pdf", "txt", "docx"],
                                         accept_multiple_files=True,
                                         help="Upload Product Manuals, FAQs, and internal knowledge bases in the form of PDFs, TXTs and/or DOCXs"
                                         )

        # upload and process
        if uploaded_docs and st.button("Process Document(s)", type="primary"):
            with st.spinner("🔎 Processing uploaded documents..."):
                results = process_uploaded_documents(uploaded_docs)

                if results["success"]:
                    st.success(results["message"])
                    st.info(f"🗐 Chunks are created: {results["chunks_created"]}")
                else:
                    st.error(results["message"])
        
        st.divider()

        # showing processed files UI
        if st.session_state.processed_docs:
            st.markdown("### 📚 Processed Documents: ")
            for file_name in st.session_state.processed_docs:
                st.markdown(f"✔️ {file_name}")
            
            # doc count
            if st.session_state.vector_store_mngr:
                doc_count = st.session_state.vector_store_mngr.get_total_document_count_in_store()
                st.info(f"🧾 Total chunks in the index: {doc_count}")

        st.divider()

        # clear chat button UI 
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_conv_memory.clear()
            st.session_state.chat_messages = []
            st.rerun()

        # clear vectore store button UI
        if st.session_state.vector_store_mngr and st.session_state.vector_store_mngr.is_vector_store_initialized():
            if st.button("🗑️ Clear Document Indexing"):
                st.session_state.vector_store_mngr.clear_vectore_store()
                st.session_state.processed_docs = []
                st.rerun()


def show_chat_messages(message: Dict, query_type: Optional[QueryType] = None,
                       confidence: float = 0.0, sources: List[Dict] = None):
    """ Render chat messages with UI """
    role = message[ROLE]
    content = message[CONTENT]

    with st.chat_message(role):
        # for assistant messages
        if role == ROLE_ASSISTANT and query_type:
            badge = f"{query_type.value}~Badge"
            badge_text = {
                QueryType.TROUBLESHOOTING: f"🛠️ Troubleshooting",
                QueryType.COMPARISON: f"🧮 Comparison",
                QueryType.GENERAL: f"❓General",
                QueryType.FOLLOW_UP: f"🔁 Follow-up"
            }.get(query_type, f"❓General")

            st.markdown(f"<span class='query-type-badge {badge}'>{badge_text} </span>",
                    unsafe_allow_html=True)
        
        # show the message content
        st.markdown(content)

        # Show content sources and confidence for assistant messages
        if role == ROLE_ASSISTANT:
            c1, c2 = st.columns([3, 1])

            with c1:
                if sources:
                    source_text = format_sources_display(sources)
                    st.markdown(f"<div class='source-info'>{source_text}</div>",
                                unsafe_allow_html=True)
            
            with c2:
                confidence_info = get_confidence_level_of_response(confidence)
                color = f"confidence-{confidence_info['color']}"
                st.markdown(f"**Confidence:** {confidence_info['icon']} {confidence_info['level']}")

def show_chat_interface():
    """this will render chat interface"""
    # Header
    st.markdown(f"<p style='font-size:40px; font-weight:bold;'>{APP_TITLE.upper()} {APP_ICON}</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"<p style='font-size:20px; font-weight:bold;'>👽 Welcome to AI Based Samsung Smart Support Copilot !</p>", unsafe_allow_html=True)
    
    # welcome message
    if not st.session_state.chat_messages:
        st.markdown(show_welcome_message())

    st.markdown(f"<p style='font-size:16px; font-weight:bold;'>How can I assist you today ?</p>", unsafe_allow_html=True)

    # show chat messages
    for i, msg in enumerate(st.session_state.chat_messages):
        # metadata for assistance
        query_type = None
        confidence = 0.0
        sources = []

        if msg[ROLE] == ROLE_ASSISTANT:
            # try getting metadata
            if i//2 < len(st.session_state.chat_conv_memory.query_types):
                query_type_str = st.session_state.chat_conv_memory.query_types[i//2]
                query_type = QueryType(query_type_str) if query_type_str else None
            if i//2 < len(st.session_state.chat_conv_memory.sources_used):
                sources = st.session_state.chat_conv_memory.sources_used[i//2]
        
        show_chat_messages(msg, query_type, confidence, sources)
    
    # chat input
    if prompt := st.chat_input("Ask a question..."):
        # add user message
        st.session_state.chat_messages.append({"role":"user", "content": prompt})
        st.session_state.chat_conv_memory.add_chat_messages(ROLE_USER, prompt)

        # show user name
        with st.chat_message(ROLE_USER):
            st.markdown(prompt)

        # process the user query
        with st.chat_message(ROLE_ASSISTANT):
            with st.spinner("Processing query..."):
                result = process_customer_query(prompt)

            # show query type UI badge
            query_type = result["query_type"]
            badge = f"{query_type.value}~Badge"
            badge_text = {
                QueryType.TROUBLESHOOTING: f"🛠️ Troubleshooting",
                QueryType.COMPARISON: f"🧮 Comparison",
                QueryType.GENERAL: f"❓General",
                QueryType.FOLLOW_UP: f"🔁 Follow-up"
            }.get(query_type, f"❓General")

            st.markdown(f"<span class='query-type-badge {badge}'>{badge_text} </span>",
                    unsafe_allow_html=True)

            # show response
            st.markdown(result["response"])

            # display sources and confidence
            c1, c2 = st.columns([3, 1])

            with c1:
                if result["sources"]:
                    source_text = format_sources_display(result["sources"])
                    st.markdown(f"<div class='source-info'>{source_text}</div>",
                                unsafe_allow_html=True)

            with c2:
                confidence_info = get_confidence_level_of_response(result["confidence"])
                st.markdown(f"**Confidence:** {confidence_info['icon']} {confidence_info['level']}")

        # add assistant message to history
        st.session_state.chat_messages.append({"role": "assistant", "content": result["response"]})
        st.session_state.chat_conv_memory.add_chat_messages(
            "assistant", result["response"], query_type=result["query_type"].value, sources=result["sources"]
        )

# --------------------------
# Main App Launch Point
# --------------------------

def main():
    """ this is main app entry point """
    
    # init session state
    initialize_session_state()

    # show streamlit sidebar
    show_sidebar_in_gui()

    # show chat interface
    show_chat_interface()

if __name__ == "__main__":
    main()


