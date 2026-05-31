"""
- System Configuration Settings Information
- Contstants, API Key place holders, etc.,
"""

import os
import streamlit as st
from pathlib import Path
# from dotenv import load_dotenv

# this loads the env variables from .env file
_env_path = Path(__file__).parent / ".env"
# load_dotenv(_env_path)

print(f"Azure API Version {st.secrets["AZURE_API_VERSION"]}")

# Open AI credentials
AZURE_OPENAI_API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_VERSION = st.secrets["AZURE_OPENAI_API_VERSION"]

# Deployment names
AZURE_OPENAI_DEPLOYMENT_NAME = st.secrets["AZURE_OPENAI_DEPLOYMENT_NAME"]
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = st.secrets["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"]

# ---------------------------------
# APP LEVEL SETTINGS
# ---------------------------------

APP_TITLE = "🥷🏻 Samsung Smart Support Copilot System"
PAGE_TITLE = "👽 Samsung AI Support Assistant"
APP_ICON = ""

# ---------------------------------
# CHAT Settings
# ---------------------------------
# Maximum message to keep in the conversation memory
MAX_CHAT_HISTORY = 30

# ---------------------------------
# QERY CLASSIFICATION
# ---------------------------------

QUERY_TYPES = {
    "TROUBLESHOOTING": "troubleshooting",
    "COMPARISON": "comparison",
    "GENERAL" : "general"
}

# KeyWords for Rule Based Classification
CLASSIFICATION_KEYWORDS = {
    "troubleshooting" : [
        "firmware update", "overheating", "running background apps", "won't boot", "frozen",
        "error", "fix", "crash", "charging issue", "glitch", "slow", "screen issue", "troubleshoot",
        "issue", "repair", "battery drain", "not responding", "audio issue", "restart", "reset",
        "problem", "broken", "won't turn on", "black screen", "stuck", "not working", "blue screen", "blue lines"
    ],
    "comparison" : [
        "between", "difference between", "vs", "versus", "or", "compare", "difference", "compared to", "better", "which one", "against"
    ],
    "general" : [
        "what is", "how is", "how to", "information about", "can you", "explain", "learn about", "tutorial", "describe", "guide",
        "tell me", "what are"
    ]
}

# -----------------------------------
# RAG Configuration
# -----------------------------------

# Doc Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Number of documents to retrieve for context
TOP_K_RETRIEVAL = 3

#Base Dir
BASE_DIRECTORY = Path(__file__).parent

DATA_DIR = BASE_DIRECTORY / "out_directory"
UPLOADED_DOC_DIR = DATA_DIR / "all_uploaded_docs"
UPLOADED_PDF_DIR = UPLOADED_DOC_DIR 
VECTOR_DB_DIR = DATA_DIR / "vector_dbs"
SAMPLE_DOCS_DIR = BASE_DIRECTORY / "sample_docs"

# Create the Dir Structure
for dir in [DATA_DIR, UPLOADED_DOC_DIR, UPLOADED_PDF_DIR, VECTOR_DB_DIR, SAMPLE_DOCS_DIR]:
    dir.mkdir(parents=True, exist_ok=True)

# Supported file types for manual
FILET_YPE_PDF = ".pdf"
FILE_TYPE_TXT = ".txt"
FILE_TYPE_DOCX = ".docx"
ALLOWED_FILE_TYPES = [".pdf", ".txt", ".docx"]

# role and other constants
ROLE = "role"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
CONTENT = "content"
