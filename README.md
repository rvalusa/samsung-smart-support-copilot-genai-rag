# 🥷🏻 Samsung Smart Support Copilot System 

## Overview

This project is a **Streamlit + Gen AI + RAG based Intelligent Support System** that classifies user queries and provides structured context aware responses.

## Features

- **Query Classification**
     - 🛠️ Troubleshooting
     - 🧮 Comparison
     - ❓General
     - 🔁 Follow-up

- **RAG Implementation**
     - TXT, PDF and DOCX document uploading and processing
     - Intelligent chunking
     - FAISS based vectore store for similarity search
     - Context aware responses

- **Conversational Memory**
     - Maintains chat history of the user 
     - Handles user's follow-up queries as well 
     ⁃ Preserves the context across the whole conversation 

- **Structured Outputs**: 
     - Troubleshooting: Step-by-step solutions with escalation criteria too
     - Comparison: Features comparison tables with recommendations 
     - General: Direct answers with required explanations

- **Source Attribution**: Shows document sources and confidence levels

## 📁 Project Structure

```
SAMSUNG_SMART_SUPPORT_COPILOT/
│
├── app.py                          # Streamlit application entry point
├── system_config.py                # Global configuration and settings
├── requirements_samsung_support.txt # Project dependencies
├── README.md                       # Project documentation
├── .env                            # Environment variables (API keys, configs)
│
├── sample_docs/                    # Sample documents for testing and indexing
│
├── out_directory/                  # Generated outputs, logs, and exports
│
├── src/
│   │
│   ├── history_storage/
│   │   └── chat_conversation_memory.py
│   │       # Handles chat history storage and retrieval
│   │
│   ├── prompt_templates/
│   │   └── prompts.py
│   │       # System prompts and prompt engineering templates
│   │
│   ├── query_classifier/
│   │   └── classifier.py
│   │       # Classifies user queries into categories
│   │       # (RAG, General QA, Comparison, Troubleshooting, etc.)
│   │
│   ├── rag_module/
│   │   │
│   │   ├── document_processor.py
│   │   │   # Document loading, parsing, cleaning, chunking
│   │   │
│   │   ├── document_embeddings.py
│   │   │   # Embedding generation for document chunks
│   │   │
│   │   └── document_vector_storage.py
│   │       # Vector database creation and retrieval operations
│   │
│   ├── utils/
│   │   └── utilities.py
│   │       # Common helper functions and reusable utilities
│   │
│   └── __init__.py
│
└── __pycache__/                    # Python cache files

```

# Setup Instructions
### 1. Prerequisites

- Python 3.9 or higher 
- Azure OpenAI account with deployed models

### 2. Installation

```bash 
# Navigate to project directory 
cd samsung_smart_support_copilot 

# Create virtual environment 
python -m venv venv 

# Activate virtual environment 
# Windows: 
venv\Scripts \activate 

# Linux/Mac: 
venv/bin/activate 

# Install dependencies 
pip install -r requirements_samsung_support.txt
```

### 3. Configuration
1. Copy the example environment file: 
     ```bash 
     copy .env.example .env
     ```

2. Edit `.env` with your Azure OpenAI credentials:
```
AZURE_OPENAI_API_KEY=<your_actual_api_ key> 
AZURE_OPENAI_ENDPOINT=<https://your-resource-openai.azure.com/> 
AZURE _OPENAI_API_VERSION=<2024-02-15-preview> 
AZURE_ DEPLOYMENT_NAME=<your_gpt_deployment_name> 
AZURE_EMBEDDING DEPLOYMENT=<your_embedding_deployment_name>

```
### 4. Run the Application

```
bash streamlit run app.py
```
### Usage Guide

### Uploading Documents
1. Use the sidebar to upload PDF documents (manuals, FAQs, knowledge base) 
2. Click "Process Documents(s)" to index the content 
3. The system will chunk and embed the documents for retrieval and update its data below

## Learning Outcomes

* Build RAG-based applications
* Design stateful AI systems
* Integrate LLM + UI + data
* Apply GenAI in healthcare

---

## License

For educational purposes only.
