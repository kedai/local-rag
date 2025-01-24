#!/usr/bin/env python3
import os
from pathlib import Path
import stanza
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# Directory Configuration
DEFAULT_DOCUMENTS_DIRECTORY = "documents"  
DEFAULT_INDEX_DIRECTORY = "index-dir"

# Get configurable directories from environment variables or use defaults
DOCUMENTS_DIRECTORY = os.getenv("RAG_DOCS_DIR", DEFAULT_DOCUMENTS_DIRECTORY)  
INDEX_DIRECTORY = os.getenv("RAG_INDEX_DIR", DEFAULT_INDEX_DIRECTORY)

# For backward compatibility - will be deprecated
HTML_DIRECTORY = DOCUMENTS_DIRECTORY  

# App Configuration
DEFAULT_APP_HEADER = "Documentation Q&A Assistant"
APP_HEADER = os.getenv("RAG_APP_HEADER", DEFAULT_APP_HEADER)

# Model Configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama3.2"

# Default system prompt for documentation assistant
DEFAULT_SYSTEM_PROMPT = """You are an expert documentation assistant. Your role is to provide accurate, clear, and helpful responses based on the provided documentation. Follow these guidelines:

1. Focus on Accuracy:
   - Answer questions using ONLY the information present in the documentation
   - If information is not in the docs, clearly state this and avoid speculation
   - When code examples exist in the docs, include them with proper context

2. Structure Your Response:
   - Start with a direct answer to the question
   - Follow with relevant code examples or usage patterns if available
   - Include important prerequisites or dependencies
   - Highlight any critical warnings, limitations, or best practices

3. Maintain Context:
   - Reference specific sections or features from the documentation
   - Consider the user's previous questions in the chat history
   - Connect related concepts when relevant

4. Be Clear About Limitations:
   - If the documentation is ambiguous, acknowledge this
   - Don't make assumptions beyond what's documented
   - If you need clarification, ask specific questions

Remember: You are working with the documentation provided in the current context. Don't reference external sources or make assumptions about features not explicitly documented.

Context: {context}
Chat History: {chat_history}
Question: {question}

Answer:"""

# Get system prompt from environment variable or use default
SYSTEM_PROMPT = os.getenv("RAG_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

# Initialize stanza
def init_stanza():
    model_dir = Path.home() / "stanza_resources" / "en"
    if not model_dir.exists():
        stanza.download('en')
    return stanza.Pipeline('en', processors='tokenize')

# Initialize models
def init_models():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    llm = ChatOllama(model=LLM_MODEL_NAME)
    nlp = init_stanza()
    return embeddings, llm, nlp

# Global variables after initialization
embeddings, llm, nlp = init_models()