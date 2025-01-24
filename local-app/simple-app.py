#!/usr/bin/env python3
from typing import Optional, Tuple, Any
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import VectorStore
from datetime import datetime
import os
from pathlib import Path

from document_processor import DocumentProcessor
from config import DOCUMENTS_DIRECTORY, INDEX_DIRECTORY, embeddings, llm, SYSTEM_PROMPT, APP_HEADER

# Use configurable system prompt
CONVERSATION_PROMPT = PromptTemplate(
    template=SYSTEM_PROMPT,
    input_variables=["context", "chat_history", "question"]
)

def initialize_conversation_chain(vectorstore: VectorStore) -> ConversationalRetrievalChain:
    """Initialize the conversation chain with memory.

    Args:
        vectorstore: The vector store containing the embedded documents

    Returns:
        ConversationalRetrievalChain: Initialized conversation chain with memory
    """
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

    # Use the system prompt from session state if available, otherwise use default
    current_prompt = st.session_state.get('system_prompt', SYSTEM_PROMPT)
    conversation_prompt = PromptTemplate(
        template=current_prompt,
        input_variables=["context", "chat_history", "question"]
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": conversation_prompt},
        verbose=True
    )

def check_if_reindex_needed() -> bool:
    """Check if reindexing is needed by comparing document and index modification times.

    Returns:
        bool: True if reindexing is needed, False otherwise
    """
    if not os.path.exists(INDEX_DIRECTORY):
        return True

    try:
        # Check all supported file types
        supported_extensions = {'.html', '.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls', '.txt', '.md', '.rtf'}
        doc_files = []
        for ext in supported_extensions:
            doc_files.extend(list(Path(DOCUMENTS_DIRECTORY).glob(f'**/*{ext}')))

        if not doc_files:
            return True

        latest_doc_mod_time = max(
            (os.path.getmtime(f) for f in doc_files),
            default=0
        )
        index_mod_time = os.path.getmtime(INDEX_DIRECTORY)
        return latest_doc_mod_time > index_mod_time

    except (OSError, ValueError) as e:
        st.error(f"Error checking index status: {str(e)}")
        return True

def start_new_chat() -> None:
    """Reset the conversation state and memory."""
    st.session_state.chat_history = []
    st.session_state.user_question = ""

    if st.session_state.vectorstore is not None:
        st.session_state.conversation_chain = initialize_conversation_chain(st.session_state.vectorstore)

    st.rerun()

def initialize_session_state() -> None:
    """Initialize session state variables."""
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'docs_dir' not in st.session_state:
        st.session_state.docs_dir = DOCUMENTS_DIRECTORY
    if 'index_dir' not in st.session_state:
        st.session_state.index_dir = INDEX_DIRECTORY
    if 'system_prompt' not in st.session_state:
        st.session_state.system_prompt = SYSTEM_PROMPT
    if 'app_header' not in st.session_state:
        st.session_state.app_header = APP_HEADER
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'conversation_chain' not in st.session_state:
        st.session_state.conversation_chain = None
    if 'doc_processor' not in st.session_state:
        st.session_state.doc_processor = DocumentProcessor(
            index_directory=INDEX_DIRECTORY,
            documents_directory=DOCUMENTS_DIRECTORY,
            embeddings=embeddings,
            llm=llm
        )
    if 'last_index_check' not in st.session_state:
        st.session_state.last_index_check = None

def analyze_document_content(doc_content: str, doc_metadata: dict) -> dict:
    """Analyze document content and provide detailed statistics.

    Args:
        doc_content: The extracted document content
        doc_metadata: Document metadata

    Returns:
        dict: Analysis results including statistics and quality metrics
    """
    # Content statistics
    total_chars = len(doc_content)
    lines = doc_content.splitlines()
    total_lines = len(lines)

    # Content quality checks
    missing_text_count = doc_content.count("<missing-text>")
    table_issues = doc_content.count("ERR#: COULD NOT CONVERT")

    # Detect section headers (lines that are short, in title case, and don't end with punctuation)
    section_headers = []
    for line in lines:
        line = line.strip()
        if (len(line) > 0 and len(line) < 50 and line.istitle()
            and not line.endswith(('.', ':', '?', '!', ','))):
            section_headers.append(line)

    # Sample content from different sections
    quarter_mark = total_lines // 4
    middle_mark = total_lines // 2
    content_samples = {
        "beginning": "\n".join(line.strip() for line in lines[:10] if line.strip()),
        "quarter": "\n".join(line.strip() for line in lines[quarter_mark:quarter_mark+5] if line.strip()),
        "middle": "\n".join(line.strip() for line in lines[middle_mark:middle_mark+5] if line.strip())
    }

    return {
        "statistics": {
            "total_characters": total_chars,
            "total_lines": total_lines,
            "section_count": len(section_headers)
        },
        "quality_metrics": {
            "missing_text_markers": missing_text_count,
            "table_conversion_issues": table_issues
        },
        "structure": {
            "section_headers": section_headers[:10],  # First 10 headers for preview
            "content_samples": content_samples
        },
        "metadata": doc_metadata
    }

def main() -> None:
    """Main application entry point."""
    initialize_session_state()
    st.title(st.session_state.app_header)

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        app_header = st.text_input("App Header", value=st.session_state.app_header)
        docs_dir = st.text_input("Documents Directory", value=st.session_state.docs_dir)
        index_dir = st.text_input("Index Directory", value=st.session_state.index_dir)
        system_prompt = st.text_area("System Prompt", value=st.session_state.system_prompt, height=300)

        if st.button("Update Configuration"):
            st.session_state.app_header = app_header
            st.session_state.docs_dir = docs_dir
            st.session_state.index_dir = index_dir
            st.session_state.system_prompt = system_prompt

            # Reinitialize conversation chain with new prompt if vectorstore exists
            if 'vectorstore' in st.session_state:
                st.session_state.conversation_chain = initialize_conversation_chain(st.session_state.vectorstore)

            # Reset conversation history
            st.session_state.chat_history = []
            st.rerun()

    # Initialize document processor with current settings
    doc_processor = DocumentProcessor(
        index_directory=st.session_state.index_dir,
        documents_directory=st.session_state.docs_dir,
        embeddings=embeddings,
        llm=llm
    )

    # Sidebar for ingestion and controls
    with st.sidebar:
        if st.button(" Start New Conversation", use_container_width=True):
            start_new_chat()

        st.markdown("---")

        reindex_needed = check_if_reindex_needed()

        if reindex_needed:
            st.warning(" Document changes detected. Please reindex.")

        if st.button("Reprocess Documents", disabled=not reindex_needed):
            ingestion_container = st.container()
            with ingestion_container:
                st.write("Starting document ingestion process...")
                vectorstore, doc_count = doc_processor.ingest_documents()

                if vectorstore is not None:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.conversation_chain = initialize_conversation_chain(vectorstore)

                    # Get processing statistics
                    stats = doc_processor.get_processing_stats()

                    # Display processing statistics
                    st.subheader("Processing Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Files", stats['total_files'])
                        st.metric("Success Rate", f"{stats['success_rate']:.1f}%")
                    with col2:
                        st.metric("Successful", stats['successful_files'])
                        st.metric("Failed", stats['failed_files'])
                    with col3:
                        st.metric("Total Characters", f"{stats['total_chars']:,}")
                        st.metric("Avg. Chars/Doc", f"{int(stats['average_chars_per_doc']):,}")

                    # Display quality metrics
                    st.subheader("Quality Metrics")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Missing Text Markers", stats['missing_text_markers'])
                    with col2:
                        st.metric("Tables Detected", stats['total_tables'])

                    # Analyze the last processed document if available
                    if hasattr(doc_processor, 'last_processed_content'):
                        analysis = analyze_document_content(
                            doc_processor.last_processed_content,
                            doc_processor.last_processed_metadata
                        )

                        # Display document analysis
                        with st.expander("Last Document Analysis", expanded=True):
                            st.subheader("Document Statistics")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Characters", f"{analysis['statistics']['total_characters']:,}")
                            with col2:
                                st.metric("Total Lines", f"{analysis['statistics']['total_lines']:,}")
                            with col3:
                                st.metric("Sections Found", analysis['statistics']['section_count'])

                            st.subheader("Document Structure")
                            with st.expander("Section Headers"):
                                for header in analysis['structure']['section_headers']:
                                    st.markdown(f"- {header}")

                            st.subheader("Content Samples")
                            with st.expander("Beginning of Document"):
                                st.text(analysis['structure']['content_samples']['beginning'])
                            with st.expander("Around 25% Mark"):
                                st.text(analysis['structure']['content_samples']['quarter'])
                            with st.expander("Around 50% Mark"):
                                st.text(analysis['structure']['content_samples']['middle'])

                    st.success(f"Successfully processed {doc_count} documents!")
                    st.session_state.last_index_check = datetime.now()
                else:
                    st.error("Failed to process documents.")

        # Display index status
        st.markdown("---")
        if st.session_state.last_index_check:
            st.info(f"Last indexed: {st.session_state.last_index_check.strftime('%Y-%m-%d %H:%M:%S')}")

        st.markdown("""
        ### About this Assistant
        This documentation assistant uses RAG (Retrieval-Augmented Generation) to:
        - Search through documentation (PDF, DOCX, HTML, etc.)
        - Maintain conversation context
        - Provide source references
        - Give accurate, contextual answers
        """)

    # Try to load existing vectorstore if not already loaded
    if st.session_state.vectorstore is None:
        try:
            vectorstore = doc_processor.get_or_create_vectorstore()
            if vectorstore is not None:
                st.session_state.vectorstore = vectorstore
                st.session_state.conversation_chain = initialize_conversation_chain(vectorstore)
                if not st.session_state.last_index_check:
                    st.session_state.last_index_check = datetime.fromtimestamp(
                        os.path.getmtime(INDEX_DIRECTORY)
                    )
        except Exception as e:
            st.error(f"Error initializing vectorstore: {str(e)}")

    # Main chat interface
    user_question = st.text_input("Ask a question about the documents:", key="user_question")

    if user_question:
        if st.session_state.conversation_chain is None:
            st.warning("Please ingest documents first using the button in the sidebar.")
        else:
            try:
                with st.spinner("Searching documentation..."):
                    response = st.session_state.conversation_chain({"question": user_question})

                    # Display answer
                    st.markdown("### Answer")
                    st.markdown(response["answer"])

                    # Display sources
                    if response.get("source_documents"):
                        with st.expander("View Sources"):
                            for idx, doc in enumerate(response["source_documents"], 1):
                                st.markdown(f"**Source {idx}:**")
                                st.markdown(doc.page_content)
                                st.markdown(f"*From: {doc.metadata.get('source', 'Unknown')} "
                                          f"(Type: {doc.metadata.get('type', 'Unknown')})*")
                                st.markdown("---")

                    # Update chat history
                    st.session_state.chat_history.append((user_question, response["answer"]))

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.warning("Please try rephrasing your question.")

    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### Recent Conversations")
        for question, answer in reversed(st.session_state.chat_history[-5:]):
            with st.expander(f"Q: {question[:50]}...", expanded=False):
                st.markdown("**Question:**")
                st.markdown(question)
                st.markdown("**Answer:**")
                st.markdown(answer)

if __name__ == "__main__":
    main()