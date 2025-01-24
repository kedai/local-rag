#!/usr/bin/env python3
import os
import json
import logging
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import streamlit as st
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from docling_parser import DoclingParser

from config import nlp

class DocumentProcessor:
    def __init__(self, index_directory: str, documents_directory: str, embeddings, llm):
        self.index_directory = Path(index_directory)
        self.documents_directory = Path(documents_directory)
        self.llm = llm.model
        self.embeddings = embeddings
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.batch_size = 50
        self.parser = DoclingParser()

        # Supported file extensions
        self.supported_extensions = {
            '.html', '.pdf', '.docx', '.doc', 
            '.pptx', '.ppt', '.xlsx', '.xls', 
            '.txt', '.md', '.rtf'
        }

        # Change cache file location to be alongside the index
        self.cache_directory = self.index_directory / "cache"
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        self.processed_cache_file = self.cache_directory / "processed_files.json"
        self.vectorstore_info_file = self.cache_directory / "vectorstore_info.json"
        self.ollama_url = "http://localhost:11434/api/generate"
        
        # Track document processing stats
        self.last_processed_content = None
        self.last_processed_metadata = None
        self.processing_stats = {
            'total_files': 0,
            'successful_files': 0,
            'failed_files': 0,
            'total_chars': 0,
            'total_tables': 0,
            'missing_text_markers': 0
        }

    def _update_processing_stats(self, content: str, success: bool = True):
        """Update document processing statistics."""
        self.processing_stats['total_files'] += 1
        if success:
            self.processing_stats['successful_files'] += 1
            self.processing_stats['total_chars'] += len(content)
            self.processing_stats['missing_text_markers'] += content.count("<missing-text>")
            self.processing_stats['total_tables'] += content.count("Table Headers:")
        else:
            self.processing_stats['failed_files'] += 1

    def _load_cache_info(self) -> Dict:
        """Load both processed files cache and vectorstore info."""
        cache_info = {
            "processed_files": {},
            "vectorstore_hash": None,
            "last_modified": None
        }

        try:
            if self.processed_cache_file.exists():
                with open(self.processed_cache_file, 'r') as f:
                    cache_info["processed_files"] = json.load(f)

            if self.vectorstore_info_file.exists():
                with open(self.vectorstore_info_file, 'r') as f:
                    vectorstore_info = json.load(f)
                    cache_info.update(vectorstore_info)

        except Exception as e:
            self.logger.error(f"Error loading cache info: {e}")

        return cache_info

    def _save_cache_info(self, processed_files: Dict, vectorstore_hash: str = None):
        """Save both processed files cache and vectorstore info."""
        try:
            with open(self.processed_cache_file, 'w') as f:
                json.dump(processed_files, f)

            vectorstore_info = {
                "vectorstore_hash": vectorstore_hash,
                "last_modified": datetime.now().isoformat()
            }
            with open(self.vectorstore_info_file, 'w') as f:
                json.dump(vectorstore_info, f)

        except Exception as e:
            self.logger.error(f"Error saving cache info: {e}")

    def _check_vectorstore_validity(self) -> bool:
        """Check if the existing vectorstore is valid and up-to-date."""
        if not os.path.exists(self.index_directory / "index.faiss"):
            return False

        try:
            cache_info = self._load_cache_info()

            if not cache_info.get("vectorstore_hash"):
                return False

            # Check for all supported file types
            current_files = {
                f for f in os.listdir(self.documents_directory) 
                if Path(f).suffix.lower() in self.supported_extensions
            }
            cached_files = set(cache_info["processed_files"].keys())

            if current_files - cached_files:
                return False

            for file_name in current_files:
                file_path = self.documents_directory / file_name
                current_modified = os.path.getmtime(file_path)
                cached_modified = cache_info["processed_files"].get(file_name)

                if not cached_modified or current_modified > cached_modified:
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking vectorstore validity: {e}")
            return False

    def _format_table(self, table: Dict) -> str:
        """Format table data in a more readable and structured way."""
        if not table.get('headers') or not table.get('rows'):
            return ""

        formatted = "Table Structure:\n"
        formatted += f"Headers: {' | '.join(table['headers'])}\n\n"

        for row in table['rows']:
            formatted += f"Row: {' | '.join(str(cell) for cell in row)}\n"

        if len(table['rows']) > 0:
            formatted += f"\nSummary: Table with {len(table['headers'])} columns and {len(table['rows'])} rows."

        return formatted

    def _analyze_table_with_ollama(self, table: Dict, context: str = "") -> str:
        """Analyze table content using Ollama to extract meaning and relationships."""
        print(f"Analyzing table: {table}")
        try:
            headers = table['headers']
            rows = table['rows']

            if any('status' in h.lower() for h in headers):
                prompt = f"""
                Analyze this API status/state table:
                Headers: {', '.join(headers)}
                Data: {str(rows)}
                Context: {context}

                Please provide:
                1. What state transitions or status codes are described
                2. Key relationships between columns
                3. Important conditions or requirements
                """
            elif any('parameter' in h.lower() or 'field' in h.lower() for h in headers):
                prompt = f"""
                Analyze this API parameter/field table:
                Headers: {', '.join(headers)}
                Data: {str(rows)}
                Context: {context}

                Please provide:
                1. Key parameters/fields and their purposes
                2. Required vs optional fields
                3. Data type requirements or constraints
                """
            else:
                prompt = f"""
                Analyze this API documentation table:
                Headers: {', '.join(headers)}
                Data: {str(rows)}
                Context: {context}

                Please provide:
                1. Main purpose of this table
                2. Key relationships between columns
                3. Important patterns or requirements
                """

            response = requests.post(
                self.ollama_url,
                headers={"Content-Type": "application/json"},
                json={
                    "model": self.llm,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )

            if response.status_code == 200:
                try:
                    full_response = ""
                    for line in response.text.strip().split('\n'):
                        try:
                            data = json.loads(line)
                            full_response += data.get('response', '')
                        except json.JSONDecodeError:
                            continue
                    return full_response.strip()
                except Exception as e:
                    self.logger.warning(f"Error parsing Ollama response: {e}")
                    return self._format_table(table)
            else:
                self.logger.warning(f"Ollama request failed with status {response.status_code}")
                return self._format_table(table)

        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Error connecting to Ollama: {e}")
            return self._format_table(table)
        except Exception as e:
            self.logger.warning(f"Unexpected error in table analysis: {e}")
            return self._format_table(table)

    def _process_file(self, file_path: str) -> List[Document]:
        """Process a single file and return a list of documents."""
        try:
            file_path = str(file_path)
            self.logger.info(f"Processing file: {file_path}")
            
            result = self.parser.parse_file(file_path)
            content = result.get('content', '')
            metadata = result.get('metadata', {})
            
            # Store the last processed document info
            self.last_processed_content = content
            self.last_processed_metadata = metadata
            
            # Update processing stats
            self._update_processing_stats(content)
            
            # Add file path and processing timestamp to metadata
            metadata.update({
                'source': file_path,
                'processed_at': datetime.now().isoformat(),
                'content_length': len(content),
                'missing_text_count': content.count("<missing-text>")
            })
            
            # Create document
            doc = Document(page_content=content, metadata=metadata)
            return [doc]
            
        except Exception as e:
            self._update_processing_stats('', success=False)
            self.logger.error(f"Error processing file {file_path}: {e}")
            return []

    def _process_documents(self, documents: List[Document]) -> List[Document]:
        """Process documents into sentence-level chunks."""
        processed_docs = []
        for doc in documents:
            if not doc.page_content.strip():
                continue

            try:
                nlp_doc = nlp(doc.page_content)
                for sentence in nlp_doc.sentences:
                    if sentence.text.strip():
                        processed_docs.append(Document(
                            page_content=sentence.text.strip(),
                            metadata=doc.metadata
                        ))
            except Exception as e:
                self.logger.error(f"Error processing document: {e}")
                continue

        return processed_docs

    def get_or_create_vectorstore(self) -> FAISS:
        """Get existing vectorstore if valid, otherwise create new one."""
        if self._check_vectorstore_validity():
            try:
                st.info("Loading existing vectorstore...")
                return FAISS.load_local(str(self.index_directory), self.embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                self.logger.error(f"Error loading existing vectorstore: {e}")

        st.warning("Need to process documents. This may take a few minutes...")
        vectorstore, _ = self.ingest_documents()
        return vectorstore

    def ingest_documents(self) -> Optional[Tuple[FAISS, int]]:
        """Ingest documents with persistent caching."""
        try:
            vectorstore = FAISS.from_documents([Document(page_content="", metadata={})], self.embeddings)
            cache_info = self._load_cache_info()
            processed_files = cache_info.get("processed_files", {})
            total_processed = 0

            progress_bar = st.progress(0)
            status_text = st.empty()

            files = [
                f for f in os.listdir(self.documents_directory)
                if Path(f).suffix.lower() in self.supported_extensions
            ]

            for idx, file_name in enumerate(files):
                file_path = self.documents_directory / file_name
                current_modified = os.path.getmtime(file_path)

                if file_name in processed_files and processed_files[file_name] == current_modified:
                    continue

                try:
                    docs = self._process_file(str(file_path))
                    processed_docs = self._process_documents(docs)
                    total_processed += len(processed_docs)

                    if processed_docs:
                        # Create a temporary vectorstore for the batch
                        batch_vectorstore = FAISS.from_documents(processed_docs, self.embeddings)
                        vectorstore.merge_from(batch_vectorstore)
                        processed_files[file_name] = current_modified

                        # Save progress periodically
                        if idx % 5 == 0 or idx == len(files) - 1:
                            vectorstore.save_local(str(self.index_directory))
                            vectorstore_hash = str(hash(frozenset(processed_files.items())))
                            self._save_cache_info(processed_files, vectorstore_hash)

                    progress = (idx + 1) / len(files)
                    progress_bar.progress(progress)
                    status_text.text(f"Processed {total_processed} documents from {idx + 1}/{len(files)} files")

                except Exception as e:
                    self.logger.error(f"Error processing {file_name}: {e}")
                    continue

            if total_processed > 0:
                return vectorstore, total_processed
            else:
                self.logger.warning("No new documents to process")
                return None, 0

        except Exception as e:
            self.logger.error(f"Error in ingest_documents: {e}")
            return None, 0

    def _update_vectorstore(self, vectorstore: FAISS, documents: List[Document]):
        """Update the vectorstore with new documents."""
        if documents:
            batch_vectorstore = FAISS.from_documents(documents, self.embeddings)
            vectorstore.merge_from(batch_vectorstore)
            vectorstore.save_local(str(self.index_directory))

    def process_documents(self, progress_bar=None) -> Optional[FAISS]:
        """Process all documents in the directory and create/update the vector store."""
        try:
            documents = []
            status_text = st.empty() if progress_bar else None

            # Get all supported files
            files = [
                f for f in os.listdir(self.documents_directory)
                if Path(f).suffix.lower() in self.supported_extensions
            ]
            
            if not files:
                self.logger.warning(f"No supported documents found in {self.documents_directory}")
                return None

            total_files = len(files)
            total_processed = 0
            processed_files = {}

            for file_name in files:
                file_path = str(self.documents_directory / file_name)
                
                if status_text:
                    status_text.text(f"Processing {file_name}...")

                try:
                    docs = self._process_file(file_path)
                    processed_docs = self._process_documents(docs)
                    documents.extend(processed_docs)
                    processed_files[file_name] = os.path.getmtime(file_path)
                    total_processed += len(processed_docs)

                    if progress_bar:
                        progress_bar.progress((len(documents) / (total_files * 5)))

                except Exception as e:
                    self.logger.error(f"Error processing file {file_name}: {e}")
                    continue

            if not documents:
                self.logger.warning("No documents were successfully processed")
                return None

            # Create vector store
            vectorstore = FAISS.from_documents(documents, self.embeddings)
            
            # Save the processed files info
            vectorstore_hash = datetime.now().isoformat()
            self._save_cache_info(processed_files, vectorstore_hash)

            # Save the vector store
            vectorstore.save_local(str(self.index_directory))

            if status_text:
                status_text.text(f"Processed {total_processed} documents from {len(files)} files.")

            return vectorstore

        except Exception as e:
            self.logger.error(f"Error in process_documents: {e}")
            if status_text:
                status_text.text(f"Error processing documents: {str(e)}")
            return None

    def get_processing_stats(self) -> Dict:
        """Get current document processing statistics."""
        return {
            **self.processing_stats,
            'success_rate': (
                self.processing_stats['successful_files'] / 
                max(self.processing_stats['total_files'], 1) * 100
            ),
            'average_chars_per_doc': (
                self.processing_stats['total_chars'] / 
                max(self.processing_stats['successful_files'], 1)
            )
        }