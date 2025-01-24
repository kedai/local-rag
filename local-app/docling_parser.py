#!/usr/bin/env python3
from typing import Dict, List, Optional
from bs4 import BeautifulSoup
from pathlib import Path
import mimetypes
import logging
import traceback
from docling.document_converter import DocumentConverter

class DoclingParser:
    """A wrapper class for parsing various document formats"""
    
    def __init__(self):
        """Initialize the DoclingParser"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)  # Set to DEBUG for more details
        
        # Add a console handler if none exists
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
            
        self.logger.info("Initializing DoclingParser")
        
        mimetypes.init()
        try:
            self.logger.info("Creating DocumentConverter instance")
            self.converter = DocumentConverter()
            self.logger.info("DocumentConverter initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing DocumentConverter: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
        # Register common MIME types
        self.docling_supported_types = {
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',  # docx
            'application/vnd.openxmlformats-officedocument.presentationml.presentation',  # pptx
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',  # xlsx
            'application/msword',  # doc
            'application/vnd.ms-powerpoint',  # ppt
            'application/vnd.ms-excel'  # xls
        }
    
    def parse_file(self, file_path: str) -> Dict:
        """
        Parse a file and extract its content and metadata
        
        Args:
            file_path (str): Path to the file to parse
            
        Returns:
            Dict: Dictionary containing parsed content and metadata
        """
        self.logger.info(f"Starting to parse file: {file_path}")
        
        # Ensure file exists
        if not Path(file_path).exists():
            self.logger.error(f"File not found: {file_path}")
            return self._empty_result(file_path)
        
        # Check file size
        file_size = Path(file_path).stat().st_size
        self.logger.info(f"File size: {file_size} bytes")
        
        mime_type, _ = mimetypes.guess_type(file_path)
        mime_type = mime_type or 'text/plain'
        self.logger.info(f"Detected MIME type: {mime_type}")
        
        # Use docling for supported document types
        if mime_type in self.docling_supported_types:
            self.logger.info(f"Using docling to parse {file_path}")
            try:
                # For PDF files, do some pre-checks
                if mime_type == 'application/pdf':
                    self.logger.info("Pre-checking PDF file")
                    try:
                        with open(file_path, 'rb') as f:
                            # Check if file starts with PDF signature
                            if not f.read(5).startswith(b'%PDF-'):
                                self.logger.error("File does not appear to be a valid PDF")
                                return self._empty_result(file_path)
                    except Exception as e:
                        self.logger.error(f"Error reading PDF file: {str(e)}")
                        return self._empty_result(file_path)
                
                self.logger.debug("Calling convert_all")
                docs = self.converter.convert_all([file_path])
                try:
                    doc = next(docs)  # Get the first document from the generator
                    self.logger.debug("File converted successfully")
                    
                    # Debug document attributes
                    self.logger.debug(f"Document attributes: {dir(doc)}")
                    if hasattr(doc, 'pages'):
                        self.logger.debug(f"Number of pages: {len(doc.pages)}")
                        # Debug first page attributes
                        if len(doc.pages) > 0:
                            self.logger.debug(f"First page attributes: {dir(doc.pages[0])}")
                            if hasattr(doc.pages[0], 'content'):
                                self.logger.debug(f"First page content type: {type(doc.pages[0].content)}")
                                self.logger.debug(f"First page content length: {len(doc.pages[0].content) if doc.pages[0].content else 0}")
                            if hasattr(doc.pages[0], 'cells'):
                                self.logger.debug(f"First page cells: {len(doc.pages[0].cells)}")
                                for i, cell in enumerate(doc.pages[0].cells[:5]):  # Look at first 5 cells
                                    self.logger.debug(f"Cell {i} content: {getattr(cell, 'content', 'No content')} type: {type(cell)}")
                                    self.logger.debug(f"Cell {i} attributes: {dir(cell)}")
                
                except StopIteration:
                    self.logger.error("No documents returned from convert_all")
                    return self._empty_result(file_path)
                except Exception as e:
                    self.logger.error(f"Error getting document from generator: {str(e)}")
                    return self._empty_result(file_path)
                
                # Extract content and metadata
                if hasattr(doc.document, 'export_to_text'):
                    content = doc.document.export_to_text()
                    self.logger.debug(f"Extracted text content length: {len(content)}")
                    if content:
                        self.logger.debug(f"First 200 chars: {content[:200]}")
                else:
                    content = doc.content if hasattr(doc, 'content') else ""
                
                # Try markdown export as fallback
                if not content and hasattr(doc.document, 'export_to_markdown'):
                    content = doc.document.export_to_markdown()
                    self.logger.debug(f"Extracted markdown content length: {len(content)}")
                
                # Try extracting from individual pages if no content yet
                if not content:
                    page_contents = []
                    for page in doc.pages:
                        if hasattr(page, 'cells'):
                            for cell in page.cells:
                                if hasattr(cell, 'text') and cell.text:
                                    page_contents.append(cell.text)
                        elif hasattr(page, 'content'):
                            page_contents.append(page.content)
                    
                    if page_contents:
                        content = "\n".join(page_contents)
                        self.logger.debug(f"Extracted page-level content length: {len(content)}")
                
                # Try to extract text directly from cells
                if not content:
                    cell_contents = []
                    for page in doc.pages:
                        if hasattr(page, 'cells'):
                            for cell in page.cells:
                                if hasattr(cell, 'text') and cell.text:
                                    cell_contents.append(cell.text)
                                elif hasattr(cell, 'bbox'):
                                    self.logger.debug(f"Cell bbox: {cell.bbox}")
                                    # Try to get text from the cell's bounding box
                                    if hasattr(doc.document, 'get_text_from_bbox'):
                                        bbox_text = doc.document.get_text_from_bbox(cell.bbox)
                                        if bbox_text:
                                            cell_contents.append(bbox_text)
                    
                    if cell_contents:
                        content = "\n".join(cell_contents)
                        self.logger.debug(f"Extracted cell content length: {len(content)}")
                        if content:
                            self.logger.debug(f"First 200 chars of cell content: {content[:200]}")
                
                # Try to extract text from predictions
                if not content:
                    prediction_contents = []
                    for page in doc.pages:
                        if hasattr(page, 'predictions'):
                            for pred in page.predictions:
                                if hasattr(pred, 'text') and pred.text:
                                    prediction_contents.append(pred.text)
                                elif hasattr(pred, 'content') and pred.content:
                                    prediction_contents.append(pred.content)
                    
                    if prediction_contents:
                        content = "\n".join(prediction_contents)
                        self.logger.debug(f"Extracted prediction content length: {len(content)}")
                        if content:
                            self.logger.debug(f"First 200 chars of prediction content: {content[:200]}")
                
                # Try to extract from document body elements
                if not content and hasattr(doc.document, 'body'):
                    body_contents = []
                    if hasattr(doc.document.body, 'texts'):
                        for text in doc.document.body.texts:
                            if hasattr(text, 'text'):
                                body_contents.append(text.text)
                    if hasattr(doc.document.body, 'tables'):
                        for table in doc.document.body.tables:
                            if hasattr(table, 'text'):
                                body_contents.append(table.text)
                    if body_contents:
                        content = "\n".join(body_contents)
                        self.logger.debug(f"Extracted body content length: {len(content)}")
                
                # Handle potential encoding issues with content
                if isinstance(content, bytes):
                    try:
                        self.logger.debug(f"Content is in bytes format, length: {len(content)}")
                        content = self._decode_text(content)
                        self.logger.debug("Successfully decoded content")
                    except Exception as e:
                        self.logger.error(f"Error during content decoding: {str(e)}")
                        self.logger.debug(f"First 100 bytes of content: {content[:100]}")
                        content = str(content)[2:-1]  # Remove b'' prefix/suffix
                
                self.logger.debug(f"Content extracted, length: {len(content)}")
                
                # For PDF files, verify content was extracted
                if mime_type == 'application/pdf' and not content:
                    self.logger.warning("No content extracted from PDF, checking document structure")
                    if hasattr(doc, 'pages'):
                        self.logger.info(f"Document has {len(doc.pages)} pages")
                        content = []
                        for i, page in enumerate(doc.pages):
                            self.logger.debug(f"Processing page {i+1}")
                            if hasattr(page, 'content'):
                                content.append(page.content)
                        content = '\n'.join(content)
                        self.logger.info(f"Extracted {len(content)} characters from pages")
                
                # Get document metadata
                doc_metadata = {}
                if hasattr(doc, 'metadata') and doc.metadata:
                    doc_metadata = {k: v for k, v in doc.metadata.items() if v is not None}
                
                metadata = {
                    'title': doc_metadata.get('title', Path(file_path).stem),
                    'author': doc_metadata.get('author', ''),
                    'created': doc_metadata.get('created', ''),
                    'modified': doc_metadata.get('modified', ''),
                    'file_type': Path(file_path).suffix[1:],
                    'mime_type': mime_type,
                    'source': file_path,
                    'page_count': len(doc.pages) if hasattr(doc, 'pages') else 1
                }
                
                # Add any additional metadata from the document
                for k, v in doc_metadata.items():
                    if k not in metadata:
                        metadata[k] = v
                
                self.logger.debug(f"Metadata extracted: {metadata}")
                
                # Extract tables
                tables = self._extract_tables(doc)
                
                # Extract images
                images = []
                if hasattr(doc, 'images'):
                    self.logger.debug(f"Found {len(doc.images)} images")
                    for img in doc.images:
                        image_data = {
                            'src': str(img.source) if hasattr(img, 'source') else '',
                            'alt': img.alt_text if hasattr(img, 'alt_text') else ''
                        }
                        images.append(image_data)
                
                result = {
                    'content': content,
                    'metadata': metadata,
                    'tables': tables,
                    'images': images
                }
                self.logger.info("Successfully parsed document with docling")
                return result
                
            except Exception as e:
                self.logger.error(f"Error using docling to parse {file_path}: {str(e)}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                
                # For PDF files, log additional information
                if mime_type == 'application/pdf':
                    self.logger.error("PDF parsing failed, checking document structure")
                    try:
                        with open(file_path, 'rb') as f:
                            header = f.read(1024)
                            self.logger.debug(f"PDF header: {header[:100]}")
                    except Exception as read_error:
                        self.logger.error(f"Error reading PDF header: {str(read_error)}")
                
                raise  # Re-raise to see the full error
        
        # Use BeautifulSoup for HTML
        if mime_type == 'text/html':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return self._parse_html(content, file_path)
            except Exception as e:
                self.logger.error(f"Error parsing HTML {file_path}: {str(e)}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                return self._empty_result(file_path)
        
        # Try text for plain text files
        if mime_type == 'text/plain':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return {
                    'content': content,
                    'metadata': {
                        'title': Path(file_path).stem,
                        'file_type': 'txt',
                        'mime_type': 'text/plain',
                        'source': file_path
                    },
                    'tables': [],
                    'images': []
                }
            except UnicodeDecodeError:
                self.logger.warning(f"Could not decode {file_path} as UTF-8")
                return self._empty_result(file_path)
        
        # Return empty result for unsupported files
        self.logger.warning(f"Unsupported file type for {file_path}: {mime_type}")
        return self._empty_result(file_path)

    def _decode_text(self, content: bytes) -> str:
        """
        Try to decode text content using different encodings with error handling
        """
        # List of encodings to try, in order of preference
        encodings = [
            'utf-8', 'latin1', 'cp1252', 'iso-8859-1', 
            'ascii', 'utf-16', 'utf-32', 'windows-1250',
            'windows-1251', 'windows-1252', 'windows-1253',
            'windows-1254', 'windows-1255', 'windows-1256',
            'windows-1257', 'windows-1258'
        ]
        
        # First try: strict decoding with each encoding
        for encoding in encodings:
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        
        # Second try: use 'replace' error handler with each encoding
        for encoding in encodings:
            try:
                decoded = content.decode(encoding, errors='replace')
                self.logger.debug(f"Decoded content with {encoding} using 'replace' error handler")
                return decoded
            except Exception as e:
                self.logger.debug(f"Failed to decode with {encoding}: {str(e)}")
                continue
        
        # Last resort: force decode as raw bytes
        try:
            self.logger.warning("Using raw byte decoding as last resort")
            return content.decode('ascii', errors='ignore')
        except Exception as e:
            self.logger.error(f"Failed all decoding attempts: {str(e)}")
            return str(content)[2:-1]  # Remove b'' prefix/suffix

    def _empty_result(self, file_path: str) -> Dict:
        """Return an empty result with basic metadata"""
        return {
            'content': '',
            'metadata': {
                'title': Path(file_path).stem,
                'file_type': Path(file_path).suffix[1:],
                'mime_type': mimetypes.guess_type(file_path)[0] or 'application/octet-stream',
                'source': file_path
            },
            'tables': [],
            'images': []
        }

    def _extract_tables(self, doc) -> List[Dict]:
        """Extract tables from document with error handling"""
        tables = []
        if not hasattr(doc, 'tables'):
            return tables
            
        for table in doc.tables:
            try:
                # Extract table data
                table_data = {
                    'headers': [],
                    'rows': []
                }
                
                # Try to extract headers
                try:
                    if hasattr(table, 'headers'):
                        table_data['headers'] = [str(h).strip() for h in table.headers if h]
                except Exception as e:
                    self.logger.warning(f"Failed to extract table headers: {str(e)}")
                
                # Try to extract rows
                try:
                    if hasattr(table, 'rows'):
                        table_data['rows'] = [
                            [str(cell).strip() if cell else '' for cell in row]
                            for row in table.rows if row
                        ]
                except Exception as e:
                    self.logger.warning(f"Failed to extract table rows: {str(e)}")
                
                # Only add table if it has some content
                if table_data['headers'] or table_data['rows']:
                    tables.append(table_data)
                    
            except Exception as e:
                self.logger.warning(f"Error processing table: {str(e)}")
                continue
                
        return tables

    def _parse_html(self, content: str, file_path: str) -> Dict:
        """Parse HTML content using BeautifulSoup"""
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract text content
            for script in soup(['script', 'style']):
                script.decompose()
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Extract metadata
            title = soup.title.string if soup.title else Path(file_path).stem
            
            # Extract tables
            tables = []
            for table in soup.find_all('table'):
                headers = []
                for th in table.find_all('th'):
                    headers.append(th.get_text().strip())
                
                rows = []
                for tr in table.find_all('tr'):
                    row = []
                    for td in tr.find_all('td'):
                        row.append(td.get_text().strip())
                    if row:
                        rows.append(row)
                
                tables.append({
                    'headers': headers,
                    'rows': rows
                })
            
            # Extract images
            images = []
            for img in soup.find_all('img'):
                images.append({
                    'src': img.get('src', ''),
                    'alt': img.get('alt', '')
                })
            
            return {
                'content': text,
                'metadata': {
                    'title': title,
                    'file_type': 'html',
                    'mime_type': 'text/html',
                    'source': file_path
                },
                'tables': tables,
                'images': images
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing HTML content: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return self._empty_result(file_path)
