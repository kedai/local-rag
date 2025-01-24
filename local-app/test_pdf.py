#!/usr/bin/env python3
import logging
from docling_parser import DoclingParser
import json
import os
import traceback

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_pdf_parsing():
    try:
        parser = DoclingParser()
        logger.info("DoclingParser initialized")
        
        pdf_path = "/tmp/xx.pdf"
        logger.info(f"Attempting to parse PDF: {pdf_path}")
        
        # Add debug info about the PDF file
        logger.debug(f"PDF file size: {os.path.getsize(pdf_path)} bytes")
        
        # Try to get raw PDF info first
        with open(pdf_path, 'rb') as f:
            header = f.read(1024)
            logger.debug(f"PDF header (first 1024 bytes): {header}")
        
        result = parser.parse_file(pdf_path)
        
        if result:
            logger.info("PDF parsed successfully")
            logger.info(f"Content length: {len(result['content'])}")
            logger.info(f"First 500 chars of content: {result['content'][:500]}")
            logger.info(f"Metadata: {json.dumps(result['metadata'], indent=2)}")
            logger.info(f"Tables found: {len(result['tables'])}")
            if result['tables']:
                for i, table in enumerate(result['tables']):
                    logger.info(f"Table {i+1}:")
                    logger.info(f"  Headers: {table['headers']}")
                    logger.info(f"  First row: {table['rows'][0] if table['rows'] else []}")
            logger.info(f"Images found: {len(result['images'])}")
            if result['images']:
                for i, img in enumerate(result['images']):
                    logger.info(f"Image {i+1}: {img}")
        else:
            logger.error("Parser returned None")
            
    except Exception as e:
        logger.error("Error during PDF parsing", exc_info=True)

def test_first_10_pages():
    """Test PDF parsing on first 10 pages of a specific PDF file."""
    parser = DoclingParser()
    parser.logger.setLevel(logging.DEBUG)
    
    # Configure logging to also print to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    parser.logger.addHandler(console_handler)
    
    pdf_path = "/tmp/xx.pdf"
    print(f"\nAnalyzing PDF: {pdf_path}")
    print("=" * 80)
    
    try:
        result = parser.parse_file(pdf_path)
        content = result.get('content', '')
        metadata = result.get('metadata', {})
        
        # 1. Metadata Analysis
        print("\n1. Document Metadata:")
        print("-" * 50)
        for key, value in metadata.items():
            print(f"{key}: {value}")
        
        # 2. Content Statistics
        print("\n2. Content Statistics:")
        print("-" * 50)
        total_chars = len(content)
        total_lines = len(content.splitlines())
        print(f"Total characters: {total_chars:,}")
        print(f"Total lines: {total_lines:,}")
        
        # 3. Content Structure Analysis
        print("\n3. Content Structure Analysis:")
        print("-" * 50)
        
        # Split content into chunks for analysis
        lines = content.splitlines()
        chunk_size = total_lines // 4  # Analyze 4 different sections
        
        # Beginning of document
        print("\nBeginning of document:")
        print("-" * 30)
        for line in lines[:10]:
            if line.strip():
                print(line.strip())
        
        # Around 25% mark
        quarter_mark = total_lines // 4
        print("\nAround 25% mark:")
        print("-" * 30)
        for line in lines[quarter_mark:quarter_mark+5]:
            if line.strip():
                print(line.strip())
        
        # Around 50% mark
        middle_mark = total_lines // 2
        print("\nAround 50% mark:")
        print("-" * 30)
        for line in lines[middle_mark:middle_mark+5]:
            if line.strip():
                print(line.strip())
        
        # 4. Content Quality Checks
        print("\n4. Content Quality Analysis:")
        print("-" * 50)
        
        # Check for common OCR/extraction issues
        missing_text_count = content.count("<missing-text>")
        table_issues = content.count("ERR#: COULD NOT CONVERT")
        
        print(f"Missing text markers: {missing_text_count}")
        print(f"Table conversion issues: {table_issues}")
        
        # Look for section headers
        print("\nDetected Section Headers:")
        print("-" * 30)
        for line in lines:
            line = line.strip()
            # Heuristic: lines that are short, in title case, and don't end with punctuation
            # are likely headers
            if (len(line) > 0 and len(line) < 50 and line.istitle() 
                and not line.endswith(('.', ':', '?', '!', ','))):
                print(line)
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    test_first_10_pages()
