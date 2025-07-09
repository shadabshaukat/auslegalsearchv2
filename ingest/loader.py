"""
Loader module for the 'ingest' component of auslegalsearchv2.

- Walks directories/files for ingestion
- Detects and parses legal document formats (.txt, .html)
- Chunks documents for embedding
- Designed for extensibility to PDF, DOCX, etc.
"""

import os
from pathlib import Path
from typing import Iterator, Dict, Any, List

from bs4 import BeautifulSoup

SUPPORTED_EXTS = {'.txt', '.html'}

def walk_legal_files(root_dirs: List[str]) -> Iterator[str]:
    """Yield supported legal document filepaths (recursively)."""
    for root_dir in root_dirs:
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                ext = Path(fname).suffix.lower()
                if ext in SUPPORTED_EXTS:
                    yield os.path.join(dirpath, fname)

def parse_txt(filepath: str) -> Dict[str, Any]:
    """Parse plain text legal file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        return {"text": text, "source": filepath, "format": "txt"}
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return {}

def parse_html(filepath: str) -> Dict[str, Any]:
    """Extract visible text from legal HTML file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            html = f.read()
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
            tag.decompose()
        raw_text = soup.get_text(separator='\n', strip=True)
        return {"text": raw_text, "source": filepath, "format": "html"}
    except Exception as e:
        print(f"Error parsing HTML {filepath}: {e}")
        return {}

def chunk_document(doc: Dict[str, Any], chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Split large documents into overlapping text chunks.
    Returns: List of chunk dicts (text, meta)
    """
    text = doc.get("text", "")
    meta = {k: v for k, v in doc.items() if k != "text"}
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        chunks.append({"text": chunk_text, **meta, "start_pos": start, "end_pos": end})
        if end == len(text):
            break
        start = end - overlap
    return chunks

# TODO: Extend with parse_pdf, parse_docx as needed
