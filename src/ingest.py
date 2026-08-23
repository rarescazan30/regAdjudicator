import os
import re
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from chromadb.utils import embedding_functions

from src.config import (
    DOCS_DIR,
    DRUGS_DOCS_DIR,
    GUIDANCE_DOCS_DIR,
    CHROMA_DB_DIR,
    CHROMA_COLLECTION_NAME,
)


def get_chroma_client() -> chromadb.PersistentClient:
    """Returns a persistent ChromaDB client pointing to data/chroma_db."""
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DB_DIR))


def get_or_create_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    """
    Returns the regulatory documents collection using Chroma's default
    local sentence-transformer / onnx embedding function.
    """
    embed_fn = embedding_functions.DefaultEmbeddingFunction()
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

def chunk_markdown_document(file_path: Path) -> List[Dict[str, Any]]:
    """
    Reads a markdown file and splits it into logical section chunks.
    Preserves document title, section headers, and assigns deterministic chunk IDs.
    """
    text = file_path.read_text(encoding="utf-8")
    file_stem = file_path.stem

    if "guidance" in str(file_path):
        doc_type = "guidance"
        drug_id = "general_guidance"
        jurisdiction = "FDA" if "fda" in file_stem else "EMA"
    else:
        doc_type = "drug_label"
        parts = file_stem.split("_")
        drug_id = parts[0]
        jurisdiction = parts[1].upper() if len(parts) > 1 else "UNKNOWN"

    sections = re.split(r"(?=(?:\n|^)##\s+)", text)
    chunks = []
    chunk_index = 0

    doc_header = ""
    for section in sections:
        section = section.strip()
        if not section:
            continue


        if section.startswith("# ") and "## " not in section:
            doc_header = section
            continue


        lines = section.splitlines()
        first_line = lines[0] if lines else ""
        section_title = first_line.lstrip("#").strip() if first_line.startswith("#") else "General Information"


        content_to_embed = f"{doc_header}\n\n{section}".strip() if doc_header else section
        chunk_id = f"{file_stem}_chunk_{chunk_index:02d}"

        chunks.append({
            "id": chunk_id,
            "text": content_to_embed,
            "metadata": {
                "chunk_id": chunk_id,
                "source_file": file_path.name,
                "drug_id": drug_id,
                "jurisdiction": jurisdiction,
                "doc_type": doc_type,
                "section_title": section_title,
            }
        })
        chunk_index += 1

    return chunks

def ingest_all_documents() -> int:
    """
    Finds all .md files in data/docs/drugs and data/docs/guidance,
    chunks them, and loads them into ChromaDB.
    """
    client = get_chroma_client()
    
    # 1. Reset collection for a clean, deterministic rebuild
    try:
        client.delete_collection(name=CHROMA_COLLECTION_NAME)
    except Exception:
        pass

    collection = get_or_create_collection(client)

    # 2. Gather all 10 markdown files (8 drugs + 2 guidance)
    all_files = list(DRUGS_DOCS_DIR.glob("*.md")) + list(GUIDANCE_DOCS_DIR.glob("*.md"))
    all_chunks = []

    for file_path in all_files:
        chunks = chunk_markdown_document(file_path)
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No documents found to ingest.")
        return 0

    # 3. Format inputs for ChromaDB batch insertion
    ids = [c["id"] for c in all_chunks]
    documents = [c["text"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    print(f"Ingestion complete: {len(all_chunks)} chunks indexed into ChromaDB.")
    return len(all_chunks)


if __name__ == "__main__":
    ingest_all_documents()
