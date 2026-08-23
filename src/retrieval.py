import csv
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.utils import embedding_functions

from src.config import (
    CHROMA_DB_DIR,
    CHROMA_COLLECTION_NAME,
    PROCESSED_DATA_DIR,
)


def get_collection() -> chromadb.Collection:
    """Connects to the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    embed_fn = embedding_functions.DefaultEmbeddingFunction()
    return client.get_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=embed_fn,
    )

def search_fda(query: str, n_results: int = 3) -> List[Dict[str, Any]]:
    """
    Performs semantic search over FDA drug labels and guidance documents.
    Filters the vector space strictly to chunks where jurisdiction == 'FDA'.
    """
    collection = get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where={"jurisdiction": "FDA"},
    )
    chunks = []
    if results and results["ids"] and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            # chromaDB stores data column by column, and is batched by query.
            # since we sent 1 query, results[0] contains our matches.
            # [0][i] accesses the i-th retrieved chunk (from nearest to furthest).
            chunks.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if "distances" in results and results["distances"] else None,
            })
    return chunks

def search_ema(query: str, n_results: int = 3) -> List[Dict[str, Any]]:
    """
    Performs semantic search over EMA SmPCs and assessment reports.
    Filters the vector space strictly to chunks where jurisdiction == 'EMA'.
    """
    collection = get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where={"jurisdiction": "EMA"},
    )
    chunks = []
    if results and results["ids"] and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            chunks.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if "distances" in results and results["distances"] else None,
            })
    return chunks

def get_drug_metadata(drug_id: str) -> Optional[Dict[str, Any]]:
    """
    Looks up high-level baseline regulatory status from data/processed/drugs_metadata.csv.
    Matches against drug_id, generic_name, or brand_name (case-insensitive).
    """
    csv_path = PROCESSED_DATA_DIR / "drugs_metadata.csv"
    if not csv_path.exists():
        return None
    query = drug_id.strip().lower()
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (
                row["drug_id"].lower() == query
                or row["generic_name"].lower() == query
                or row["brand_name"].lower() == query
            ):
                return row
    return None

