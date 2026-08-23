import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Directory Paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DOCS_DIR = DATA_DIR / "docs"
DRUGS_DOCS_DIR = DOCS_DIR / "drugs"
GUIDANCE_DOCS_DIR = DOCS_DIR / "guidance"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
EVALS_DIR = BASE_DIR / "evals"

# LLM Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

# Embedding Settings
CHROMA_COLLECTION_NAME = "regulatory_documents"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
