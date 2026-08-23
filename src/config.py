import os
import time
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
# Model Allocation: Heavy Reasoning for Agent, Lightweight Speed for Verifier
# for proof of concept, we're using a single lightweight model - if quotas permit, 
# heavier models could be used for better results.
GEMINI_AGENT_MODEL = os.getenv("GEMINI_AGENT_MODEL", "gemini-3.5-flash-lite")
GEMINI_VERIFIER_MODEL = os.getenv("GEMINI_VERIFIER_MODEL", "gemini-3.5-flash-lite")

# Embedding Settings
CHROMA_COLLECTION_NAME = "regulatory_documents"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def call_gemini_with_retry(client, model, contents, config, max_retries=6):
    """
    Executes Gemini generate_content with automated backoff for 429/503 rate limits.
    """
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            err_str = str(e)
            if any(code in err_str for code in ["429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE"]):
                # Sleep 10s on first retry, 15s on subsequent retries to clear the 1-minute rate bucket
                wait_time = 10 if attempt == 0 else 15
                print(f" [Rate limit 429/503. Waiting {wait_time}s for window reset...] ", end="", flush=True)
                time.sleep(wait_time)
            else:
                raise e
    raise RuntimeError("Max retries exceeded for Gemini API call.")
