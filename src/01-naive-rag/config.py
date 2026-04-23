"""Shared configuration for step 01 — naive RAG."""

from pathlib import Path

# Corpus
LETTERS_DIR = Path("/Users/alexgouyet/Gauntlet/rag-cookbook/letters")

# Chunking
CHUNK_SIZE_TOKENS = 500        # per chunk
CHUNK_OVERLAP_TOKENS = 50      # overlap between consecutive chunks

# Vector store (local Chroma)
CHROMA_PATH = Path(__file__).parent / ".chroma"
COLLECTION_NAME = "berkshire_naive"

# Models
EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4o-mini"

# Retrieval
TOP_K = 5
