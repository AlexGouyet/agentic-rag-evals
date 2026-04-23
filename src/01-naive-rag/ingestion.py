"""Step 01 — Naive RAG: ingestion.

Reads Berkshire Hathaway annual shareholder letters (PDFs), splits into
fixed-size chunks by token count, embeds each chunk via OpenAI's
text-embedding-3-small, and stores in a local Chroma collection.

Run once (or when the corpus changes). Generation.py queries the collection
after this has run.
"""

import os
import sys
from pathlib import Path

import chromadb
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

from config import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LETTERS_DIR,
)

# Load env from repo root
load_dotenv(Path(__file__).parent.parent.parent / ".env")


def read_pdf(path: Path) -> str:
    """Return the full text of a PDF as a single string."""
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_by_tokens(text: str, size: int, overlap: int, encoder) -> list[str]:
    """Split text into overlapping chunks by token count."""
    tokens = encoder.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + size
        chunk = encoder.decode(tokens[start:end])
        chunks.append(chunk)
        if end >= len(tokens):
            break
        start = end - overlap
    return chunks


def main():
    client = OpenAI()
    encoder = tiktoken.encoding_for_model("gpt-4o-mini")

    # Chroma persistent client — stores embeddings in a local directory
    chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))
    # Fresh slate each run for now (naive RAG = no incremental updates yet)
    if COLLECTION_NAME in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(COLLECTION_NAME)
    collection = chroma.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    pdfs = sorted(LETTERS_DIR.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {LETTERS_DIR}")

    print(f"Ingesting {len(pdfs)} letters from {LETTERS_DIR}")
    total_chunks = 0

    for pdf_path in pdfs:
        year = pdf_path.stem.replace("ltr", "")  # "2004ltr" -> "2004"
        text = read_pdf(pdf_path)
        chunks = chunk_by_tokens(text, CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS, encoder)
        print(f"  {year}: {len(chunks)} chunks ({len(text):,} chars)")

        # Batch-embed via OpenAI
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=chunks,
        )
        embeddings = [item.embedding for item in response.data]

        # Upsert to Chroma
        ids = [f"{year}-{i:04d}" for i in range(len(chunks))]
        metadatas = [{"year": year, "source": pdf_path.name, "chunk_index": i}
                     for i in range(len(chunks))]
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        total_chunks += len(chunks)

    print(f"\nDone. {total_chunks} chunks across {len(pdfs)} letters.")
    print(f"Chroma collection '{COLLECTION_NAME}' persisted at {CHROMA_PATH}")


if __name__ == "__main__":
    main()
