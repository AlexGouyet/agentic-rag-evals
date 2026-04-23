"""Step 01 — Naive RAG: generation.

At query time: embed the query, retrieve top-k nearest chunks from Chroma,
stuff them into a prompt, and generate an answer via gpt-4o-mini.

Usage:
    python generation.py "What does Buffett say about intrinsic value?"
    python generation.py                    # interactive mode
"""

import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    TOP_K,
)

load_dotenv(Path(__file__).parent.parent.parent / ".env")


SYSTEM_PROMPT = """You are a careful research assistant answering questions about
Warren Buffett's annual shareholder letters.

Only use information from the provided context. If the context does not contain
the answer, say so explicitly — do not invent. When possible, cite the year of
the letter the information comes from (e.g., "In the 2012 letter...").
"""


def retrieve(query: str, collection, client: OpenAI, k: int = TOP_K) -> list[dict]:
    """Embed query, return top-k chunks with metadata."""
    query_embedding = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query],
    ).data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "document": doc,
            "metadata": meta,
            "distance": dist,
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def format_context(chunks: list[dict]) -> str:
    """Build the context block sent to the generation model."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        year = chunk["metadata"].get("year", "?")
        parts.append(f"[Chunk {i} — {year} letter]\n{chunk['document']}")
    return "\n\n---\n\n".join(parts)


def answer(query: str, collection, client: OpenAI) -> dict:
    """Full RAG pass: retrieve + generate."""
    chunks = retrieve(query, collection, client)
    context = format_context(chunks)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n\n{context}\n\n---\n\nQuestion: {query}"},
    ]

    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=messages,
        temperature=0.2,
    )

    return {
        "query": query,
        "answer": response.choices[0].message.content,
        "retrieved_chunks": [
            {"year": c["metadata"]["year"], "distance": c["distance"], "preview": c["document"][:200]}
            for c in chunks
        ],
        "usage": response.usage.model_dump() if response.usage else {},
    }


def interactive_loop(collection, client):
    print(f"Naive RAG over Berkshire letters. Top-K = {TOP_K}. Ctrl+C to exit.\n")
    while True:
        try:
            query = input("❯ ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nbye.")
            return
        if not query:
            continue
        result = answer(query, collection, client)
        print(f"\n{result['answer']}\n")
        print("— retrieved —")
        for c in result["retrieved_chunks"]:
            print(f"  {c['year']} (dist={c['distance']:.3f}): {c['preview'][:120]}…")
        print()


def main():
    client = OpenAI()
    chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        collection = chroma.get_collection(COLLECTION_NAME)
    except Exception:
        sys.exit(
            f"Collection '{COLLECTION_NAME}' not found. "
            f"Run `python ingestion.py` first."
        )

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        result = answer(query, collection, client)
        print(result["answer"])
        print("\n— retrieved —")
        for c in result["retrieved_chunks"]:
            print(f"  {c['year']} (dist={c['distance']:.3f}): {c['preview'][:120]}…")
    else:
        interactive_loop(collection, client)


if __name__ == "__main__":
    main()
