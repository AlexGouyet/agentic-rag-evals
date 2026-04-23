# Step 01 — Naive RAG

Baseline pattern. Everything else in this repo is a delta against this.

## Architecture

```
PDFs (Berkshire letters)
  → read_pdf (pypdf)
  → chunk_by_tokens (tiktoken, 500 tokens / 50 overlap)
  → embed (OpenAI text-embedding-3-small)
  → persist (Chroma local collection)

query
  → embed (same model)
  → Chroma.query (cosine similarity, top-5)
  → stuff into prompt
  → generate (gpt-4o-mini)
  → answer + retrieved chunks
```

## Design decisions

- **Corpus**: Warren Buffett's Berkshire Hathaway annual letters 2004-2023 (cookbook standard).
- **Chunk size**: 500 tokens with 50-token overlap. Small enough that each chunk is focused, large enough to keep paragraph-level context intact.
- **Embedding model**: `text-embedding-3-small` (1536 dims). Cheap (~$0.02/1M tokens), well-calibrated, matches cookbook for direct comparability.
- **Vector store**: local Chroma with cosine similarity. Zero signup. Swap to MongoDB Atlas or pgvector in step 02+ to show portability.
- **Generation model**: `gpt-4o-mini`. Small, cheap, fast. The eval harness matters more than the generator.
- **Retrieval**: top-5 nearest neighbors. No filtering, no reranking, no hybrid. That's the "naive" part.
- **System prompt**: explicit instruction to cite the letter year and refuse if the context doesn't contain the answer. Gives faithfulness a fighting chance.

## Run

```bash
# from repo root, with venv active
cd src/01-naive-rag
python ingestion.py       # ~1 min, ~$0.20 in embedding cost
python generation.py "What does Buffett say about intrinsic value?"
python generation.py      # interactive mode
```

## What this baseline does NOT do (by design)

These are the failures step 02+ will fix:

- No metadata filters → can't scope to "letters from 2018 only"
- No keyword search → weak on proper-noun queries ("GEICO", specific deal names)
- No reranking → top-5 are purely embedding-similar, not quality-ranked
- No query rewriting → user typos or vague queries get vague retrieval
- No evaluation → no numbers yet. Adding next.

## Next

- `src/02-metadata-filtered/` — pre-filter by year/topic before vector search
- `evals/datasets/01-naive-rag.jsonl` — golden set (≥10 queries) to score this baseline
