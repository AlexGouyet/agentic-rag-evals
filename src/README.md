# src

Implementations of each RAG pattern. One directory per step.

- `01-naive-rag/` — chunk, embed, vector search, generate. Baseline.
- `02-metadata-filtered/` — pre-filter corpus slice before vector search.
- `03-hybrid-search/` — BM25 + vector, fused via Reciprocal Rank Fusion.
- `04-graph-rag/` — Neo4j entity/relationship traversal + vector search.
- `05-agentic-rag/` — LLM chooses retrieval strategy per query.

Each step is self-contained with `ingestion.py`, `generation.py`, and a local README. Evals for each live under `../evals/`.
