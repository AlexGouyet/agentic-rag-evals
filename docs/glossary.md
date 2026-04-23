# Glossary

Living dictionary. Open this in a second monitor while working.

Terms are ordered roughly from foundational → specialized.

---

### Corpus
The full body of text the RAG system can retrieve from. Plural: *corpora*. Your corpus might be "all Berkshire Hathaway letters 2004–2023," "every past Swift Fit proposal," or "Gauntlet Night School transcripts." Whatever the system is allowed to read.

### Tokens
The small units an LLM processes — roughly fragments of words or characters. `"intrinsic"` might be 2-3 tokens. Models charge per token, in and out. The cost and length of a prompt are measured in tokens, not characters.

### Chunk
A slice of your corpus, usually a few hundred tokens. Ingestion breaks the corpus into chunks so each piece is small enough to retrieve individually. A typical Berkshire letter might become 50-100 chunks of ~500 tokens each.

### Embedding
A vector (list of numbers, ~1,500 long) that captures the *meaning* of a chunk. "Intrinsic value" and "core worth of a business" land close in embedding space. Computed once at ingestion, stored in a vector DB.

### Vector
The output of an embedding model. Just a list of floats. `[0.023, -0.11, 0.78, ...]`.

### Vector database
A database optimized to store vectors and find *nearest neighbors* quickly. Examples: Chroma, pgvector (Postgres extension), MongoDB Atlas Vector Search, Pinecone, Weaviate.

### Ingestion
The one-time process of taking a corpus, chunking it, embedding each chunk, and storing the results in the vector DB. Slow (you pay per chunk) but only happens when the corpus changes.

### Retrieval
At query time: embed the user's query the same way you embedded chunks, find chunks whose vectors are closest to the query vector. Those chunks become the "context" for generation.

### Generation
After retrieval, the LLM reads `query + retrieved chunks` and writes a natural-language answer. Different model and different API call than embedding.

### RAG (Retrieval-Augmented Generation)
The whole pattern: retrieve relevant chunks from outside the LLM → stuff them into the LLM's context → generate an answer grounded in those chunks. Solves "the LLM doesn't know my data" without retraining.

### Agentic RAG
Same pattern but the LLM **decides** which retrieval tool to call (vector search, SQL query, web search, graph traversal), how many times, and when to stop. vs. a fixed pipeline that always does the same steps.

### Golden set
A hand-curated list of queries paired with the correct retrieved chunks and/or correct answers. The ground truth an eval scorer compares against. Typically 10–200 queries.

### Eval / Evaluator / Scorer
Automated measurement of how well retrieval and generation performed. Each eval has a metric (e.g., "context precision") and a scorer (the code that computes it).

### Faithfulness / Groundedness
Does the generated answer actually come from the retrieved chunks, or is the LLM falling back to training-data memory? Ragas calls this "faithfulness," Ash at Night School called it "groundedness" — same thing.

### Context Precision vs Context Recall
- **Context Precision** — of chunks retrieved, how many were relevant (signal-to-noise)
- **Context Recall** — of chunks that *should* have been retrieved, how many actually were (did you miss anything)

### Answer Relevancy
Does the final answer address the user's question? You can have high faithfulness (grounded in retrieved chunks) but low relevancy (correctly cited but off-topic).

### LLM-as-judge
Using an LLM (often GPT-4) to score another LLM's output against a rubric. Cheap and scales. Noisy but roughly agrees with humans ~80-95% of the time depending on task.

### BM25
Classical keyword-matching search algorithm from the 1990s. Lexical, not semantic. Beats vector search when the query contains exact proper nouns ("GEICO", "Tesla", specific product SKUs). In hybrid search, BM25 and vector search run in parallel; results get merged.

### RRF (Reciprocal Rank Fusion)
A simple formula for merging two ranked lists of search results (e.g., BM25 results + vector results). Weights documents by `1 / (k + rank)` in each list, sums across lists.

### Reranker
A second-stage model that re-orders the top-N retrieved chunks for better precision. Typically a smaller LLM or cross-encoder that reads `(query, chunk)` pairs and scores them. Slower per pair but much more accurate than pure vector similarity for the top few slots.

### Context window
How much text the LLM can see at once, measured in tokens. GPT-4o-mini: 128k. Claude Opus 4.7: 200k (or 1M in extended mode). RAG exists because corpora are bigger than any context window.

### Prompt caching
Anthropic/OpenAI feature where repeated system prompts or static context don't get re-billed on every call. Cuts costs ~90% on workloads with lots of context reuse.

### Latency (p50, p95)
How long a query takes. p50 = median (half of queries finish faster), p95 = 95th percentile (only 5% of queries take longer). p95 is the honest user-experience number.

### Observability / tracing
Logging every step of a RAG pipeline (retrieval calls, LLM calls, tool calls) so you can debug *why* a bad answer happened. LangSmith is the defacto tool; Langfuse is the open-source alternative.

---

## Stack-specific

### OpenAI `text-embedding-3-small`
The cookbook's embedding model. Output vector length: 1536. Cheap, fast, decent quality.

### OpenAI `gpt-4o-mini`
The cookbook's generation model. Small, fast, cheap (~$0.15 / 1M input tokens). Good enough for demos.

### Anthropic Claude
The model family powering Claude Code. For this repo: `claude-sonnet-4-6` is the budget workhorse; `claude-opus-4-7` for hard reasoning tasks. You'd use these if pivoting generation away from OpenAI.

### EmbeddingGemma
Google's open-weight embedding model (308M parameters). Runs locally on CPU. Multilingual. Zero per-query cost. Not quite as sharp as OpenAI's embedding models but good enough for most workloads. Probably what OpenClaw uses for personal-data embedding.

### Chroma
Simple local vector database. One-file storage, Python-native, zero signup. Great for dev and personal projects. Swap to pgvector or MongoDB Atlas when scaling.

### pgvector
Postgres extension that adds a `vector` column type. SQL filters + vector search in one query. Most popular production choice in 2026.

### MongoDB Atlas Vector Search
MongoDB's managed cloud offering with vector search built in. Free tier (M0). What the cookbook uses.

### LangChain
Python/TS framework that wraps common LLM operations: prompt templates, chains (sequences of calls), agents, vector store integrations. Useful when it saves you boilerplate, painful when you're fighting its abstractions.

### LangSmith
LangChain's observability + eval platform. Traces every step of your chain, runs eval datasets, LLM-as-judge evaluators built in. Free tier, paid tiers beyond that.

### Ragas
Open-source RAG eval library. Reference implementations of faithfulness, answer relevancy, context precision, context recall. Industry-standard terminology.

### Langfuse
Open-source alternative to LangSmith. Same job, self-hostable.

---

*Add terms as they come up. Format: entry title → one plain-language sentence → then technical details if needed.*
