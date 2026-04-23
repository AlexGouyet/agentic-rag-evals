# Eval framework

Every retrieval pattern in this repo is scored on the same six metrics. Scores are committed as markdown tables in `docs/eval-runs/` so progress is visible in git history.

## The six metrics

### 1. Precision
**What:** of the chunks the system retrieved, how many were actually relevant to the query.
**Why:** high precision means less noise in the context window and fewer tokens wasted on irrelevant material.
**How scored:** for each golden-set query, a human-labeled or LLM-judged set of "relevant" chunks. Precision = relevant retrieved / total retrieved.

### 2. Recall (+ coverage)
**What:** of the chunks in the corpus that should have been retrieved, how many were.
**Why:** missing relevant material is the #1 cause of hallucinated or evasive answers. Low recall can't be recovered downstream.
**How scored:** recall@k against the golden set. Coverage tracks whether the corpus itself has the answer at all — distinguishes "we didn't find it" from "it isn't there."

### 3. Groundedness
**What:** is the generated output supported by the retrieved context, or is the LLM falling back to training data?
**Why:** this is the RAG-specific failure mode. Beautiful-looking answers that ignore the retrieved chunks are worse than useful — they give false confidence.
**How scored:** LLM-as-judge pass over each (generated claim, retrieved chunks) pair. Binary per claim; aggregated per response.

### 4. Latency
**What:** wall-clock time to answer, bucketed by query type.
**Why:** an agentic system can legitimately take 30s–5min for deep queries. The eval isn't "make it fast" — it's "set user expectations per query type and measure drift."
**How scored:** p50 and p95 per bucket, logged via LangSmith traces.

### 5. Cost
**What:** tokens per query, broken out by embedding, retrieval, and generation.
**Why:** agentic systems with retries can silently 10x costs. Tracked at the metric level so regressions show up in PRs.
**How scored:** token counts from LangSmith spans, converted to USD per call.

### 6. Agent-specific (step 05 only)

#### Tool selection
**What:** for a given query, did the agent pick the right retrieval tool (vector / BM25 / graph / SQL / web)?
**How scored:** golden-set queries labeled with the *correct* tool. Binary match.

#### Query decomposition
**What:** did the agent break a multi-hop query into the right sub-queries?
**How scored:** LLM-as-judge against a reference decomposition.

#### Stop quality
**What:** did the agent stop when it had enough context, or keep retrieving / retrying wastefully?
**How scored:** ratio of productive tool calls to total. Low ratio = loops or over-retrieval.

## Golden sets

Each step in this repo ships with a ≥10-query golden set under `evals/datasets/<step>.jsonl`. Schema:

```json
{
  "query_id": "string",
  "query": "user-facing question",
  "expected_chunks": ["chunk_id_1", "chunk_id_2"],
  "expected_answer_fragments": ["fragment that must appear"],
  "must_not_mention": ["things the answer should not say"],
  "expected_tool": "vector | bm25 | graph | sql | web",
  "query_type": "factual | synthesis | multi-hop | out-of-scope",
  "notes": "human context"
}
```

Golden sets grow as failures are encountered in the wild. Every bug becomes a test.

## Fuzzy vs deterministic

Not every retrieval needs the LLM. Three hardcoded API calls in sequence is *deterministic* and gets a unit test, not a rubric. A vector-search + LLM-judgment-call is *fuzzy* and gets LLM-as-judge scoring.

This repo tests both kinds but keeps them separate. Deterministic scorers live under `evals/scorers/deterministic/`, fuzzy ones under `evals/scorers/fuzzy/`.

## References

- [Gauntlet Night School: Agentic RAG](https://www.gauntletai.com/) — 2026-04-22 session with Ash Tilawat
- [Ragas](https://github.com/explodinggradients/ragas) — reference eval framework
- [LangSmith docs](https://docs.smith.langchain.com/) — observability + eval runner
