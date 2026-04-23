# agentic-rag-evals

**A lab for building agentic RAG systems with production-grade evals.**

Most RAG demos ship retrieval and call it done. The interesting work — and where real systems fall over — is in the agent's tool-choice decisions and the evaluation harness that catches regressions before users do. This repo progresses from naive RAG to agentic RAG, scoring each step on the six metrics that matter in production.

## Why this exists

From Gauntlet AI's Night School (Agentic RAG, 2026-04-22) — the three things Ash Tilawat argued separate shipped AI systems from demo-ware:

1. **Evals are the moat.** Every retrieval pattern is scored on precision, recall, groundedness, latency, cost, and agent-specific metrics (tool selection, query decomposition, stop quality).
2. **Fuzzy vs deterministic split.** Not every retrieval needs an LLM. Unit tests catch the deterministic path; LLM-as-judge rubrics catch the fuzzy path.
3. **Training backwards.** Prompts, models, and tools change in response to eval scores. The system gets better month-over-month because the harness keeps it honest.

This repo is a working implementation of that discipline.

## Structure

```
agentic-rag-evals/
├── src/                 # retrieval + generation code
├── evals/
│   ├── scorers/         # accuracy, groundedness, recall, precision, latency, cost
│   └── datasets/        # golden sets (JSONL) per pattern
├── docs/                # architecture notes, eval deltas between patterns
├── requirements.txt
└── README.md
```

## Patterns implemented (roadmap)

Progressing from simple to sophisticated, each step measured against the previous:

| Step | Pattern | Status | Delta vs. prior |
|---|---|---|---|
| 01 | Naive RAG (chunk → embed → vector search → generate) | 🟡 in progress | baseline |
| 02 | Metadata-filtered RAG | ⬜ | precision ↑ on scoped queries |
| 03 | Hybrid search (BM25 + vector + RRF) | ⬜ | recall ↑ on proper nouns |
| 04 | Graph RAG | ⬜ | relational query coverage |
| 05 | Agentic RAG (LLM picks retrieval strategy per query) | ⬜ | end-to-end task success ↑ |

Each step includes: working code, a ≥10-query golden set, and scored eval runs committed to `docs/`.

## Stack

- **LLM / embeddings:** OpenAI (text-embedding-3-small, gpt-4o-mini) — swapable to Anthropic
- **Vector store:** MongoDB Atlas (step 01-05) — with pgvector as alternate path
- **Graph store:** Neo4j (step 04 only)
- **Eval / observability:** LangSmith (has first-class support for RAG datasets + LLM-as-judge scorers)
- **Framework:** LangChain where it pulls weight, raw SDK where it doesn't

## Metrics

Every eval run reports these. Numbers live in `docs/eval-runs/` as markdown tables, commit-by-commit.

Mapped to [Ragas](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/) terminology where different, so reviewers can cross-reference:

1. **Context Precision** — of retrieved chunks, how many relevant *(Ragas: context_precision)*
2. **Context Recall (+ coverage)** — of relevant chunks in corpus, how many found *(Ragas: context_recall)*
3. **Faithfulness / Groundedness** — output grounded in retrieved context, not training-data hallucination *(Ragas: faithfulness)*
4. **Answer Relevancy** — does the response actually address the query *(Ragas: answer_relevancy)*
5. **Latency** — p50 / p95 per query-type bucket
6. **Cost** — tokens per query, broken out by embedding / retrieval / generation
7. **Agent-specific** (step 05 only) — tool-selection accuracy, query-decomposition quality, stop-quality (escalation / retry behavior)

Following [LangSmith's recommended mix](https://docs.langchain.com/langsmith/evaluate-rag-tutorial): heuristic checks for deterministic outputs, LLM-as-judge for subjective quality, and versioned golden-set datasets to prevent drift across runs.

## How to run

```bash
git clone https://github.com/AlexGouyet/agentic-rag-evals.git
cd agentic-rag-evals
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in OPENAI_API_KEY, MONGO_DB_URL, LANGSMITH_API_KEY

# step 01
cd src/01-naive-rag
python ingestion.py
python generation.py

# run evals
cd ../../evals
python run_evals.py --step 01
```

## About

Built by [Alexander Gouyet](https://alexandergouyet.com) ([@AlexGouyet](https://github.com/AlexGouyet)) as part of applying to Gauntlet AI. Iterating publicly — each commit advances one pattern or one eval.

Feedback, issues, and PRs welcome.
