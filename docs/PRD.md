# PRD — agentic-rag-evals

**Author:** Alexander Gouyet
**Created:** 2026-04-22
**Status:** in progress
**Review target:** Monday 2026-04-28 (Drew Rice admissions office hours) — also Friday 2026-04-25 office hours

## Why this exists

Alexander was rejected from Gauntlet AI (first application) with **specific feedback**: the LinkedIn profile didn't show enough technical / coding experience, and the reviewer wanted to see a GitHub profile with public projects demonstrating technical chops.

This repo is the response. Not a throwaway — a sustained public artifact that shows judgment about RAG architecture, eval discipline, and production thinking. The pitch on the reviewer's screen should read "this person has done the work," not "this person watched a tutorial."

## Audience

Primary: Gauntlet admissions reviewers (Drew Rice, Adam, Ash Tilawat, and whoever does the re-review).
Secondary: anyone evaluating Alexander for AI engineering work in 2026+.

Assume they spend 30 seconds on the README before deciding whether to scroll. Optimize for that.

## v1 Scope — What ships by Monday 2026-04-28

The minimum artifact that passes the 30-second test:

- [x] **Public GitHub repo** at [github.com/AlexGouyet/agentic-rag-evals](https://github.com/AlexGouyet/agentic-rag-evals) with clean README
- [x] **Eval framework doc** at `docs/eval-framework.md` with all 7 metrics defined, Ragas terminology cross-referenced
- [x] **PRD** (this doc) showing product thinking, not just code
- [x] **Glossary** at `docs/glossary.md` — living dictionary of terms (corpus, tokens, chunks, embeddings, vectors, ingestion, retrieval, generation, RAG variants, BM25, RRF, reranker, MCP, graph, tuples, etc.). Doubles as a proof-of-learning artifact reviewers can skim.
- [ ] **Step 01 (Naive RAG) working end-to-end** over the Berkshire letters corpus (cookbook-compatible)
  - [ ] `src/01-naive-rag/ingestion.py` — chunk + embed + upsert to vector store
  - [ ] `src/01-naive-rag/generation.py` — query + retrieve + generate
  - [ ] Local README under `src/01-naive-rag/` explaining design decisions
- [ ] **First eval run committed** with real numbers — not placeholders
  - [ ] Golden set of 10 queries under `evals/datasets/01-naive-rag.jsonl`
  - [ ] Scorer implementations for at minimum: context precision, context recall, faithfulness
  - [ ] `docs/eval-runs/01-naive-rag-2026-04-xx.md` with the scored numbers + brief commentary
- [ ] **LangSmith wired up** — traces from ingestion + generation visible in dashboard
- [ ] **A commit message trail** that tells a story: baseline → fix → improvement

## v1.1 Scope — What ships by end of week (2026-04-27 to 05-04)

- [ ] **Step 02 (Metadata-filtered RAG)** with eval delta vs. step 01
- [ ] **Step 03 (Hybrid search)** — BM25 + vector + RRF, with eval delta
- [ ] **First blog post / write-up** on alexandergouyet.com linking back to the repo, framed around the eval deltas

## v2 Scope — Deferred

- Step 04 (Graph RAG) — Neo4j + entity extraction
- Step 05 (Agentic RAG) — LLM picks retrieval strategy
- Ragas + LangSmith side-by-side comparison write-up
- Full-stack web app interface
- Additional corpora (Night School transcripts, Alexander's portfolio projects)
- **MCP-bonus chapter** — same agentic RAG pattern reimplemented via Anthropic's Model Context Protocol. Shows both paradigms (LangChain-style tools vs. MCP-native). Positions Alexander as fluent in Anthropic's ecosystem, not just open-source stacks.

## Parallel portfolio piece (separate repo, future work)

The Swift Fit Cowork plugin (`~/SwiftFitEvents/SwiftFitEvents-Claude/`) is **already doing agentic retrieval over MCP** — every skill that invokes HubSpot / Gmail / Drive / Slack / Notion / Calendar via MCP tools is agentic retrieval-augmented generation in the broad sense. What it lacks:

1. **A vector layer** — a `swift-knowledge-search` skill over past proposals / debriefs / transcripts to give it true semantic retrieval alongside the structured MCP queries
2. **An eval harness** — no LangSmith-equivalent exists for MCP-based Claude Code agents today; this is an open-problem opportunity
3. **A public write-up** — documenting the architecture + delta vs. classical RAG stacks

Potential second public portfolio piece: **`swift-skills-evals`** or similar — an MCP-native eval framework for Claude Code agents. Would be scope-appropriate *during* the Gauntlet program (or immediately before admission if time permits). Strongest possible pitch: "I built the eval framework that Ragas/LangSmith don't cover."

## Stack decisions (locked)

| Choice | Reason |
|---|---|
| OpenAI embeddings + gpt-4o-mini | Matches cookbook exactly; cheap; fast to get numbers |
| MongoDB Atlas M0 (vector store) | Cookbook-compatible; free tier |
| LangSmith for tracing + evals | Recommended by Ash at Night School; free tier; has MCP |
| LangChain where it pulls weight | Default framework; drop to raw SDK where LangChain slows us down |
| Berkshire Hathaway letters as v1 corpus | Cookbook default — gives direct comparability. Can swap later. |
| Python 3.10+ | Cookbook convention |
| MIT License | Maximum openness; signals "use this" to reviewers |

## Success criteria for the admissions re-review

The reviewer, spending 60 seconds on this repo, should think:

1. **Tech chops confirmed** — this person writes code, ships to GitHub, knows RAG beyond demos
2. **Judgment confirmed** — eval framework + Ragas terminology + metric selection signals production thinking
3. **Not a tutorial clone** — the PRD, the eval framework doc, and the commit history all show original thought layered on the cookbook pattern

The reviewer should NOT think:

- "This is just a fork of the Gauntlet cookbook with no added value"
- "No numbers — just boilerplate"
- "README is aspirational only"

## Risks / what could go wrong

| Risk | Mitigation |
|---|---|
| Ship scaffold-only, no real eval numbers by Monday | Commit step 01 + at least 10 real scored queries before Monday |
| Terminology invented from thin air (credibility hit) | Ragas mapping done; cite docs explicitly |
| README reads arrogant / too polished for the content underneath | Keep "in progress" markers honest; commit often; don't oversell |
| Alexander runs out of steam after tonight | Keep PRs small; set one-metric-at-a-time cadence |
| Corpus choice undermines originality (Berkshire is cookbook default) | v1.1 ships a second corpus (Night School or personal) with eval delta write-up |

## Session log

### 2026-04-22 (initial build)
- Set up repo structure, README, `.env.example`, requirements.txt, LICENSE, PRD, eval-framework.md
- Wrote glossary covering ~40 terms (foundational → stack-specific)
- Aligned metric terminology to Ragas conventions (context_precision, context_recall, faithfulness, answer_relevancy)
- Deep learning pass with Claude as tutor: embeddings vs generation, corpus, tokens, tuples, graph DB basics, MCP vs RAG, types of RAG (naive → agentic → Self-RAG / CRAG variants)
- Identified the Swift Fit skills plugin as a **parallel portfolio piece** — already doing agentic retrieval via MCP
- Decision: cookbook stack (OpenAI + Chroma/Mongo + LangChain) is the v1 public artifact; MCP version is v2
- **Not yet done:** step 01 code. Will pick up in next session with OpenAI API key in hand.

## Reference

- Cookbook: [Gauntlet-AIDP/rag-cookbook](https://github.com/Gauntlet-AIDP/rag-cookbook)
- Night School transcript: `/Users/alexgouyet/Documents/Zoom/2026-04-22 17.55.54 Night School_ Agentic RAG_ .../meeting_saved_closed_caption.txt`
- Ragas metrics: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/
- LangSmith RAG tutorial: https://docs.langchain.com/langsmith/evaluate-rag-tutorial
- Gauntlet Application Notion project: https://www.notion.so/34aea347ed438081a0dae8d11214325c
- Alexander's portfolio: https://alexandergouyet.com
- Alexander's LinkedIn: https://www.linkedin.com/in/alexander-gouyet
