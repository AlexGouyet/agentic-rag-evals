"""Eval runner — import the RAG system, feed it golden queries, score outputs.

Usage:
    python evals/run_evals.py --step 01-naive-rag

Writes a dated report to docs/eval-runs/<step>-<YYYY-MM-DD>.md and prints
a summary table to stdout.
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).parent.parent
load_dotenv(REPO_ROOT / ".env")

sys.path.insert(0, str(REPO_ROOT / "evals"))
from scorers.basic import task_success, year_recall, leakage_check
from scorers.faithfulness import score_faithfulness


def load_golden_set(step: str) -> list[dict]:
    path = REPO_ROOT / "evals" / "datasets" / f"{step}.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_system(step: str):
    """Dynamically import the RAG system for the given step."""
    sys.path.insert(0, str(REPO_ROOT / "src" / step))
    import generation
    import config
    from openai import OpenAI
    import chromadb

    client = OpenAI()
    chroma = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
    collection = chroma.get_collection(config.COLLECTION_NAME)
    return generation, client, collection


def format_context(retrieved_chunks: list[dict]) -> str:
    """Concat retrieved chunk previews into a single context string for the judge."""
    parts = []
    for i, c in enumerate(retrieved_chunks, 1):
        year = c.get("year", "?")
        parts.append(f"[Chunk {i} — {year}]\n{c.get('preview', '')}")
    return "\n\n---\n\n".join(parts)


def run(step: str):
    dataset = load_golden_set(step)
    generation, client, collection = load_system(step)

    print(f"\nRunning eval: {step}  ({len(dataset)} queries)\n")

    results = []
    for q in dataset:
        print(f"  {q['query_id']:25s} → running...", end="", flush=True)
        output = generation.answer(q["query"], collection, client)

        # Deterministic scorers
        ts = task_success(output["answer"], q.get("expected_fragments", []))
        yr = year_recall(output["retrieved_chunks"], q.get("expected_years", []))
        leak = leakage_check(output["answer"], q.get("must_not_mention", []))

        # LLM-as-judge faithfulness
        context = format_context(output["retrieved_chunks"])
        faith = score_faithfulness(client, output["answer"], context)

        results.append({
            "query_id": q["query_id"],
            "query_type": q.get("query_type", ""),
            "query": q["query"],
            "answer_preview": output["answer"][:280].replace("\n", " ") + "...",
            "task_success": ts,
            "year_recall": yr,
            "leakage": leak,
            "faithfulness": faith["score"],
            "faithfulness_reasoning": faith.get("reasoning", ""),
            "retrieved_years": sorted({c["year"] for c in output["retrieved_chunks"]}),
            "expected_years": q.get("expected_years", []),
            "tokens_in": output.get("usage", {}).get("prompt_tokens", 0),
            "tokens_out": output.get("usage", {}).get("completion_tokens", 0),
        })
        print(f" task={ts:.2f} yrs={yr:.2f} leak={leak:.2f} faith={faith['score']:.2f}")

    # Summary
    summary = {
        "task_success_avg": mean(r["task_success"] for r in results),
        "year_recall_avg": mean(r["year_recall"] for r in results),
        "leakage_avg": mean(r["leakage"] for r in results),
        "faithfulness_avg": mean(r["faithfulness"] for r in results),
        "total_tokens_in": sum(r["tokens_in"] for r in results),
        "total_tokens_out": sum(r["tokens_out"] for r in results),
        "n_queries": len(results),
    }

    print(f"\n=== Summary ({summary['n_queries']} queries) ===")
    print(f"  Task success:  {summary['task_success_avg']:.2f}")
    print(f"  Year recall:   {summary['year_recall_avg']:.2f}")
    print(f"  Leakage clean: {summary['leakage_avg']:.2f}")
    print(f"  Faithfulness:  {summary['faithfulness_avg']:.2f}")
    print(f"  Tokens in:     {summary['total_tokens_in']:,}")
    print(f"  Tokens out:    {summary['total_tokens_out']:,}")

    # Write report
    report_dir = REPO_ROOT / "docs" / "eval-runs"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{step}-{date.today().isoformat()}.md"
    with open(report_path, "w") as f:
        f.write(render_report(step, summary, results))
    print(f"\nReport written: {report_path.relative_to(REPO_ROOT)}")


def render_report(step: str, summary: dict, results: list[dict]) -> str:
    lines = [
        f"# Eval run — {step} — {date.today().isoformat()}",
        "",
        f"**Queries:** {summary['n_queries']}",
        f"**Token cost:** {summary['total_tokens_in']:,} in + {summary['total_tokens_out']:,} out",
        "",
        "## Summary",
        "",
        "| Metric | Score |",
        "|---|---|",
        f"| Task success (avg) | **{summary['task_success_avg']:.2f}** |",
        f"| Year recall (avg) | **{summary['year_recall_avg']:.2f}** |",
        f"| Leakage clean (avg) | **{summary['leakage_avg']:.2f}** |",
        f"| Faithfulness (avg) | **{summary['faithfulness_avg']:.2f}** |",
        "",
        "## Per-query results",
        "",
        "| ID | Type | Task | Year Recall | Leak | Faith | Retrieved years |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['query_id']} | {r['query_type']} | "
            f"{r['task_success']:.2f} | {r['year_recall']:.2f} | "
            f"{r['leakage']:.2f} | {r['faithfulness']:.2f} | "
            f"{', '.join(r['retrieved_years'])} |"
        )

    lines.append("")
    lines.append("## Per-query detail")
    lines.append("")
    for r in results:
        lines.append(f"### `{r['query_id']}`")
        lines.append(f"**Query:** {r['query']}")
        lines.append("")
        lines.append(f"**Answer preview:** {r['answer_preview']}")
        lines.append("")
        lines.append(f"- Task success: {r['task_success']:.2f}")
        lines.append(f"- Year recall: {r['year_recall']:.2f} "
                     f"(expected {r['expected_years']}, got {r['retrieved_years']})")
        lines.append(f"- Leakage: {r['leakage']:.2f}")
        lines.append(f"- Faithfulness: {r['faithfulness']:.2f} — "
                     f"{r['faithfulness_reasoning']}")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", default="01-naive-rag",
                        help="Directory under src/ containing the RAG system")
    args = parser.parse_args()
    run(args.step)
