"""LLM-as-judge faithfulness scorer.

Given an answer and the retrieved context, ask a small LLM: is every claim
in the answer actually supported by the context?

This is the RAG-specific failure mode — the generation model can produce
beautiful, confident answers that silently ignore the retrieved chunks and
fall back to training data instead.

We use a simple single-prompt judge here. Ragas does this too but with more
plumbing; we'll swap in Ragas later for comparison.
"""

import json

from openai import OpenAI

JUDGE_MODEL = "gpt-4o-mini"  # cheap; sufficient for binary faithfulness calls

FAITHFULNESS_PROMPT = """You are evaluating whether an AI-generated answer is \
faithful to the source context it was given.

An answer is FAITHFUL if every substantive claim in it can be traced back \
to the provided context. It is UNFAITHFUL if it contains claims, numbers, \
dates, or specifics that are not supported by the context (even if they \
happen to be true in the real world).

A statement like "the context does not contain the answer" is FAITHFUL \
(it is an accurate statement about the context).

Context:
---
{context}
---

Answer:
---
{answer}
---

Evaluate faithfulness. Return a JSON object with exactly these keys:
- "score": float 0.0 to 1.0 (fraction of substantive claims supported by context)
- "unsupported_claims": list of strings (claims in the answer NOT supported by context, or empty list)
- "reasoning": short string explaining your score

Return ONLY the JSON, no other text."""


def score_faithfulness(client: OpenAI, answer: str, context: str) -> dict:
    """Returns {'score': float, 'unsupported_claims': [...], 'reasoning': str}."""
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "user", "content": FAITHFULNESS_PROMPT.format(
                context=context, answer=answer)},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    content = response.choices[0].message.content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"score": 0.0, "unsupported_claims": ["<judge returned invalid JSON>"],
                "reasoning": content[:200]}
