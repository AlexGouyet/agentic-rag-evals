"""Deterministic scorers — cheap, fast, no LLM required.

These catch the failures a human can pattern-match on: did the required words
show up? did retrieval hit the right years? did the output mention something
it shouldn't?
"""


def task_success(answer: str, required_fragments: list[str]) -> float:
    """1.0 if every required fragment appears in the answer, else 0.0.

    Empty required_fragments list → 1.0 (nothing to check, vacuously true).
    """
    if not required_fragments:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for frag in required_fragments if frag.lower() in answer_lower)
    return hits / len(required_fragments)


def year_recall(retrieved_chunks: list[dict], expected_years: list[str]) -> float:
    """Of the years we expected retrieval to cover, what fraction did it hit?

    Ignores chunks outside the expected set. Empty expected_years → 1.0 (vacuous).
    """
    if not expected_years:
        return 1.0
    retrieved_years = {c["year"] for c in retrieved_chunks if "year" in c}
    expected = set(expected_years)
    overlap = retrieved_years & expected
    return len(overlap) / len(expected)


def leakage_check(answer: str, forbidden_fragments: list[str]) -> float:
    """1.0 if NONE of the forbidden fragments appear. 0.0 on first leak."""
    if not forbidden_fragments:
        return 1.0
    answer_lower = answer.lower()
    for frag in forbidden_fragments:
        if frag.lower() in answer_lower:
            return 0.0
    return 1.0
