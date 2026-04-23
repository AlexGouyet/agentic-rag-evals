# Evals

Golden sets + scorers + run history.

```
evals/
├── datasets/            # JSONL golden sets (per step)
├── scorers/
│   ├── deterministic/   # unit-test-style checks on structured output
│   └── fuzzy/           # LLM-as-judge rubrics for generated text
└── run_evals.py         # CLI: python run_evals.py --step 01
```

See `../docs/eval-framework.md` for the metric definitions and schema.
