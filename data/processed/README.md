# Governance Evaluation Dataset

`governance_evaluation_dataset.xlsx` contains 600 labeled queries: 200 each for
Uncertainty Handling, Citation Grounding, and Risk Control. Each category is
balanced across four batches of 50.

The workbook includes consolidated and category-specific query sheets, a
deduplicated source index, controlled label definitions, planned metric
coverage, and a quality report. Citation and risk records contain exact evidence
quotes verified against their raw source documents.

Rebuild it from the repository root with:

```bash
python scripts/generate_governance_evaluation_dataset.py
```

The generator uses only the Python standard library. It recursively reads the
canonical TXT corpus, excludes exact duplicates and obvious navigation pages,
and produces deterministic Office Open XML for a fixed raw corpus.

All records are marked `programmatically_validated`. Independent policy and
compliance subject-matter review remains required before publication or use as
a high-stakes benchmark. Multi-agent handoff and latency metrics require runtime
traces and cannot be established from a static workbook alone.
