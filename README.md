# CS224N Assignments

Assignments for Stanford's CS224N: Natural Language Processing with Deep Learning.

## Assignment List

| # | Topic | Source |
|---|-------|--------|
| A1 | Exploring Word Vectors | Winter 2026 |
| A2 | Word2Vec and Dependency Parsing | Winter 2026 |
| A3 | Self-Attention and Transformers | Winter 2026 |
| A4 | Neural Machine Translation with sequence-to-sequence, attention, and subwords | Winter 2024 |
| A5 | Self-Supervised Learning and Fine-tuning with Transformers | Winter 2024 |

A1–A3 come from the current Winter 2026 offering. A4 and A5 come from the
Winter 2024 offering ([cs224n.1244](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1244/)),
which does not include the LLM Evals assignment. They replace the original
Winter 2026 A4 (LLM Evals).

## Download

Download all assignments with a single script (A1–A3 from Winter 2026,
A4–A5 from Winter 2024). Assignments already present on disk are skipped,
so re-running is safe:

```bash
bash download.sh
```

If you see a permission error, use:

```bash
sh download.sh
```

## Setup

Create a virtual environment and sync dependencies:

```bash
uv venv
uv init --bare
uv sync
```
