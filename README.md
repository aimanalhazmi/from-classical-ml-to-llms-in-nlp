# From Classical ML to LLMs in NLP

This project examines the transition from traditional machine learning approaches to modern Large Language Models (LLMs) in NLP. It addresses two core challenges: cross-lingual generalization in specialized domains and model reasoning on structured data.

---
## Task1:Multilingual Transfer in Medical NLP
Using medical social media data from SMM4H 2025, this task investigates how NLP models generalize across languages.
- Monolingual vs. Multilingual Models: Comparison of language-specific models with unified multilingual architectures. 
- Zero-Shot Evaluation: Evaluation on unseen languages to assess cross-lingual transfer and robustness. 
- Translation Artifacts: Analysis of performance degradation and error patterns introduced by machine-translated data

## Task2: Hybrid Modeling on Structured Data

This task compares classical tree-based models with Transformer-based and LLM approaches on the Iris dataset.
- Data Textification: Transformation of tabular features into natural language and semi-structured text formats. 
- Few-Shot Reasoning: Evaluation of LLM performance under limited supervision using training examples. 
- Hybrid Systems: Augmenting LLMs with structured outputs (e.g., decision rules or feature importance) from classical models to improve interpretability and performance.

---
## Project structure
```text
.
├── data/                   # Raw datasets
│   ├── IRIS.csv            # Dataset for Task 2
│   └── SMM4H_2026/         # Dataset for Task 1
├── src/                    # Source code with modular logic
│   ├── task1/              # Scripts for Multilingual NLP
│   └── task2/              # Scripts for Hybrid/Tree models
├── notebooks/              # Exploratory Data Analysis (EDA) and testing
├── report/                 # report (PDF/LaTeX)
├── output/                 # Model checkpoints, logs, and visualizations
├── main.py                 # Entry point to run tasks
├── pyproject.toml          # Project metadata and dependencies (uv)
└── uv.lock                 # Deterministic lockfile for reproducibility
``` 

---

---
## Prerequisites
- Python 3.12+
- [Astral UV](https://docs.astral.sh/uv/) installed on your system.
- Ollama Setup (mistral model by default )  for Benchmarking Task 

---
## Environment setup

### Create & sync the environment from the project root
``` Bash
uv sync
```
This will:
- Create a virtual environment (managed by uv)
- Install all dependencies specified in pyproject.toml 
- Use uv.lock to ensure reproducible versions

### Environment variables & secrets
Create a .env file (not committed to git)

Example .env:

```text
OPENAI_API_KEY=sk-...
```
Make sure .env is in .gitignore.

---

## Adding / removing dependencies with uv
Always manage dependencies via uv so that both pyproject.toml and uv.lock stay consistent.

1. Add a new dependency
``` Bash
# Add a runtime dependency
uv add package-name

# Add a dev-only dependency (e.g., testing, linting)
uv add --dev pytest
```
This will:
- Update [project.dependencies] (or [project.optional-dependencies] / dev section)
- Update uv.lock with the resolved versions

2. Remove a dependency
``` Bash
uv remove package-name
```

This will:
- Remove it from pyproject.toml
- Update uv.lock accordingly

After adding or removing dependencies, you can re-sync to ensure the environment matches:
``` Bash
uv sync
```
---

## Running the Application

``` Bash

uv run main.py
```
---
## Working With JupyterLab
``` Bash
uv run jupyter lab
```
---