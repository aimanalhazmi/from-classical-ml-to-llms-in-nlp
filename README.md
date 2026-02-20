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
├── configs/                # JSON configurations for Task 1 and Task 2
├── data/                   # Raw datasets
│   ├── IRIS.csv            # Dataset for Task 2
│   └── SMM4H_2026/         # Dataset for Task 1
├── src/                    # Source code with modular logic
│   ├── task1/              # Scripts for Multilingual NLP
│   └── task2/              # Scripts for Hybrid/Tree models
├── report/                 # report (PDF)
├── output/                 # Model checkpoints, logs, and visualizations
├── main.py                 # Entry point for Task 2 execution
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

## Running Task 1: Multilingual NLP

### 1. Preprocessing

Clean the raw data and create splits (80% Train, 10% Val, 10% Test) in data/SMM4H_2026.

``` Bash

uv run src/task1/preprocessor.py
```

### 2. Exploratory Data Analysis (Optional)

Run EDA on the preprocessed splits. Results are saved in output/task1/eda.

``` Bash

uv run src/task1/explorer.py
```

### 3. Generate Configurations

Generate model-specific config files in the configs/ directory. These files define hyperparameters for the training process.

``` Bash

uv run src/task1/generate_configs.py
```

### 4. Training

Train a model using a specific config (e.g., Multilingual English-Russian).

``` Bash

uv run src/task1/trainer.py configs/task1-multi-en-ru.json
```

### 5. Evaluation

Evaluate a specific model checkpoint with a custom threshold.

``` Bash

uv run src/task1/evaluate_model.py \
  --config configs/task1-mono-ru.json \
  --checkpoint output/task1/mono_ru/xlm-roberta-base/2026-02-19_13-00/final \
  --threshold 0.15 \
  --per_language

```

---

## Running Task 2: Hybrid Modeling

ask 2 requires a local LLM server and specific configurations.

### 1. Modify Configuration

Before running, you can fine-tune Task 2 parameters (sample size, shots, etc.) in the config file:
Location: configs/task2-config.json

### 2. LLM Client Setup

Ensure the LLMClient in main.py matches your local server settings.

``` Bash

# Default: LM Studio on port 1234
llm = LLMClient(
    host="localhost", 
    port=1234, 
    model_name="mistralai_devstral-small-2-24b-instruct-2512", 
    system_role="developer"
)

```

### 3. Run Pipeline

Ensure the LLMClient in main.py matches your local server settings.

``` Bash

uv run main.py

```

---
## Working With JupyterLab
``` Bash
uv run jupyter lab
```

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