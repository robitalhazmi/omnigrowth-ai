# OmniGrowth-AI — Full-Stack Product Growth, Experimentation & Prescriptive Analytics Suite

[![CI](https://github.com/robitalhazmi/omnigrowth-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/robitalhazmi/omnigrowth-ai/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![DuckDB](https://img.shields.io/badge/OLAP-DuckDB-FFF000?logo=duckdb&logoColor=black)](https://duckdb.org/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-261230?logo=ruff&logoColor=D7FF64)](https://docs.astral.sh/ruff/)

**OmniGrowth-AI** is a production-grade Full-Stack Data Science, Experimentation, and Prescriptive Analytics platform for omnichannel retail and D2C product growth. It bridges offline statistical modeling with a low-memory online analytical serving engine designed to run comfortably within a $300\text{ MB}$ RSS memory envelope on a shared $1\text{ vCPU} / 2\text{ GB RAM}$ Linux VPS.

---

## Architecture Overview

```text
+-----------------------------------------------------------------------------------+
| STAGE 1: OFFLINE DATA & MODEL PIPELINE (Local Dev Workstation)                    |
|                                                                                   |
|  [Public Benchmark Datasets]                                                      |
|   |-- GA4 Ecommerce Sample (Events, Sessions, Funnel Steps)                       |
|   |-- Olist E-Commerce & Marketing Funnel (100k Orders + 8k B2B/D2C MQLs)         |
|   +-- E-Commerce Randomized A/B Experiment Logs (Pre/Post Covariates)             |
|          |                                                                        |
|          v                                                                        |
|  [DuckDB ETL & Feature Engineering (scripts/build_data_mart.py)]                  |
|   |-- Builds Zstandard-compressed Parquet marts (< 25 MB total)                   |
|   +-- Generates Looker Studio Semantic Mart (looker_executive_mart.csv)           |
|          |                                                                        |
|          v                                                                        |
|  [Batch Model Training & Cross-Validation (scripts/train_forecasts.py)]           |
|   |-- Fits Meta Prophet + Nixtla AutoARIMA across Regional Cohorts                |
|   +-- Fits Bass Diffusion (p, q, M) & Logistic S-Curves via Non-Linear Least Sq.  |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| STAGE 2: LOW-MEMORY ONLINE SERVING ENGINE (DuckDB 192MB Cap + Streamlit)          |
|                                                                                   |
|   |-- Module 1: DuckDB SQL Lab & GA4 Funnel + Looker Studio Semantic Layer        |
|   |-- Module 2: Frequentist (CUPED + Sequential) & Analytical Bayesian A/B Engine |
|   |-- Module 3: Interactive Time-Series & Bass S-Curve Scenario Planner           |
|   |-- Module 4: SciPy.optimize Prescriptive Budget & Inventory Solver             |
|   +-- Module 5: Groq/Gemini NL-to-SQL & Memo Agent + OpenPyXL .xlsx Generator     |
+-----------------------------------------------------------------------------------+
```

### Core Capabilities

| Module | Technical Implementation | Open-Source Stack |
| :--- | :--- | :--- |
| **1. Advanced SQL & GA4 Funnel Analytics** | Multi-stage CTEs, `ROW_NUMBER`, `LAG`/`LEAD`, `NTILE`, rolling windows, LookML views (`looker/*.lkml`), and Looker Studio mart export. | `DuckDB`, `Plotly`, `LookML` |
| **2. Dual-Paradigm Experimentation** | Sample size & MDE power analysis, **CUPED** variance reduction, O'Brien-Fleming sequential boundaries, and exact conjugate **Bayesian** posteriors ($ROPE$ & Expected Loss). | `Statsmodels`, `SciPy.stats`, `NumPy` |
| **3. Forecasting & Growth S-Curves** | Offline `Prophet` (with holiday/promo regressors) and `AutoARIMA` backtesting paired with **Bass Diffusion** ($p, q, M$) and **Logistic** S-curve adoption models. | `Prophet`, `StatsForecast`, `SciPy.optimize` |
| **4. Prescriptive Optimization** | Constrained non-linear programming (`SLSQP`) translating demand forecasts into optimal regional promotional budgets and safety-stock allocations. | `SciPy.optimize` |
| **5. GenAI Automation & Live Excel Export** | Read-only AST/regex-validated NL-to-DuckDB SQL assistant, automated C-Suite memo generator with offline cache fallback, and a 5-tab `.xlsx` financial model with live Excel formulas. | `Groq`, `Gemini`, `OpenPyXL` |

---

## Prerequisites

- **OS:** Linux / macOS (tested on Ubuntu 24.04 LTS)
- **Python:** `3.11+` (`3.12` recommended)
- **Make:** GNU Make (optional convenience runner)

---

## Local Setup & Quickstart

1. **Clone the repository and create the virtual environment:**
   ```bash
   git clone https://github.com/robitalhazmi/omnigrowth-ai.git
   cd omnigrowth-ai
   make setup
   source .venv/bin/activate
   ```

2. **Configure environment variables (optional for live GenAI inference):**
   ```bash
   cp .env.example .env
   ```
   > The application includes a deterministic offline fallback cache (`data/fallback/genai_cache.json`) and works with zero API keys configured.

3. **Download public benchmark datasets and build compressed Parquet marts:**
   ```bash
   make data
   ```
   To run in strictly offline mode using deterministic seeded replicas:
   ```bash
   .venv/bin/python scripts/download_datasets.py --offline
   .venv/bin/python scripts/build_data_mart.py
   ```

4. **Train offline forecasting & S-curve models:**
   ```bash
   make train
   ```

5. **Launch the Streamlit application:**
   ```bash
   make run
   ```

---

## Testing & Quality Gates

Run the automated test suite and static linter before submitting any changes:

```bash
# Run pytest suite
make test
# or directly:
.venv/bin/pytest tests/ -v

# Run Ruff linter
make lint
# or directly:
.venv/bin/ruff check src/ app/ tests/ scripts/
```

---

## Project Documentation

- [Implementation Plan](implementation_plan.md) — Architectural and mathematical blueprint
- [Task Checklist](task_checklist.md) — Phase-by-phase engineering execution tracker
- [Walkthrough Guide](walkthrough_guide.md) — End-to-end build, VPS deployment, and technical design guide
- [Agent Guardrails (`AGENTS.md`)](AGENTS.md) — Strict memory, modularity, and testing invariants

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our branch workflow, Conventional Commits standard, and pull request requirements, or open an issue via our [GitHub Issue Forms](https://github.com/robitalhazmi/omnigrowth-ai/issues/new/choose).
