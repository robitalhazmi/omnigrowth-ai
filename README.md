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


## Mathematical & Statistical Methodology

| Module | Technique | Mathematical / Algorithmic Formulation | Implementation |
| :--- | :--- | :--- | :--- |
| **1. SQL & GA4 Funnel** | Multi-Stage CTEs & Window Analytics | Chronological `session_start` -> `purchase` tracking (`ROW_NUMBER`, `LAG`, `LEAD`, `NTILE`), cohort retention, DuckDB zero-copy Parquet views (`192MB`, `1` thread) | [`sql_queries.py`](src/analytics/sql_queries.py), [`ga4_funnel.py`](src/analytics/ga4_funnel.py) |
| **2. Experimentation** | CUPED, Sequential & Bayesian | $\hat{Y}_i^{\text{CUPED}} = Y_i - \theta(X_i - \bar{X})$, Lan-DeMets O'Brien-Fleming boundaries $\alpha(t_k) = 2 - 2\Phi(Z_{1-\alpha/2}/\sqrt{t_k})$, conjugate Beta-Binomial & Normal-Inverse-Gamma Expected Loss & ROPE | [`frequentist_ab.py`](src/experimentation/frequentist_ab.py), [`sequential_testing.py`](src/experimentation/sequential_testing.py), [`bayesian_ab.py`](src/experimentation/bayesian_ab.py) |
| **3. Forecasting & Diffusion** | Prophet, AutoARIMA & Bass S-Curve | Offline rolling-origin CV artifacts in `data/processed/`; online analytical shocks (`< 50 ms`) & Bass diffusion $N(t) = M\frac{1 - e^{-(p+q)t}}{1 + (q/p)e^{-(p+q)t}}$, $t^* = \frac{\ln(q/p)}{p+q}$ | [`time_series.py`](src/forecasting/time_series.py), [`s_curve.py`](src/forecasting/s_curve.py) |
| **4. Prescriptive Solver** | Constrained SLSQP Optimization | Joint marketing budget $b_r$ & regional inventory $I_r$ solver maximizing net margin net of holding ($c_{\text{hold}}$) & 95th-pctl stockout penalties ($c_{\text{short}}$) subject to 80% safety-stock floors | [`prescriptive_solver.py`](src/optimization/prescriptive_solver.py) |
| **5. GenAI & Excel Engine** | Failover NL-to-SQL & Live Formulas | Groq (`10s` timeout + exponential backoff) -> Gemini -> offline cache (`data/fallback/genai_cache.json`); read-only AST/regex SQL validator; 5-tab formula `.xlsx` workbook | [`assistant.py`](src/genai/assistant.py), [`sql_validator.py`](src/genai/sql_validator.py), [`excel_generator.py`](src/reporting/excel_generator.py) |

## Google Looker Studio & Cloud Backup Architecture

- **LookML Semantic Layer:** Defined in [`ga4_funnel.view.lkml`](looker/ga4_funnel.view.lkml) alongside the refreshed summary extract [`looker_executive_mart.csv`](data/processed/looker_executive_mart.csv).
- **Live BI Embed:** Configure `LOOKER_STUDIO_EMBED_URL` in `.env` to render your Looker Studio dashboard directly inside Tab 1 (`SQL Funnel & Looker BI`).
- **Cloud Storage Offload:** Large raw CSV inputs stay outside the VPS runtime footprint; sync compressed Parquet marts (`data/processed/*.parquet`) and fallback snapshots to private object storage (e.g., Cloudflare R2 / S3 / Google Drive) for reproducible zero-downtime VPS provisioning.

## Production VPS Deployment (Co-Existing 1 vCPU / 2 GB RAM)

OmniGrowth-AI ships ready for co-located deployment on a shared Ubuntu 24.04 VPS alongside existing services while strictly isolating CPU and RSS memory under **300 MB**:

1. **Provision & Train Artifacts:**
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   python scripts/prepare_data.py
   python scripts/train_forecasts.py
   ```
2. **Install Cgroup-Capped `systemd` Service ([`omnigrowth.service`](deploy/omnigrowth.service)):**
   ```bash
   sudo cp deploy/omnigrowth.service /etc/systemd/system/omnigrowth.service
   sudo systemctl daemon-reload && sudo systemctl enable --now omnigrowth
   sudo systemctl status omnigrowth
   ```
   Enforces `MemoryHigh=280M`, `MemoryMax=350M`, `CPUQuota=50%`, `OMP_NUM_THREADS=1`, and binds headless Streamlit to `127.0.0.1:8501`.
3. **Configure Nginx WebSocket Proxy ([`nginx_omnigrowth.conf`](deploy/nginx_omnigrowth.conf)) & TLS:**
   ```bash
   sudo cp deploy/nginx_omnigrowth.conf /etc/nginx/sites-available/omnigrowth
   sudo ln -s /etc/nginx/sites-available/omnigrowth /etc/nginx/sites-enabled/omnigrowth
   sudo nginx -t && sudo systemctl reload nginx
   sudo certbot --nginx -d omnigrowth.yourdomain.com
   ```
