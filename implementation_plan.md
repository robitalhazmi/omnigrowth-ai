# Implementation Plan: OmniGrowth-AI — Full-Stack Product Growth, Experimentation & Prescriptive Analytics Suite

**Domain Scope:** Omnichannel Retail, D2C Digital Product Growth & Supply Chain Analytics  
**Development Environment:** Antigravity IDE on Local Laptop (Ubuntu 24.04.2 LTS, 16.0 GiB RAM, AMD Ryzen 5 PRO 3500U $\times 8$ threads)  
**Production Hosting:** Co-existing Shared VPS (Ubuntu 24.04 x86_64, 1 vCPU / 2 GB RAM via `systemd` cgroup memory cap + Nginx) & Streamlit Community Cloud Backup  
**Source Control:** GitHub (`main` production branch + feature branches)

---

## 1. Executive Summary & Architectural Alignment

Modern consumer brands and omnichannel retail ecosystems operate across two interconnected digital pillars:
1. **D2C Digital Product Growth:** Acquiring users across digital channels, optimizing onboarding and checkout funnels, running rigorous conversion/retention experiments, and modeling regional adoption S-curves.
2. **B2B/B2C Omnichannel Retail Operations:** Forecasting regional inventory demand, optimizing promotional budgets, and automating executive decision workflows across distributed merchant networks.

**OmniGrowth-AI** is an end-to-end, production-grade analytics and machine learning application built on 100% open-source market leaders and real public benchmark datasets:

| Core Capability | Project Module & Technical Implementation | Open-Source Tech Stack |
| :--- | :--- | :--- |
| **Advanced SQL (CTEs & Window Functions)** | Interactive **SQL Lab & Query Inspector** executing multi-stage CTEs, `LAG`/`LEAD`, `ROW_NUMBER`, `NTILE`, and rolling windows over real event logs. | `DuckDB` (In-Process OLAP Engine) |
| **GA4 & Looker Product Analytics** | **GA4 User Journey & Drop-Off Engine** using real GA4 ecommerce events + automated **Looker Studio** semantic data mart export, LookML view template, and embedded executive dashboard. | `DuckDB`, `Plotly`, `Looker Studio` |
| **A/B Testing (Frequentist & Bayesian)** | **Dual-Paradigm Experimentation Lab** featuring Frequentist power analysis, **CUPED** variance reduction, O'Brien-Fleming sequential testing, and analytical **Bayesian** conjugate posteriors ($ROPE$ & Expected Loss). | `Statsmodels`, `SciPy.stats`, `NumPy` |
| **Time-Series Forecasting (Prophet & ARIMA)** | **Multi-Horizon Forecasting Engine** comparing Meta `Prophet` (with holiday/promo regressors) and `AutoARIMA` with rolling-origin cross-validation and calibrated $80\% / 95\%$ confidence intervals. | `Prophet`, `StatsForecast` (Nixtla) |
| **Growth Modeling (S-Curves)** | **Market Adoption S-Curve Modeler** fitting **Bass Diffusion** (innovation $p$ vs. imitation $q$) and **Logistic Growth** models across regional cohorts to identify inflection points and market saturation ceilings. | `SciPy.optimize` (`curve_fit`) |
| **Prescriptive Analytics (Supply Chain & Marketing)** | **Constrained Resource Allocation Solver** translating demand forecasts and S-curve elasticities into optimal regional inventory safety stock and promotional spend allocations. | `SciPy.optimize` (`minimize` SLSQP) |
| **Advanced Excel & Financial Modeling** | **One-Click Executive `.xlsx` Generator** producing a 5-tab financial & forecasting workbook populated with live Excel formulas (`INDEX/MATCH`, dynamic CIs, ROI waterfalls) and native charts. | `OpenPyXL`, `Pandas` |
| **GenAI Workflow Automation** | **Dual GenAI Automation Suite**: (1) Read-only AST-validated Natural-Language-to-DuckDB SQL Agent and (2) Automated C-Suite Executive Memo Generator with deterministic offline fallback. | `Groq API` (`Llama-3.3-70B`), `Gemini API` (`2.5-Flash`) |

---

## 2. Antigravity IDE Workspace Rules (`AGENTS.md`)

Save the block below as `AGENTS.md` in the root of your repository before prompting Antigravity IDE. This ensures the coding agent respects your 2 GB VPS memory limits, modular architecture, and automated testing gates.

```markdown
# AGENTS.md — Antigravity IDE Autonomous Engineering Rules

## Project Context
You are building **OmniGrowth-AI**, a production Full-Stack Data Science, Experimentation & Prescriptive Analytics platform for omnichannel retail and D2C product growth.
- **Local Dev Machine:** Ubuntu 24.04.2 LTS, 16 GB RAM, 8 CPU threads. Heavy model training (`Prophet`, `AutoARIMA`, data ETL) runs offline via CLI scripts in `scripts/` and outputs compressed artifacts to `data/processed/`.
- **Production Target:** Shared Ubuntu 24.04 VPS with 1 vCPU and 2 GB RAM running other active web applications. The live Streamlit web app (`app/`) MUST stay under 300 MB RSS memory at all times.

## Strict Architectural Guardrails
1. **Offline-Training / Online-Serving Split:**
   - NEVER fit heavy `Prophet` or `AutoARIMA` models synchronously inside the main Streamlit request thread on raw multi-million-row tables.
   - Pre-compute base forecasts and store them in `data/processed/*.parquet`. Live scenario sliders in the UI must apply fast analytical impulse-response adjustments or fit lightweight aggregated series ($N \le 730$ daily points) in $< 500\text{ ms}$.
2. **DuckDB Memory Discipline:**
   - Every DuckDB connection in `src/database/duckdb_manager.py` MUST initialize with:
     `SET memory_limit = '192MB'; SET threads = 1;`
   - Query `.parquet` files directly using zero-copy views rather than loading entire tables into Pandas DataFrames.
3. **File & Code Modularity:**
   - Keep every Python file under 300 lines of code.
   - Use strict Python 3.11+ type hints (`typing`, `pydantic` or `dataclasses`) and Google-style docstrings.
   - Separate business/statistical logic (`src/`) from UI rendering (`app/`). `src/` modules must NEVER import `streamlit`.
4. **GenAI Resilience & Security:**
   - Every LLM call (`Groq` or `Gemini`) must have a 10-second timeout, automatic retry with exponential backoff, and a deterministic offline JSON fallback (`data/fallback/genai_cache.json`) so the app works even with zero API keys or exhausted rate limits.
   - All LLM-generated SQL MUST pass `src/genai/sql_validator.py` (enforcing read-only `SELECT` / `WITH` queries and rejecting `DROP`, `ALTER`, `INSERT`, `UPDATE`, `DELETE`, `COPY`, `ATTACH`, `INSTALL`, `LOAD`).
5. **Verification Gate:**
   - After creating or modifying any module in `src/`, you MUST run `pytest tests/ -v` and verify $100\%$ pass rate before proceeding to the next task.
```

---

## 3. System Architecture & Dual-Environment Strategy

To deliver rapid responsiveness on a shared 1 vCPU / 2 GB RAM VPS while showcasing heavy statistical modeling, the system uses a **Two-Stage Hybrid Architecture**:

```
+-----------------------------------------------------------------------------------+
| STAGE 1: OFFLINE DATA & MODEL PIPELINE (Local Ubuntu 24.04 Laptop - 16 GB RAM)    |
|                                                                                   |
|  [Raw Public Datasets]                                                            |
|   |-- GA4 BigQuery Ecommerce Sample (Events, Sessions, Funnel Steps)              |
|   |-- Olist E-Commerce & Marketing Funnel (100k Orders + 8k B2B/D2C MQLs)         |
|   +-- E-Commerce Randomized A/B Experiment Logs (Pre/Post Covariates)             |
|          |                                                                        |
|          v                                                                        |
|  [ETL & Feature Engineering (scripts/build_data_mart.py)]                         |
|   |-- Flattens nested GA4 event params into columnar Parquet                      |
|   |-- Generates Looker Studio Semantic Mart (looker_executive_mart.csv)           |
|          |                                                                        |
|          v                                                                        |
|  [Batch Model Training & Cross-Validation (scripts/train_forecasts.py)]           |
|   |-- Fits Meta Prophet + Nixtla AutoARIMA across Regional Cohorts                |
|   |-- Fits Bass Diffusion (p, q, M) & Logistic S-Curves via Non-Linear Least Sq.  |
|   +-- Exports compressed artifacts -> data/processed/*.parquet (< 25 MB total)    |
+-----------------------------------------------------------------------------------+
                                         |
                        (Git LFS / rsync to Production VPS)
                                         v
+-----------------------------------------------------------------------------------+
| STAGE 2: LOW-MEMORY ONLINE SERVING ENGINE (Ubuntu 24.04 VPS - Cgroup Cap: 350 MB) |
|                                                                                   |
|  [Nginx Reverse Proxy (HTTPS Let's Encrypt)] -> [Streamlit App (Port 8501)]       |
|   |-- Module 1: DuckDB SQL Lab & GA4 Funnel + Looker Studio Embed                 |
|   |-- Module 2: Frequentist (CUPED + Sequential) & Analytical Bayesian A/B Engine |
|   |-- Module 3: Interactive Time-Series & Bass S-Curve Scenario Planner           |
|   |-- Module 4: SciPy.optimize Prescriptive Budget & Inventory Solver             |
|   +-- Module 5: Groq/Gemini NL-to-SQL & Memo Agent + OpenPyXL .xlsx Generator     |
+-----------------------------------------------------------------------------------+
```

---

## 4. Real Public Dataset Strategy & Lineage

All datasets are 100% real public benchmark datasets, unified under an executive case study skin (**"Omnichannel Retail & Next-Gen Product Growth Suite"**) with a transparent **"Data Lineage & Methodology"** badge in the UI sidebar:

1. **GA4 User Journey & Funnel Dataset (`data/processed/ga4_events.parquet`)**
   * **Source:** Google Merchandise Store GA4 BigQuery Public Export (`bigquery-public-data.ga4_obfuscated_sample_ecommerce`).
   * **Schema:** `event_date`, `event_timestamp`, `event_name` (`session_start`, `view_item`, `add_to_cart`, `begin_checkout`, `add_payment_info`, `purchase`), `user_pseudo_id`, `ga_session_id`, `device_category`, `traffic_source`, `traffic_medium`, `geo_region`, `item_category`, `event_value_in_usd`.
   * **Business Mapping:** Mirrors digital user acquisition and checkout drop-offs on D2C e-commerce platforms and consumer apps.

2. **Regional Time-Series, S-Curve & Supply Chain Dataset (`data/processed/regional_demand.parquet` & `mql_funnel.parquet`)**
   * **Source:** Olist Brazilian E-Commerce & Marketing Qualified Leads Dataset (100,000 real orders across 27 states + 8,000 B2B/merchant funnel leads).
   * **Schema:** `order_date`, `region_code`, `product_segment`, `daily_orders`, `gross_revenue`, `freight_cost`, `promo_discount_pct`, `cumulative_active_merchants`, `lead_origin`, `won_timestamp`.
   * **Business Mapping:** Mirrors regional market penetration, B2B merchant network onboarding, and multi-region supply chain inventory planning.

3. **Randomized A/B Testing Dataset (`data/processed/ab_experiment_logs.parquet`)**
   * **Source:** Public E-Commerce A/B Testing Dataset enriched with 30-day pre-experiment baseline activity (`pre_exp_sessions`, `pre_exp_spend`) for CUPED covariate adjustment.
   * **Schema:** `user_id`, `experiment_group` (`control`, `treatment`), `assigned_date`, `device_os`, `region_tier`, `pre_exp_spend`, `converted` ($0$ or $1$), `post_exp_revenue`, `retention_d7`, `retention_d30`.
   * **Business Mapping:** Evaluates a new guided onboarding & loyalty mission flow designed to lift conversion and 30-day retention.

---

## 5. Deep-Dive Technical Specifications by Module

### Module 1: Advanced SQL Lab, GA4 Funnel Analytics & Looker Integration
* **DuckDB SQL Showcase (`src/analytics/sql_queries.py`):**
  Every visual chart in Module 1 is backed by a named, parameterized DuckDB SQL query that analysts and stakeholders can inspect and edit live in the UI:
  * **Query 1 (Closed-Funnel Step Conversion & Drop-Off Attribution):** Uses multi-stage CTEs (`WITH step_1 AS (...), step_2 AS (...)`) and `ROW_NUMBER() OVER (PARTITION BY user_pseudo_id, ga_session_id ORDER BY event_timestamp)` to enforce strict chronological funnel progression (`session_start` $\rightarrow$ `view_item` $\rightarrow$ `add_to_cart` $\rightarrow$ `begin_checkout` $\rightarrow$ `purchase`).
  * **Query 2 (Session Inter-Arrival Time & Churn Velocity):** Uses `LAG(event_timestamp) OVER (PARTITION BY user_pseudo_id ORDER BY event_timestamp)` and `LEAD()` to compute days between repeat orders and identify at-risk cohorts.
  * **Query 3 (RFM Quartile Scoring & Rolling Revenue):** Uses `NTILE(4) OVER (ORDER BY monetary_value DESC)` and `AVG(daily_revenue) OVER (PARTITION BY geo_region ORDER BY event_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)` for 7-day rolling trends.
* **Looker Studio & LookML Bridge (`looker/`):**
  * Generates `data/processed/looker_executive_mart.csv` for zero-cost connection to **Google Looker Studio**.
  * Includes a production-grade LookML semantic model (`looker/omni_growth.model.lkml` and `looker/ga4_funnel.view.lkml`) demonstrating enterprise Looker dimension/measure definitions, plus an embedded Looker Studio iframe tab inside the Streamlit UI.

### Module 2: Frequentist, Sequential & Bayesian A/B Experimentation Engine
* **Sample Size & Power Calculator (`src/experimentation/power_analysis.py`):**
  Computes minimum required sample size per arm $n$ given baseline conversion $p_1$, Minimum Detectable Effect ($MDE = p_2 - p_1$), significance level $\alpha$, and statistical power $1 - \beta$:
  $$n = \frac{\left(Z_{1 - \alpha/2}\sqrt{2\bar{p}(1-\bar{p})} + Z_{1-\beta}\sqrt{p_1(1-p_1) + p_2(1-p_2)}\right)^2}{(p_2 - p_1)^2}$$
* **CUPED Variance Reduction (`src/experimentation/frequentist_ab.py`):**
  Uses pre-experiment covariate $X$ (`pre_exp_spend`) to reduce the variance of the post-experiment metric $Y$ (`post_exp_revenue`):
  $$\hat{Y}_i^{\text{CUPED}} = Y_i - \theta (X_i - \bar{X}), \quad \text{where } \theta = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}$$
  The variance of the adjusted metric is reduced by a factor of $1 - \rho_{X,Y}^2$, increasing statistical power and shrinking confidence intervals without requiring additional traffic.
* **Sequential Testing Boundary (`src/experimentation/sequential_testing.py`):**
  Implements O'Brien-Fleming $\alpha$-spending boundaries across $K$ interim looks ($t_k = k/K \in (0, 1]$) so product teams can monitor experiments daily without inflating Type I error:
  $$\alpha(t_k) = 2 - 2\Phi\left(\frac{Z_{1 - \alpha/2}}{\sqrt{t_k}}\right)$$
* **Analytical Bayesian A/B Testing (`src/experimentation/bayesian_ab.py`):**
  Uses exact conjugate posteriors via `scipy.stats.beta` (zero MCMC memory overhead):
  * Prior: $\text{Beta}(\alpha_0, \beta_0)$ with $\alpha_0 = 1, \beta_0 = 1$.
  * Posterior for Arm $A$: $p_A \sim \text{Beta}(\alpha_0 + s_A, \beta_0 + n_A - s_A)$ and Arm $B$: $p_B \sim \text{Beta}(\alpha_0 + s_B, \beta_0 + n_B - s_B)$.
  * Computes exact **Probability Treatment Beats Control** $P(p_B > p_A)$, **Expected Loss** (risk of choosing $B$ if $A$ is actually better):
    $$\text{Expected Loss}(B) = \int_0^1 \int_0^1 \max(p_A - p_B, 0) \, f(p_A) f(p_B) \, dp_A dp_B$$
  * Evaluates the **Region of Practical Equivalence ($ROPE$)**, e.g., $[-0.2\%, +0.2\%]$.

### Module 3: Time-Series Forecasting (Prophet & ARIMA) & S-Curve Growth Modeling
* **Predictive Time-Series Engine (`src/forecasting/time_series.py`):**
  * **Meta Prophet:** Decomposes regional daily demand $y(t)$ into trend $g(t)$, weekly/yearly seasonality $s(t)$, Indonesian/regional holiday effects $h(t)$, and exogenous promotional regressors $\beta X(t)$:
    $$y(t) = g(t) + s(t) + h(t) + \beta X(t) + \epsilon_t$$
  * **Nixtla `StatsForecast` (`AutoARIMA`):** Fits seasonal ARIMA $(p, d, q)(P, D, Q)_m$ selecting optimal orders via Corrected Akaike Information Criterion ($AIC_c$).
  * **Evaluation Metrics:** Rolling-origin backtesting reporting Mean Absolute Percentage Error ($MAPE$), Root Mean Squared Error ($RMSE$), Symmetric MAPE ($sMAPE$), and $80\% / 95\%$ empirical interval coverage.
* **S-Curve Market Adoption Modeler (`src/forecasting/s_curve.py`):**
  Models regional product adoption and merchant network saturation using two parametric S-curve models fitted via `scipy.optimize.curve_fit`:
  1. **Logistic S-Curve:**
     $$N(t) = \frac{M}{1 + \exp\left(-k(t - t_0)\right)}$$
     where $M$ is the carrying capacity (Total Addressable Market ceiling), $k$ is the steepness of adoption, and $t_0$ is the inflection point time.
  2. **Bass Diffusion Model:**
     $$\frac{dN(t)}{dt} = \left(p + \frac{q}{M} N(t)\right)\left(M - N(t)\right)$$
     Closed-form cumulative adoption trajectory:
     $$N(t) = M \cdot \frac{1 - \exp\left(-(p+q)t\right)}{1 + \frac{q}{p}\exp\left(-(p+q)t\right)}$$
     where $p$ is the **Coefficient of Innovation** (external advertising/marketing effect), $q$ is the **Coefficient of Imitation** (word-of-mouth/network effect), and the peak adoption rate occurs at:
     $$t^* = \frac{\ln(q / p)}{p + q}$$

### Module 4: Prescriptive Analytics & Supply Chain / Marketing Optimization
* **Constrained Optimization Engine (`src/optimization/prescriptive_solver.py`):**
  Connects predictive forecasts to executive resource allocation across $R$ regions using `scipy.optimize.minimize` (`method='SLSQP'`):
  * **Decision Variables:** Regional promotional budget allocation $b_r$ and inventory allocation $I_r$ for $r \in \{1, \dots, R\}$.
  * **Objective Function:** Maximize total expected regional net contribution margin minus holding and stockout penalty costs:
    $$\max_{b_r, I_r} \sum_{r=1}^{R} \left[ m_r \cdot \hat{D}_r(b_r) - b_r - c_{\text{hold}} \max\left(0, I_r - \hat{D}_r(b_r)\right) - c_{\text{short}} \max\left(0, \hat{D}_{r, 0.95}(b_r) - I_r\right) \right]$$
    where regional demand responds to marketing spend with diminishing returns: $\hat{D}_r(b_r) = \hat{D}_r^{\text{base}} \left(1 + \gamma_r \ln\left(1 + \frac{b_r}{b_r^0}\right)\right)$.
  * **Constraints:**
    $$\sum_{r=1}^{R} b_r \le B_{\text{total}}, \quad \sum_{r=1}^{R} I_r \le I_{\text{warehouse}}, \quad I_r \ge \hat{D}_{r, 0.80}(b_r) \quad \forall r$$

### Module 5: GenAI Workflow Automation & Automated Excel Engine
* **GenAI Workflow Engine (`src/genai/assistant.py` & `src/genai/sql_validator.py`):**
  * **Provider Hierarchy:** Primary inference via **Groq Cloud Free Tier** (`llama-3.3-70b-versatile` for $< 1\text{ s}$ latency), secondary failover via **Google AI Studio Free Tier** (`gemini-2.5-flash`), and tertiary failover via deterministic pre-computed JSON responses (`data/fallback/genai_cache.json`).
  * **Workflow A — Natural-Language-to-DuckDB SQL Assistant:** Translates natural-language questions from executives into DuckDB SQL, validates the query via regex/AST guardrails (`SELECT`/`WITH` only), executes it against DuckDB, and explains the tabular output.
  * **Workflow B — One-Click C-Suite Executive Memo Generator:** Ingests live metrics from Modules 1–4 (funnel drop-off rates, CUPED A/B lift & Bayesian risk, Bass S-curve inflection dates, and prescriptive budget reallocations) and generates a structured Board Memo (`Executive Summary`, `Key Analytical Findings`, `Prescriptive Action Plan`, `Risk & Confidence Intervals`).
* **Automated Excel Financial & Forecasting Workbook (`src/reporting/excel_generator.py`):**
  Uses `OpenPyXL` to dynamically build a formatted, multi-tab `.xlsx` file downloadable from the dashboard:
  * **Tab 1 (`Executive_Summary`):** KPI scorecard with live cell references across tabs and conditional formatting.
  * **Tab 2 (`GA4_Funnel_Economics`):** Step-by-step funnel conversion table with live Excel formulas (`=C6/C5`, `=C5-C6`, revenue-at-risk sensitivity calculation).
  * **Tab 3 (`AB_Test_Model`):** Live Excel normal distribution & ROI projection formulas (`=NORM.S.DIST(...)`, annualized incremental revenue).
  * **Tab 4 (`Forecast_And_SCurve`):** Regional forecast table with live Excel formulas for lower/upper confidence bounds and Bass S-curve trajectory formulas, plus an embedded `OpenPyXL` LineChart.
  * **Tab 5 (`Prescriptive_Allocation`):** Baseline vs. Optimal budget and inventory allocation table with variance formulas (`=(D5-C5)/C5`).

---

## 6. Repository Directory Structure

```text
omnigrowth-ai/
├── AGENTS.md                         # Antigravity IDE autonomous coding rules
├── implementation_plan.md            # Architectural & mathematical blueprint
├── task_checklist.md                 # Step-by-step execution checklist
├── walkthrough_guide.md              # End-to-end build, deploy & architecture walkthrough guide
├── README.md                         # Executive portfolio showcase & badges
├── pyproject.toml                    # Dependencies & pytest/ruff configuration
├── requirements.txt                  # Pinned lightweight production dependencies
├── .env.example                      # Template for GROQ_API_KEY & GEMINI_API_KEY
├── Makefile                          # Shortcuts: make data, make train, make test, make run
├── data/
│   ├── raw/                          # Downloaded raw public datasets (gitignored)
│   ├── processed/                    # Compressed Parquet marts & model artifacts (<25 MB)
│   └── fallback/
│       └── genai_cache.json          # Deterministic GenAI cache for zero-key/offline resilience
├── looker/
│   ├── omni_growth.model.lkml        # LookML semantic model definition
│   └── ga4_funnel.view.lkml          # LookML GA4 funnel view with dimensions & measures
├── scripts/
│   ├── download_datasets.py          # Automated fetcher for GA4, Olist & A/B public samples
│   ├── build_data_mart.py            # DuckDB ETL creating Parquet files & Looker CSV mart
│   └── train_forecasts.py            # Offline Prophet, AutoARIMA & Bass S-Curve fitting
├── src/
│   ├── __init__.py
│   ├── config.py                     # Paths, constants, and memory guardrail settings
│   ├── database/
│   │   └── duckdb_manager.py         # Singleton read-only DuckDB connection (192 MB cap)
│   ├── analytics/
│   │   ├── ga4_funnel.py             # Funnel conversion, drop-offs & cohort retention logic
│   │   └── sql_queries.py            # Catalog of CTE & Window Function SQL queries
│   ├── experimentation/
│   │   ├── power_analysis.py         # MDE & sample size calculator
│   │   ├── frequentist_ab.py         # Two-sample tests & CUPED variance reduction
│   │   ├── sequential_testing.py     # O'Brien-Fleming alpha-spending boundaries
│   │   └── bayesian_ab.py            # Beta-Binomial & Normal conjugate posteriors + ROPE
│   ├── forecasting/
│   │   ├── time_series.py            # Prophet & AutoARIMA evaluation & scenario adjuster
│   │   └── s_curve.py                # Logistic & Bass Diffusion (p, q, M) curve fitting
│   ├── optimization/
│   │   └── prescriptive_solver.py    # SciPy SLSQP marketing & inventory allocation solver
│   ├── genai/
│   │   ├── sql_validator.py          # AST/Regex read-only SQL sanitizer
│   │   └── assistant.py              # Groq + Gemini client with deterministic fallback
│   └── reporting/
│       └── excel_generator.py        # OpenPyXL 5-tab workbook builder with live formulas
├── app/
│   ├── main.py                       # Unified Streamlit entrypoint & executive layout
│   └── components/
│       ├── tab_sql_ga4.py            # Tab 1: GA4 Funnel, SQL Lab & Looker Studio View
│       ├── tab_ab_testing.py         # Tab 2: Frequentist (CUPED) & Bayesian A/B Lab
│       ├── tab_forecasting.py        # Tab 3: Prophet, ARIMA & Bass S-Curve Scenario Planner
│       ├── tab_optimization.py       # Tab 4: Prescriptive Supply Chain & Budget Solver
│       └── tab_genai_excel.py        # Tab 5: NL-to-SQL Agent, Executive Memo & Excel Export
├── deploy/
│   ├── omnigrowth.service            # systemd unit with MemoryMax=350M & CPUQuota=50%
│   └── nginx_omnigrowth.conf         # Nginx reverse proxy config with WebSocket support
└── tests/
    ├── test_duckdb_queries.py        # Validates CTE & window function SQL execution
    ├── test_experimentation.py       # Validates CUPED variance reduction & Bayesian posteriors
    ├── test_forecasting_scurve.py    # Validates Bass Diffusion parameters & S-curve bounds
    ├── test_prescriptive_solver.py   # Validates SLSQP constraint satisfaction
    └── test_genai_and_excel.py       # Validates SQL injection blocking & .xlsx formula cells
```

---

## 7. Co-Existing VPS Deployment Specification (1 vCPU / 2 GB RAM)

Because your Ubuntu 24.04 VPS already hosts other live web applications, we avoid Docker daemon overhead and deploy directly via an isolated Python virtual environment (`venv`) managed by `systemd` resource control (`cgroups v2`):

* **Memory & CPU Isolation (`deploy/omnigrowth.service`):**
  * `MemoryHigh=280M` (throttles cache smoothly if approached)
  * `MemoryMax=350M` (strict hard ceiling so existing VPS workloads are never starved)
  * `CPUQuota=50%` (prevents a single analytical query from spiking 100% of your 1 vCPU)
* **Streamlit Runtime Flags:**
  * `--server.headless=true --server.address=127.0.0.1 --server.port=8501 --server.maxUploadSize=5 --browser.gatherUsageStats=false`
* **Nginx Reverse Proxy (`deploy/nginx_omnigrowth.conf`):**
  * Proxies traffic from a dedicated subdomain (e.g., `omnigrowth.<your-vps-ip>.sslip.io` or your custom domain) to `127.0.0.1:8501` with WebSocket upgrade headers (`Upgrade $http_upgrade`, `Connection "upgrade"`) and free Let's Encrypt TLS via `certbot --nginx`.