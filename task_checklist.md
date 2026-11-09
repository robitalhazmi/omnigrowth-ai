# Task Checklist: OmniGrowth-AI (Full-Stack Product Growth, Experimentation & Prescriptive Analytics Suite)

**Domain Scope:** Omnichannel Retail, D2C Digital Product Growth & Supply Chain Analytics  
**Local Dev Machine:** Ubuntu 24.04.2 LTS (16.0 GiB RAM, AMD Ryzen 5 PRO 3500U $\times 8$ threads)  
**Production Target:** Shared Ubuntu 24.04 VPS ($1\text{ vCPU} / 2\text{ GB RAM}$ with `MemoryMax=350M` cgroup isolation) + Streamlit Community Cloud Backup  

---

## How to Use This Task File in Antigravity IDE

1. Keep `AGENTS.md`, `implementation_plan.md`, and `task_checklist.md` in the root directory of your repository.
2. In Antigravity IDE, instruct the agent phase by phase:  
   > *"Read `AGENTS.md`, `implementation_plan.md`, and `task_checklist.md`. Execute all unchecked items in **Phase X**, run the verification gate commands, fix any failing tests, and mark the checkboxes `[x]` in `task_checklist.md` once verified."*
3. Commit and push to GitHub at the end of every phase.

---

## Phase 0: Repository Initialization, Workspace Guardrails & Environment Setup

- [x] **0.1 Initialize Git Repository & Directory Structure**
  - Create standard directories: `data/raw/`, `data/processed/`, `data/fallback/`, `looker/`, `scripts/`, `src/database/`, `src/analytics/`, `src/experimentation/`, `src/forecasting/`, `src/optimization/`, `src/genai/`, `src/reporting/`, `app/components/`, `deploy/`, and `tests/`.
  - Create `.gitignore` excluding `.venv/`, `data/raw/`, `.env`, `__pycache__/`, `.pytest_cache/`, and `*.pyc` while tracking `data/processed/*.parquet`, `data/processed/looker_executive_mart.csv`, and `data/fallback/genai_cache.json`.

- [x] **0.2 Configure `AGENTS.md` & Environment Templates**
  - Save the `AGENTS.md` workspace rules from `implementation_plan.md` in the root folder.
  - Create `.env.example` with placeholders:
    ```ini
    GROQ_API_KEY=your_free_groq_api_key_here
    GEMINI_API_KEY=your_free_gemini_api_key_here
    DUCKDB_MEMORY_LIMIT=192MB
    DUCKDB_THREADS=1
    LOOKER_STUDIO_EMBED_URL=https://lookerstudio.google.com/embed/reporting/your-report-id/page/p1
    ```

- [x] **0.3 Configure Dependencies (`requirements.txt` & `pyproject.toml`)**
  - Create `requirements.txt` with lightweight, CPU-friendly production packages:
    - `duckdb>=1.1.0`
    - `pandas>=2.2.0`
    - `pyarrow>=17.0.0`
    - `numpy>=1.26.0`
    - `scipy>=1.13.0`
    - `statsmodels>=0.14.2`
    - `scikit-learn>=1.5.0`
    - `prophet>=1.1.5`
    - `statsforecast>=1.7.6`
    - `openpyxl>=3.1.5`
    - `plotly>=5.23.0`
    - `streamlit>=1.38.0`
    - `httpx>=0.27.0`
    - `pydantic>=2.8.0`
    - `python-dotenv>=1.0.1`
    - `pytest>=8.3.0`
    - `ruff>=0.6.0`
  - Create `Makefile` with targets: `setup`, `data`, `train`, `test`, `lint`, and `run`.

- [x] **0.4 Create Global Configuration (`src/config.py`)**
  - Define project root paths, Parquet file paths, DuckDB memory limit (`192MB`), thread count ($1$), and default model hyperparameters using `pydantic` or `dataclasses`.

- [x] **0.5 Phase 0 Verification Gate**
  - Run `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
  - Verify `pytest --version` and `ruff --version` execute without errors.

---

## Phase 1: Real Benchmark Dataset Pipeline, DuckDB OLAP Marts & Looker Bridge

- [x] **1.1 Build Automated Public Dataset Downloader (`scripts/download_datasets.py`)**
  - Fetch or assemble the three real public benchmark datasets into `data/raw/` (with deterministic seed fallback if offline):
    1. **GA4 Ecommerce Event Export Sample** (`ga4_obfuscated_sample_ecommerce` schema: `event_date`, `event_timestamp`, `event_name`, `user_pseudo_id`, `ga_session_id`, `device_category`, `traffic_source`, `traffic_medium`, `geo_region`, `item_category`, `event_value_in_usd`).
    2. **Olist Regional E-Commerce & Marketing Funnel Dataset** (daily order demand, freight cost, promotional discount percentage, and B2B/D2C MQL onboarding timestamps across regions).
    3. **E-Commerce Randomized A/B Testing Dataset** (`user_id`, `experiment_group`, `assigned_date`, `device_os`, `region_tier`, `pre_exp_spend`, `converted`, `post_exp_revenue`, `retention_d7`, `retention_d30`).

- [x] **1.2 Build DuckDB ETL & Parquet Data Mart Script (`scripts/build_data_mart.py`)**
  - Transform raw tables into Zstandard-compressed Parquet files in `data/processed/` totaling $< 25\text{ MB}$:
    - `data/processed/ga4_events.parquet`
    - `data/processed/regional_demand.parquet`
    - `data/processed/ab_experiment_logs.parquet`
  - Export `data/processed/looker_executive_mart.csv` containing aggregated daily funnel conversions, regional revenue, and cohort retention metrics for **Google Looker Studio**.

- [x] **1.3 Implement Low-Memory DuckDB Manager (`src/database/duckdb_manager.py`)**
  - Create a thread-safe DuckDB connection manager that initializes with:
    ```sql
    SET memory_limit = '192MB';
    SET threads = 1;
    ```
  - Register zero-copy virtual views (`v_ga4_events`, `v_regional_demand`, `v_ab_logs`) pointing directly to `data/processed/*.parquet`.

- [x] **1.4 Create LookML Semantic Model Templates (`looker/`)**
  - Create `looker/omni_growth.model.lkml` and `looker/ga4_funnel.view.lkml` defining dimensions (`event_date`, `geo_region`, `device_category`, `traffic_source`) and measures (`session_Starts`, `cart_additions`, `checkout_rate`, `purchase_conversion_rate`, `average_order_value`).

- [x] **1.5 Phase 1 Verification Gate**
  - Run `python scripts/download_datasets.py && python scripts/build_data_mart.py`.
  - Verify `du -sh data/processed/` is under $25\text{ MB}$ and all three `.parquet` files plus `looker_executive_mart.csv` exist.

---

## Phase 2: Module 1 — Advanced SQL Lab & GA4 Funnel Analytics

- [x] **2.1 Build Parameterized Advanced SQL Catalog (`src/analytics/sql_queries.py`)**
  - Write production-documented DuckDB SQL queries showcasing advanced analytical SQL constructs:
    1. **Strict Chronological Closed-Funnel SQL:** Uses multi-stage CTEs (`WITH s1 AS (...), s2 AS (...)`) and `ROW_NUMBER() OVER (PARTITION BY user_pseudo_id, ga_session_id ORDER BY event_timestamp)` across the 5 GA4 funnel stages (`session_start`, `view_item`, `add_to_cart`, `begin_checkout`, `purchase`).
    2. **User Purchase Velocity & Inter-Arrival Lag SQL:** Uses `LAG(event_date) OVER (PARTITION BY user_pseudo_id ORDER BY event_date)` and `LEAD()` to measure repeat purchase intervals and churn risk by cohort.
    3. **RFM Customer Segmentation & Rolling Window SQL:** Uses `NTILE(4) OVER (ORDER BY total_spend DESC)` for quartile segmentation and `AVG(daily_revenue) OVER (PARTITION BY geo_region ORDER BY event_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)` for 7-day moving averages.

- [x] **2.2 Implement GA4 Funnel & Cohort Analytics Service (`src/analytics/ga4_funnel.py`)**
  - Implement functions returning structured DataFrames and summary KPIs:
    - `get_funnel_conversion_summary(region, device, start_date, end_date)`
    - `get_dropoff_attribution_by_segment(dimension)`
    - `get_cohort_retention_matrix()`
    - `execute_readonly_sql_lab(query_str)`

- [x] **2.3 Write Unit Tests (`tests/test_duckdb_queries.py`)**
  - Test that all catalog SQL queries execute in $< 200\text{ ms}$, return non-empty DataFrames, enforce monotonic funnel counts ($N_{\text{session}} \ge N_{\text{view}} \ge N_{\text{cart}} \ge N_{\text{checkout}} \ge N_{\text{purchase}}$), and respect the DuckDB `192MB` memory limit.

- [x] **2.4 Phase 2 Verification Gate**
  - Run `pytest tests/test_duckdb_queries.py -v` and confirm $100\%$ pass rate.

---

## Phase 3: Module 2 — Frequentist (CUPED, Sequential) & Bayesian A/B Testing Engine

- [x] **3.1 Implement Sample Size & Power Calculator (`src/experimentation/power_analysis.py`)**
  - Implement two-proportion and two-sample continuous power calculators computing minimum sample size $n$ per arm given baseline rate $p_1$, Minimum Detectable Effect ($MDE$), significance level $\alpha \in [0.01, 0.10]$, and power $1 - \beta \in [0.70, 0.95]$.
  - Generate power curves across varying sample sizes and MDE thresholds.

- [x] **3.2 Implement Frequentist A/B & CUPED Variance Reduction (`src/experimentation/frequentist_ab.py`)**
  - Implement standard two-sample $Z$-test for conversion rate and Welch's $t$-test for continuous revenue.
  - Implement **CUPED (Controlled-experiment Using Pre-Experiment Data)**:
    $$\hat{Y}_i^{\text{CUPED}} = Y_i - \theta (X_i - \bar{X}), \quad \theta = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}$$
  - Return both unadjusted and CUPED-adjusted lift estimates, standard errors, $p$-values, $95\%$ confidence intervals, and percentage variance reduction ($1 - \rho_{X,Y}^2$).

- [x] **3.3 Implement O'Brien-Fleming Sequential Testing (`src/experimentation/sequential_testing.py`)**
  - Compute daily cumulative $Z$-statistics over experiment duration $t_k = k/K \in (0, 1]$ and compare against O'Brien-Fleming $\alpha$-spending boundaries:
    $$\alpha(t_k) = 2 - 2\Phi\left(\frac{Z_{1 - \alpha/2}}{\sqrt{t_k}}\right)$$
  - Flag early stopping for efficacy or futility without inflating Type I error.

- [x] **3.4 Implement Analytical Conjugate Bayesian A/B Engine (`src/experimentation/bayesian_ab.py`)**
  - Implement Beta-Binomial conjugate updating ($\text{Beta}(\alpha_0 + s, \beta_0 + n - s)$) for conversion rates and Normal-Inverse-Gamma updating for continuous revenue using `scipy.stats` (zero MCMC RAM overhead).
  - Compute and return:
    - Posterior probability treatment beats control: $P(p_B > p_A)$
    - Expected Loss of choosing Treatment vs. Control: $\mathbb{E}[\max(p_A - p_B, 0)]$
    - Probability inside the Region of Practical Equivalence ($ROPE$, default $[-0.2\%, +0.2\%]$)
    - $95\%$ Highest Density Credible Intervals ($HDI$).

- [x] **3.5 Write Unit Tests (`tests/test_experimentation.py`)**
  - Verify CUPED strictly reduces standard error when $\text{Corr}(X, Y) > 0.3$.
  - Verify Bayesian $P(p_B > p_A) \in [0, 1]$, Expected Loss $\ge 0$, and sequential boundaries converge to $Z_{1-\alpha/2}$ at $t_K = 1.0$.

- [x] **3.6 Phase 3 Verification Gate**
  - Run `pytest tests/test_experimentation.py -v` and confirm $100\%$ pass rate.

---

## Phase 4: Module 3 — Time-Series Forecasting (`Prophet`, `AutoARIMA`) & S-Curve Growth Modeling

- [ ] **4.1 Implement S-Curve Market Adoption Modeler (`src/forecasting/s_curve.py`)**
  - Implement non-linear least squares curve fitting (`scipy.optimize.curve_fit`) with bounded parameters for:
    1. **Logistic Growth Model:** $N(t) = \frac{M}{1 + \exp(-k(t - t_0))}$
    2. **Bass Diffusion Model:** $N(t) = M \cdot \frac{1 - \exp(-(p+q)t)}{1 + (q/p)\exp(-(p+q)t)}$
  - Extract and return carrying capacity $M$ (TAM ceiling), innovation coefficient $p$ (marketing effect), imitation coefficient $q$ (network effect), peak adoption timing $t^* = \frac{\ln(q/p)}{p+q}$, and goodness-of-fit $R^2$.

- [ ] **4.2 Implement Time-Series Forecasting & Scenario Engine (`src/forecasting/time_series.py`)**
  - Implement offline training wrapper comparing **Meta `Prophet`** (with seasonality, Indonesian national holidays, and promotional discount regressor) and **Nixtla `StatsForecast` (`AutoARIMA`)**.
  - Implement rolling-origin cross-validation computing $MAPE$, $RMSE$, $sMAPE$, and $80\% / 95\%$ empirical prediction interval coverage.
  - Implement fast online scenario adjuster (`apply_scenario_shock(base_forecast_df, spend_change_pct, promo_discount_pct, conversion_lift_pct)`) that recalculates forecast trajectories and shifts Bass S-curve parameters ($p \rightarrow p(1 + \delta_{\text{spend}})$) in $< 50\text{ ms}$.

- [ ] **4.3 Build Offline Model Training Script (`scripts/train_forecasts.py`)**
  - Train `Prophet`, `AutoARIMA`, and Bass Diffusion S-Curve models across all regional cohorts on the local laptop and save pre-computed forecast trajectories, residuals, and S-curve parameter tables to:
    - `data/processed/forecasts_pretrained.parquet`
    - `data/processed/scurve_params.parquet`
    - `data/processed/model_backtest_metrics.parquet`

- [ ] **4.4 Write Unit Tests (`tests/test_forecasting_scurve.py`)**
  - Test Bass Diffusion parameter recovery on synthetic noise-free S-curve series ($p=0.03, q=0.38, M=10000$) within $5\%$ tolerance.
  - Test that pre-computed forecast artifacts contain valid $80\%$ and $95\%$ confidence bounds ($\hat{y}_{\text{lower,95}} \le \hat{y}_{\text{lower,80}} \le \hat{y} \le \hat{y}_{\text{upper,80}} \le \hat{y}_{\text{upper,95}}$).

- [ ] **4.5 Phase 4 Verification Gate**
  - Run `python scripts/train_forecasts.py && pytest tests/test_forecasting_scurve.py -v` and confirm $100\%$ pass rate.

---

## Phase 5: Module 4 — Prescriptive Supply Chain & Marketing Budget Optimization

- [ ] **5.1 Implement Constrained Prescriptive Solver (`src/optimization/prescriptive_solver.py`)**
  - Formulate the multi-region marketing spend ($b_r$) and inventory stock ($I_r$) optimization problem using `scipy.optimize.minimize(..., method='SLSQP')`.
  - Maximize expected regional net contribution margin minus inventory holding cost ($c_{\text{hold}}$) and stockout penalty ($c_{\text{short}}$) subject to:
    - Total promotional budget cap: $\sum_{r=1}^R b_r \le B_{\text{total}}$
    - Total warehouse supply capacity cap: $\sum_{r=1}^R I_r \le I_{\text{warehouse}}$
    - Regional service-level safety stock floor: $I_r \ge \hat{D}_{r, 0.80}(b_r)$
  - Return a comparison DataFrame of **Baseline vs. Optimal Allocation** per region, expected incremental revenue, stockout reduction percentage, and ROI lift.

- [ ] **5.2 Write Unit Tests (`tests/test_prescriptive_solver.py`)**
  - Verify solver converges (`success == True`), respects total budget $\sum b_r \le B_{\text{total}} + 10^{-4}$, satisfies regional safety stock floors, and achieves net contribution margin $\ge$ equal-weight baseline allocation.

- [ ] **5.3 Phase 5 Verification Gate**
  - Run `pytest tests/test_prescriptive_solver.py -v` and confirm $100\%$ pass rate.

---

## Phase 6: Module 5 — Resilient GenAI Automation & Multi-Tab Excel Engine

- [ ] **6.1 Implement Read-Only SQL AST/Regex Sanitizer (`src/genai/sql_validator.py`)**
  - Validate any natural-language-generated SQL query before execution:
    - Must begin with `SELECT` or `WITH`.
    - Must reject any statement containing multiple semicolons or forbidden keywords (`DROP`, `ALTER`, `TRUNCATE`, `INSERT`, `UPDATE`, `DELETE`, `CREATE`, `REPLACE`, `COPY`, `ATTACH`, `DETACH`, `INSTALL`, `LOAD`, `PRAGMA`, `CALL`, `EXPORT`).

- [ ] **6.2 Implement Multi-Provider GenAI Assistant with Offline Fallback (`src/genai/assistant.py`)**
  - Implement `GenAIAssistant` supporting:
    1. **Groq Cloud API** (`llama-3.3-70b-versatile`) via `httpx` with $10\text{ s}$ timeout.
    2. **Google Gemini API** (`gemini-2.5-flash`) automatic failover.
    3. **Deterministic Offline Cache** (`data/fallback/genai_cache.json`) containing pre-computed executive memos and NL-to-SQL answers for 10 curated executive prompts so users never hit an error if API keys are missing or rate-limited.
  - Implement two workflows:
    - `ask_analytics_sql(user_question)` $\rightarrow$ Generates DuckDB SQL $\rightarrow$ Validates via `sql_validator.py` $\rightarrow$ Executes on DuckDB $\rightarrow$ Returns SQL, DataFrame, and executive interpretation.
    - `generate_executive_memo(kpi_payload)` $\rightarrow$ Produces a structured C-Suite Board Memo in Markdown.

- [ ] **6.3 Implement Automated Multi-Tab Excel Workbook Generator (`src/reporting/excel_generator.py`)**
  - Use `OpenPyXL` to dynamically build an in-memory `BytesIO` `.xlsx` workbook with corporate styling, number formatting, and **live Excel formulas**:
    - **Tab 1 (`Executive_Summary`):** Cross-sheet formula links (`=GA4_Funnel_Economics!E8`, `=AB_Test_Model!D10`, `=Prescriptive_Allocation!F12`).
    - **Tab 2 (`GA4_Funnel_Economics`):** Step conversion formulas (`=C6/C5`), drop-off rate (`=1-D6`), and revenue-at-risk sensitivity formula.
    - **Tab 3 (`AB_Test_Model`):** Live Excel $Z$-score, two-tailed $p$-value (`=2*(1-NORM.S.DIST(ABS(D8),TRUE))`), and annualized revenue lift formula.
    - **Tab 4 (`Forecast_And_SCurve`):** Regional forecast series, live Bass Diffusion formula column, and an embedded `OpenPyXL` `LineChart`.
    - **Tab 5 (`Prescriptive_Allocation`):** Baseline vs. Optimal budget/inventory table with `SUM` totals and `% Variance` formulas (`=(D5-C5)/C5`).

- [ ] **6.4 Write Unit Tests (`tests/test_genai_and_excel.py`)**
  - Verify `sql_validator.py` blocks `DROP TABLE v_ga4_events;` and `SELECT * FROM v_ga4_events; COPY ...` while allowing multi-CTE `WITH ... SELECT` queries.
  - Verify `GenAIAssistant` seamlessly returns cached output when API keys are unset.
  - Verify `excel_generator.py` produces a valid `.xlsx` stream containing all 5 worksheet names and formula strings starting with `'='`.

- [ ] **6.5 Phase 6 Verification Gate**
  - Run `pytest tests/test_genai_and_excel.py -v` and confirm $100\%$ pass rate.

---

## Phase 7: Unified Executive Streamlit Application (`app/`)

- [ ] **7.1 Build Main Executive App Shell & Sidebar (`app/main.py`)**
  - Configure page title (`OmniGrowth-AI | Full-Stack Product Growth & Prescriptive Analytics`), global filters (Region, Date Range, Product Segment), and a **"Data Lineage, Architecture & VPS Memory Monitor"** sidebar badge displaying current process RSS memory in MB (`psutil` / `/proc/self/statm`).

- [ ] **7.2 Build Tab 1: GA4 User Journey, Advanced SQL Lab & Looker Studio View (`app/components/tab_sql_ga4.py`)**
  - Render interactive Plotly Funnel chart, segment drop-off heatmap, cohort retention matrix, an expandable **"SQL Lab & Live Query Inspector"** drawer (displaying and executing the CTE + Window Function SQL), and an embedded **Google Looker Studio** executive view toggle.

- [ ] **7.3 Build Tab 2: Dual-Paradigm A/B Testing & Experimentation Lab (`app/components/tab_ab_testing.py`)**
  - Render side-by-side **Frequentist (Standard vs. CUPED Variance Reduction + O'Brien-Fleming Sequential Boundary)** and **Analytical Bayesian (Posterior Distributions, $P(\text{Treatment} > \text{Control})$, Expected Loss, and $ROPE$)** interactive dashboards plus an interactive Sample Size / MDE Power Calculator.

- [ ] **7.4 Build Tab 3: Time-Series (`Prophet` vs. `AutoARIMA`) & S-Curve Scenario Planner (`app/components/tab_forecasting.py`)**
  - Render regional forecast comparisons with $80\% / 95\%$ confidence ribbons, rolling backtest accuracy table ($MAPE$, $RMSE$, $sMAPE$), and **Bass Diffusion vs. Logistic S-Curve** plots marking the exact inflection point $t^*$ and carrying capacity $M$.
  - Connect interactive executive sliders (*Marketing Spend % Change*, *Promo Discount %*, *Onboarding Lift %*) for real-time what-if simulation.

- [ ] **7.5 Build Tab 4: Prescriptive Supply Chain & Marketing Optimizer (`app/components/tab_optimization.py`)**
  - Allow executives to set Total Marketing Budget ($B_{\text{total}}$) and Total Warehouse Inventory ($I_{\text{warehouse}}$), run the `SciPy` SLSQP optimizer live, and visualize the optimal regional reallocation waterfall and stockout risk reduction.

- [ ] **7.6 Build Tab 5: GenAI Workflow Automation & One-Click Excel Workbook (`app/components/tab_genai_excel.py`)**
  - Render the **Natural-Language-to-DuckDB SQL Agent** chat interface, the **One-Click C-Suite Board Memo Generator**, and the **Download 5-Tab Executive Financial Model (`.xlsx`)** button.

- [ ] **7.7 Phase 7 Verification Gate**
  - Run full test suite: `pytest tests/ -v` ($100\%$ pass rate) and `ruff check src/ app/ tests/`.
  - Launch app locally (`streamlit run app/main.py`) and verify RSS memory consumption stays $< 250\text{ MB}$.

---

## Phase 8: Co-Existing VPS Deployment ($1\text{ vCPU} / 2\text{ GB RAM}$) & Cloud Backup

- [ ] **8.1 Create `systemd` Cgroup-Isolated Service (`deploy/omnigrowth.service`)**
  - Configure `MemoryHigh=280M`, `MemoryMax=350M`, and `CPUQuota=50%` binding Streamlit to `127.0.0.1:8501` so existing VPS applications are completely protected.

- [ ] **8.2 Create Nginx Reverse Proxy Config (`deploy/nginx_omnigrowth.conf`)**
  - Configure virtual host for `omnigrowth.<vps-ip>.sslip.io` (or custom subdomain) with WebSocket upgrade headers and Let's Encrypt TLS (`certbot --nginx`).

- [ ] **8.3 Polish Portfolio `README.md` & Deploy Live**
  - Document architecture diagrams, live VPS URL, Streamlit Community Cloud backup URL, Looker Studio link, and mathematical methodology in `README.md`.