# Walkthrough Guide: Building, Deploying & Presenting OmniGrowth-AI

**Domain Scope:** Full-Stack Product Growth, Experimentation & Prescriptive Analytics Suite  
**Local Dev Environment:** Ubuntu 24.04.2 LTS (16.0 GiB RAM, AMD Ryzen 5 PRO 3500U $\times 8$ cores) + Antigravity IDE  
**Production Environment:** Shared Ubuntu 24.04 VPS ($1\text{ vCPU} / 2\text{ GB RAM}$) running alongside existing web apps + Streamlit Community Cloud Backup  

---

## 1. How to Drive Antigravity IDE from Zero to Production

Before writing any code, create a new directory on your Ubuntu 24.04 laptop and place `AGENTS.md` (extracted from Section 2 of `implementation_plan.md`), `implementation_plan.md`, `task_checklist.md`, and `walkthrough_guide.md` into the root folder:

```bash
mkdir -p ~/projects/omnigrowth-ai
cd ~/projects/omnigrowth-ai
git init -b main
```

Open `~/projects/omnigrowth-ai` in **Antigravity IDE**. Instead of asking the AI agent to build the entire project in one massive prompt, feed it these exact **Milestone Prompts** one phase at a time:

### Prompt 1 (Phases 0 & 1 — Foundation, Data Marts & DuckDB)
> *"Read `AGENTS.md`, `implementation_plan.md`, and `task_checklist.md`. Execute **Phase 0** and **Phase 1** completely. Create the virtual environment configuration, `src/config.py`, `scripts/download_datasets.py`, `scripts/build_data_mart.py`, `src/database/duckdb_manager.py` (enforcing `SET memory_limit='192MB'; SET threads=1;`), and the LookML files in `looker/`. Run the data pipeline scripts, verify that `data/processed/` is under 25 MB, and mark Phases 0 and 1 complete in `task_checklist.md`."*

### Prompt 2 (Phases 2 & 3 — GA4 SQL Lab & Frequentist/Bayesian A/B Engine)
> *"Read `AGENTS.md` and `task_checklist.md`. Execute **Phase 2** (Module 1: `src/analytics/sql_queries.py` with multi-stage CTEs, `ROW_NUMBER`, `LAG`, `LEAD`, `NTILE`, and rolling windows, plus `src/analytics/ga4_funnel.py`) and **Phase 3** (Module 2: `power_analysis.py`, `frequentist_ab.py` with CUPED variance reduction, `sequential_testing.py` with O'Brien-Fleming boundaries, and `bayesian_ab.py` with analytical conjugate Beta-Binomial/Normal posteriors, ROPE, and Expected Loss). Write and run `pytest tests/test_duckdb_queries.py tests/test_experimentation.py -v` until 100% pass."*

### Prompt 3 (Phases 4 & 5 — Prophet/ARIMA, Bass S-Curves & Prescriptive Solver)
> *"Read `AGENTS.md` and `task_checklist.md`. Execute **Phase 4** (`src/forecasting/s_curve.py` fitting Logistic and Bass Diffusion models $p, q, M$; `src/forecasting/time_series.py` with Prophet and AutoARIMA; and `scripts/train_forecasts.py` to generate pre-computed artifacts in `data/processed/`) and **Phase 5** (`src/optimization/prescriptive_solver.py` using `scipy.optimize.minimize` SLSQP for regional budget and inventory allocation). Run `python scripts/train_forecasts.py` and `pytest tests/test_forecasting_scurve.py tests/test_prescriptive_solver.py -v` until 100% pass."*

### Prompt 4 (Phases 6, 7 & 8 — GenAI Guardrails, Live Excel Engine, Streamlit UI & Deployment Configs)
> *"Read `AGENTS.md` and `task_checklist.md`. Execute **Phase 6** (`src/genai/sql_validator.py`, `src/genai/assistant.py` with Groq/Gemini + deterministic cache `data/fallback/genai_cache.json`, and `src/reporting/excel_generator.py` building a 5-tab `.xlsx` workbook with live Excel formulas), **Phase 7** (the complete 5-tab Streamlit application in `app/`), and **Phase 8** (`Dockerfile`, `.dockerignore`, `requirements-serve.txt`, `deploy/docker-compose.coolify.yml`, optional `deploy/` bare-metal templates, and `README.md`). Run `pytest tests/ -v` and `ruff check .` to verify zero errors."*

---

## 2. Setting Up Free No-Credit-Card GenAI API Keys

Your GenAI module uses a 3-tier failover design (**Groq Free API $\rightarrow$ Google Gemini Free API $\rightarrow$ Deterministic Offline Cache**), meaning it works out of the box even with no keys, and never asks for a credit card:

1. **Groq Cloud Free Tier (`Llama-3.3-70B-Versatile`):**
   - Visit `https://console.groq.com/keys` and sign in with your GitHub or Google account (no credit card required).
   - Generate an API key and paste it into `.env` as `GROQ_API_KEY=gsk_...`.
2. **Google AI Studio Free Tier (`Gemini-2.5-Flash`):**
   - Visit `https://aistudio.google.com/app/apikey` and sign in with your Google account (no credit card required).
   - Click **Create API Key** and paste it into `.env` as `GEMINI_API_KEY=AIza...`.

---

## 3. Zero-Cost Google Looker Studio Integration (5-Minute Setup)

Pairing the repository's LookML semantic definitions (`looker/*.lkml`) with a live embedded **Google Looker Studio** report provides an interactive BI layer alongside the DuckDB SQL Lab:

1. Run `python scripts/build_data_mart.py` locally, which generates `data/processed/looker_executive_mart.csv`.
2. Go to `https://lookerstudio.google.com/` (100% free with any Google account) and click **Create $\rightarrow$ Report**.
3. Select **File Upload** connector and upload `data/processed/looker_executive_mart.csv`.
4. Add 3 clean visual cards:
   - **Scorecard Banner:** Total Sessions, Checkout Conversion Rate, Total Revenue, 30-Day Retention Rate.
   - **Bar/Funnel Chart:** Stage-by-Stage GA4 User Progression (`session_start` $\rightarrow$ `purchase`).
   - **Time-Series Chart:** Daily Regional Revenue by Device Category.
5. Click **File $\rightarrow$ Embed report $\rightarrow$ Enable embedding**, copy the Embed URL, and set `LOOKER_STUDIO_EMBED_URL` in your `.env` file. Tab 1 of your Streamlit app will render the interactive Looker Studio dashboard natively alongside your DuckDB SQL Lab.

---

## 4. Safe Co-Existing Deployment on Your Shared $1\text{ vCPU} / 2\text{ GB RAM}$ Ubuntu 24.04 VPS (via Coolify)

Because **Coolify** (Docker + Traefik reverse proxy) is already managing applications and TLS on your VPS, we deploy **OmniGrowth-AI** directly through Coolify using a **slim serving container** (`Dockerfile` + `requirements-serve.txt`) with strict container memory/CPU limits (`350M` RAM / `0.50` vCPU). Avoid running a separate bare-metal `nginx` or `certbot` on ports 80/443 so nothing conflicts with Coolify's Traefik proxy.

### Step 4.1: Ensure Your 2 GB VPS Has a 2 GB Swap File (Safety Buffer)
SSH into your VPS and verify swap is active so Docker builds in Coolify and background OS bursts never trigger OOM kills:
```bash
free -h
# If Swap is 0B, create a 2GB swapfile:
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Step 4.2: Why Coolify Uses `Dockerfile` + `requirements-serve.txt` (No Build OOM)
- Full training libraries (`Prophet`, `CmdStan`, `AutoARIMA`, `statsmodels`, `scikit-learn`) are listed in `requirements.txt` for **local offline training only**. Pre-computed forecasts and S-curve artifacts (`< 25 MB`) are already committed in `data/processed/`.
- The root [`Dockerfile`](Dockerfile) installs only [`requirements-serve.txt`](requirements-serve.txt) (`duckdb`, `pandas`, `numpy`, `scipy`, `openpyxl`, `plotly`, `streamlit`, `httpx`, `python-dotenv`), enforces `OMP_NUM_THREADS=1` / `OPENBLAS_NUM_THREADS=1` / `DUCKDB_MEMORY_LIMIT=192MB` / `DUCKDB_THREADS=1`, exposes port `8501`, and binds Streamlit to `0.0.0.0:8501` so Coolify's Traefik proxy can reach it over the Docker bridge network.

### Step 4.3: Deploy OmniGrowth-AI in the Coolify Dashboard
1. Push your latest `main` branch (including `Dockerfile`, `.dockerignore`, `requirements-serve.txt`, and `data/processed/*`) to GitHub.
2. Open **Coolify $\rightarrow$ Projects $\rightarrow$ + New Resource $\rightarrow$ Public / Private Git Repository** and select `omnigrowth-ai` (`Branch: main`).
3. Under **Configuration $\rightarrow$ General**:
   - **Build Pack:** `Dockerfile` *(or `Docker Compose` pointing to `deploy/docker-compose.coolify.yml`)*
   - **Port Exposes:** `8501`
   - **Domains:** Set your domain or wildcard DNS (e.g., `https://omnigrowth.yourdomain.com` or `https://omnigrowth.<YOUR-VPS-IP>.sslip.io`). Coolify's Traefik automatically provisions Let's Encrypt HTTPS and proxies Streamlit WebSockets (`/_stcore/stream`).
4. Under **Configuration $\rightarrow$ Environment Variables**, add:
   - `GROQ_API_KEY` (optional, free tier)
   - `GEMINI_API_KEY` (optional, free tier)
   - `LOOKER_STUDIO_EMBED_URL` (optional, your Looker Studio embed URL)
   - `DUCKDB_MEMORY_LIMIT=192MB`
   - `DUCKDB_THREADS=1`
5. Under **Configuration $\rightarrow$ Resource Limits** (critical for co-existing on 2 GB RAM):
   - **Memory Limit:** `350M` (`350m`)
   - **Memory Reservation / Soft Limit:** `256M`
   - **CPU Limit:** `0.5`
6. Click **Deploy** and monitor **Build & Container Logs** until the `Dockerfile` `HEALTHCHECK` (`/_stcore/health`) reports **Healthy**.

### Step 4.4: Verify Container Health & Memory Limits on the VPS
```bash
# Check container health and real-time RAM usage (expect ~150-240 MB active, < 300 MB peak)
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
```
*(Note: Legacy bare-metal templates [`deploy/omnigrowth.service`](deploy/omnigrowth.service) and [`deploy/nginx_omnigrowth.conf`](deploy/nginx_omnigrowth.conf) remain in `deploy/` solely as reference if migrating off Coolify.)*

---

## 5. End-to-End Verification Checklist

Run these checks locally and on your VPS before production release:

1. **Automated Unit & Integration Tests:**
   ```bash
   pytest tests/ -v
   ```
   - Confirms DuckDB CTE/Window queries execute in $< 200\text{ ms}$.
   - Confirms CUPED reduces variance by $1 - \rho_{X,Y}^2$ and Bayesian Expected Loss matches analytical integration.
   - Confirms Bass Diffusion S-curve recovers $(p, q, M)$ and `SciPy` SLSQP respects budget and inventory constraints.
   - Confirms SQL validator blocks destructive queries (`DROP`, `COPY`, `ALTER`) and `OpenPyXL` generates live Excel formulas.
2. **Coolify / VPS Memory Footprint Audit:**
   ```bash
   docker stats --no-stream
   ```
   - Verify active container memory stays between $140\text{ MB}$ and $260\text{ MB}$ (well below the $350\text{ MB}$ container limit) and check the sidebar **Live Process RSS Memory** gauge inside the live Streamlit UI (`< 300 MB`).
3. **Excel Formula Verification:**
   - Click **"Download 5-Tab Executive Financial Model (.xlsx)"** in Tab 5, open the file in LibreOffice Calc or Microsoft Excel, edit a baseline traffic assumption cell in `Tab 2 (GA4_Funnel_Economics)`, and verify that `Tab 1 (Executive_Summary)` automatically recalculates.

---

## 6. Architecture & Technical Deep-Dive Reference

### 6.1 System Overview
> **OmniGrowth-AI** connects D2C digital user acquisition and funnel analytics with regional B2B/B2C demand forecasting and inventory allocation. Rather than leaving models in isolated Jupyter notebooks, it provides an end-to-end production system backed by DuckDB OLAP SQL, Frequentist CUPED and Bayesian experimentation, Prophet/AutoARIMA and Bass Diffusion S-curve growth models, a SciPy prescriptive resource solver, and a GenAI workflow automation engine that exports live multi-tab Excel financial models.

### 6.2 Module-by-Module Technical Design Rationale

| Technical Domain | Design Question | Architectural Solution |
| :--- | :--- | :--- |
| **Advanced SQL (CTEs & Window Functions)** | *"How do you build a strict closed conversion funnel in SQL over raw GA4 event logs without double-counting out-of-order events?"* | **Tab 1 SQL Inspector**: Chains CTEs (`WITH step_1 ... step_5`) using `ROW_NUMBER() OVER (PARTITION BY user_pseudo_id, ga_session_id ORDER BY event_timestamp)` to isolate the earliest valid timestamp per session stage, and uses `LAG()` to measure inter-purchase velocity. |
| **Frequentist vs. Bayesian A/B Testing** | *"When would you use Frequentist CUPED vs. Bayesian A/B testing, and how do you handle sequential monitoring?"* | **Tab 2**: **CUPED** uses pre-experiment spend $X$ to subtract out baseline variance ($\hat{Y}^{\text{CUPED}} = Y - \theta(X - \bar{X})$), shrinking confidence intervals by $1 - \rho^2$ so experiments finish faster. **O'Brien-Fleming sequential boundaries** prevent Type I error inflation during interim looks, while **Bayesian Expected Loss** and **$ROPE$** translate statistical uncertainty directly into decision risk. |
| **Prophet, ARIMA & S-Curve Growth** | *"How do short-term time-series forecasts (Prophet/ARIMA) connect to long-term market adoption S-curves?"* | **Tab 3**: `Prophet` and `AutoARIMA` capture short-to-medium horizon seasonality, holiday spikes, and promo elasticity, whereas the **Bass Diffusion S-curve** models long-term structural market saturation ($M$), decomposing growth into external marketing innovation ($p$) and word-of-mouth network imitation ($q$), pinpointing the exact inflection time $t^* = \frac{\ln(q/p)}{p+q}$. |
| **Predictive vs. Prescriptive Analytics** | *"How do you turn a regional demand forecast into a prescriptive supply chain and marketing decision?"* | **Tab 4**: Constrained `SciPy` SLSQP optimization model that ingests the $80\%$ and $95\%$ forecast quantiles alongside regional S-curve elasticities to allocate finite marketing budget $B_{\text{total}}$ and warehouse inventory $I_{\text{warehouse}}$, maximizing net margin while enforcing safety-stock floors. |
| **GenAI Automation & Executive Excel** | *"How do you prevent hallucinations when applying GenAI to analytics workflows, and how do you deliver live financial models in Excel?"* | **Tab 5**: The LLM never invents numbers—it writes read-only DuckDB SQL validated by an AST/regex sanitizer (`src/genai/sql_validator.py`) and summarizes deterministic model outputs into a Board Memo, paired with an automated `OpenPyXL` engine that generates a 5-tab `.xlsx` workbook populated with live Excel formulas (`=NORM.S.DIST`, `=INDEX/MATCH`, dynamic CIs). |