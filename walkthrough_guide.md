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
> *"Read `AGENTS.md` and `task_checklist.md`. Execute **Phase 6** (`src/genai/sql_validator.py`, `src/genai/assistant.py` with Groq/Gemini + deterministic cache `data/fallback/genai_cache.json`, and `src/reporting/excel_generator.py` building a 5-tab `.xlsx` workbook with live Excel formulas), **Phase 7** (the complete 5-tab Streamlit application in `app/`), and **Phase 8** (`deploy/omnigrowth.service`, `deploy/nginx_omnigrowth.conf`, and `README.md`). Run `pytest tests/ -v` and `ruff check .` to verify zero errors."*

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

## 4. Safe Co-Existing Deployment on Your Shared $1\text{ vCPU} / 2\text{ GB RAM}$ Ubuntu 24.04 VPS

Because your VPS already runs other web applications, we deploy **OmniGrowth-AI** using native Linux `cgroups v2` resource isolation inside `systemd` (no heavy Docker daemon overhead, and all heavy model training artifacts are already pre-computed in `data/processed/*.parquet`).

### Step 4.1: Ensure Your 2 GB VPS Has a 2 GB Swap File (Safety Buffer)
SSH into your VPS and verify swap is active so background OS bursts never trigger OOM kills:
```bash
free -h
# If Swap is 0B, create a 2GB swapfile:
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Step 4.2: Clone Repository & Install Lightweight Runtime on VPS
Note that on the VPS, because `Prophet` and `AutoARIMA` forecasts are pre-computed on your 16 GB laptop and committed as compact `.parquet` artifacts ($< 25\text{ MB}$), the VPS only needs fast analytical serving packages:
```bash
sudo mkdir -p /var/www/omnigrowth-ai
sudo chown -R $USER:$USER /var/www/omnigrowth-ai
git clone https://github.com/<your-username>/omnigrowth-ai.git /var/www/omnigrowth-ai
cd /var/www/omnigrowth-ai

python3 -m venv .venv
source .venv/bin/activate
pip install --no-cache-dir -r requirements.txt
cp .env.example .env
nano .env  # Paste your free GROQ_API_KEY and GEMINI_API_KEY
```

### Step 4.3: Install Cgroup-Isolated `systemd` Service (`deploy/omnigrowth.service`)
Create `/etc/systemd/system/omnigrowth.service` with strict memory (`MemoryMax=350M`) and CPU (`CPUQuota=50%`) caps so your existing web apps on the VPS are never impacted:
```ini
[Unit]
Description=OmniGrowth-AI Full-Stack Data Science Suite (Streamlit + DuckDB)
After=network.target

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/var/www/omnigrowth-ai
EnvironmentFile=/var/www/omnigrowth-ai/.env
Environment="MALLOC_TRIM_THRESHOLD_=65536"
Environment="PYTHONUNBUFFERED=1"

# Strict cgroups v2 isolation for shared 1 vCPU / 2 GB RAM VPS
MemoryAccounting=true
MemoryHigh=280M
MemoryMax=350M
CPUAccounting=true
CPUQuota=50%

ExecStart=/var/www/omnigrowth-ai/.venv/bin/streamlit run app/main.py \
    --server.port=8501 \
    --server.address=127.0.0.1 \
    --server.headless=true \
    --server.fileWatcherType=none \
    --server.maxUploadSize=5 \
    --browser.gatherUsageStats=false

Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo chown -R www-data:www-data /var/www/omnigrowth-ai
sudo systemctl daemon-reload
sudo systemctl enable omnigrowth.service
sudo systemctl start omnigrowth.service
sudo systemctl status omnigrowth.service
```

### Step 4.4: Configure Nginx Reverse Proxy (`deploy/nginx_omnigrowth.conf`)
Pick an unused port internally (`127.0.0.1:8501`) and expose it on a dedicated subdomain (either `omnigrowth.yourdomain.com` or free wildcard DNS `omnigrowth.<YOUR-VPS-IP>.sslip.io`):
```nginx
server {
    listen 80;
    server_name omnigrowth.<YOUR-VPS-IP>.sslip.io;

    client_max_body_size 5M;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Host $host;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }

    location /_stcore/stream {
        proxy_pass http://127.0.0.1:8501/_stcore/stream;
        proxy_http_version 1.1;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Host $host;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
```

Enable the site and provision free HTTPS via Let's Encrypt:
```bash
sudo ln -s /etc/nginx/sites-available/omnigrowth.conf /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d omnigrowth.<YOUR-VPS-IP>.sslip.io
```

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
2. **VPS Memory Footprint Audit:**
   ```bash
   systemctl status omnigrowth.service | grep Memory
   ```
   - Verify active memory stays between $140\text{ MB}$ and $240\text{ MB}$ (well below the $350\text{ MB}$ hard ceiling).
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