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