# Contributing to OmniGrowth-AI

Thank you for contributing to **OmniGrowth-AI**. To keep the codebase reliable, modular, and within our production memory budget ($< 300\text{ MB}$ RSS on a $1\text{ vCPU} / 2\text{ GB RAM}$ VPS), all contributors follow the workflow and engineering standards below.

---

## 1. Branching Strategy (Trunk-Based Feature Workflow)

`main` is protected and always deployable. Direct pushes to `main` are disabled.

1. Sync your local `main` branch:
   ```bash
   git checkout main
   git pull origin main
   ```
2. Create a short-lived branch named after the change type and scope:
   - `feat/<short-slug>` — new analytical module, UI component, or pipeline feature
   - `fix/<short-slug>` — bug fix or numerical stability patch
   - `chore/<short-slug>` — CI/CD, dependency, or tooling maintenance
   - `docs/<short-slug>` — documentation or LookML catalog updates
   - `test/<short-slug>` — unit or integration test additions

---

## 2. Commit Message Convention (Conventional Commits)

All commits and Pull Request titles must follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```text
<type>(<optional-scope>): <imperative summary under 72 chars>

[optional body explaining why the change was made]

[optional footer(s), e.g., Closes #12]
```

**Allowed types:**
- `feat`: A new feature or analytical capability
- `fix`: A bug fix
- `perf`: A performance or memory footprint improvement
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `test`: Adding or updating unit/integration tests
- `docs`: Documentation-only changes
- `ci`: Changes to GitHub Actions workflows or deployment templates
- `chore`: Build scripts, dependency bumps, or repo configuration

**Examples:**
- `feat(experimentation): implement CUPED variance reduction estimator`
- `fix(database): clamp DuckDB memory_limit env override to 192MB ceiling`
- `chore(ci): add weekly dependabot grouping for minor pip updates`

---

## 3. Architectural & Code Quality Guardrails

Every change must respect the project invariants defined in [AGENTS.md](AGENTS.md):

1. **Offline-Training / Online-Serving Split:** Never fit heavy `Prophet` or `AutoARIMA` models synchronously inside a Streamlit request thread on raw multi-million-row tables.
2. **DuckDB Memory Discipline:** Every connection in `src/database/duckdb_manager.py` must initialize with `SET memory_limit = '192MB'; SET threads = 1;` and query `.parquet` artifacts via zero-copy views.
3. **File & Code Modularity:**
   - Every Python file must remain **under 300 lines of code**.
   - Use Python 3.11+ type annotations (`typing`, `dataclasses`, or `pydantic`) and Google-style docstrings.
   - `src/` modules must **never** import `streamlit`.
4. **GenAI Security:** All LLM-generated SQL must pass `src/genai/sql_validator.py` (read-only `SELECT` / `WITH` only).

---

## 4. Local Verification Before Opening a Pull Request

Run both quality gates locally and ensure zero errors before pushing your branch:

```bash
# 1. Static analysis & formatting check
.venv/bin/ruff check src/ app/ tests/ scripts/

# 2. Full automated test suite (must pass 100%)
.venv/bin/pytest tests/ -v
```

---

## 5. Pull Request Expectations

1. Open a Pull Request targeting `main` using the repository's [Pull Request Template](.github/PULL_REQUEST_TEMPLATE.md).
2. Link the relevant issue (`Closes #<issue-number>`).
3. Ensure all GitHub Actions status checks (`lint-and-format` and `test`) pass.
4. Resolve all review conversations before merging via **Squash and merge** or **Rebase and merge**.
