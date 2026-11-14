"""Read-only SQL AST/regex sanitizer for GenAI natural-language-to-SQL outputs.

Enforces ``AGENTS.md`` security guardrails before any LLM-generated query runs on DuckDB:
- Must be a single statement beginning with ``SELECT`` or ``WITH`` (CTEs).
- Rejects multiple semicolons and all DDL, DML, file-I/O, extension, or pragma keywords
  (``DROP``, ``ALTER``, ``TRUNCATE``, ``INSERT``, ``UPDATE``, ``DELETE``, ``CREATE``,
  ``REPLACE``, ``COPY``, ``ATTACH``, ``DETACH``, ``INSTALL``, ``LOAD``, ``PRAGMA``,
  ``CALL``, ``EXPORT``, ``IMPORT``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

FORBIDDEN_SQL_KEYWORDS: Final[tuple[str, ...]] = (
    "DROP",
    "ALTER",
    "TRUNCATE",
    "INSERT",
    "UPDATE",
    "DELETE",
    "CREATE",
    "REPLACE",
    "COPY",
    "ATTACH",
    "DETACH",
    "INSTALL",
    "LOAD",
    "PRAGMA",
    "CALL",
    "EXPORT",
    "IMPORT",
    "VACUUM",
    "SET",
)

ALLOWED_SQL_VIEWS: Final[tuple[str, ...]] = (
    "v_ga4_events",
    "v_regional_demand",
    "v_ab_logs",
    "v_mql_funnel",
)

_BLOCK_COMMENT_RE: Final[re.Pattern[str]] = re.compile(r"/\*[\s\S]*?\*/")
_LINE_COMMENT_RE: Final[re.Pattern[str]] = re.compile(r"--[^\n]*")
_STRING_LITERAL_RE: Final[re.Pattern[str]] = re.compile(r"'(?:''|[^'])*'")
_FORBIDDEN_KW_RE: Final[re.Pattern[str]] = re.compile(
    r"\b(" + "|".join(FORBIDDEN_SQL_KEYWORDS) + r")\b",
    re.IGNORECASE,
)


class SQLValidationError(ValueError):
    """Raised when an input SQL query violates read-only execution guardrails."""


@dataclass(frozen=True)
class SQLValidationResult:
    """Structured result of SQL read-only validation."""

    is_valid: bool
    normalized_sql: str
    statement_type: str
    reason: str = "ok"


def strip_sql_comments(sql: str) -> str:
    """Removes block (``/* ... */``) and line (``-- ...``) comments from SQL text."""
    no_block = _BLOCK_COMMENT_RE.sub(" ", sql or "")
    no_line = _LINE_COMMENT_RE.sub(" ", no_block)
    return no_line.strip()


def extract_sql_from_markdown(text: str) -> str:
    """Extracts the SQL statement from optional ```sql ... ``` markdown fences."""
    raw = (text or "").strip()
    fence_match = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", raw, re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    return raw


def validate_readonly_sql(query_str: str) -> str:
    """Validates and normalizes a read-only DuckDB ``SELECT`` or ``WITH`` query.

    Args:
        query_str: Raw SQL string (may include markdown fences or comments).

    Returns:
        Normalized single-statement SQL without trailing semicolon.

    Raises:
        SQLValidationError: If the statement is empty, contains multiple statements,
            does not start with ``SELECT`` / ``WITH``, or uses forbidden keywords.
    """
    unwrapped = extract_sql_from_markdown(query_str)
    cleaned = strip_sql_comments(unwrapped)
    if not cleaned:
        raise SQLValidationError("SQL query must not be empty.")

    # Normalize a single trailing semicolon; reject embedded semicolons cleanly.
    normalized = cleaned[:-1].strip() if cleaned.endswith(";") else cleaned
    outside_strings = _STRING_LITERAL_RE.sub(" '' ", normalized)

    if ";" in outside_strings:
        raise SQLValidationError("Multiple SQL statements or embedded semicolons are forbidden.")

    first_token = normalized.split(None, 1)[0].upper()
    if first_token not in {"SELECT", "WITH"}:
        raise SQLValidationError(f"Only SELECT or WITH queries are permitted; got {first_token!r}.")

    if match := _FORBIDDEN_KW_RE.search(outside_strings):
        kw = match.group(1).upper()
        raise SQLValidationError(f"Forbidden SQL keyword detected: {kw!r}.")

    # Ensure balanced parentheses for basic AST sanity.
    depth = 0
    for ch in outside_strings:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                raise SQLValidationError("Unbalanced closing parenthesis in SQL query.")
    if depth != 0:
        raise SQLValidationError("Unbalanced opening parenthesis in SQL query.")

    return normalized


def inspect_sql(query_str: str) -> SQLValidationResult:
    """Non-raising wrapper returning a :class:`SQLValidationResult`."""
    try:
        normalized = validate_readonly_sql(query_str)
        stmt_type = normalized.split(None, 1)[0].upper()
        return SQLValidationResult(is_valid=True, normalized_sql=normalized, statement_type=stmt_type)
    except SQLValidationError as exc:
        return SQLValidationResult(is_valid=False, normalized_sql="", statement_type="INVALID", reason=str(exc))
