"""Shared helpers: configuration, paths, logging and database access."""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def project_path(relative: str) -> Path:
    """Resolve a path from config.yaml against the project root and make sure its folder exists."""
    p = ROOT / relative
    target_dir = p if p.suffix == "" else p.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    return p


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def connect(cfg: dict | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    cfg = cfg or load_config()
    return duckdb.connect(str(project_path(cfg["paths"]["duckdb"])), read_only=read_only)


def run_sql_file(con: duckdb.DuckDBPyConnection, name: str, **params):
    """Run a named query from sql/ and return a DataFrame.

    Queries live in plain .sql files so they can be reviewed on their own and
    reused in the report. Simple {placeholder} substitution is used for config values.
    """
    sql = (ROOT / "sql" / f"{name}.sql").read_text(encoding="utf-8")
    if params:
        sql = sql.format(**params)
    return con.execute(sql).df()
