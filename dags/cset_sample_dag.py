"""Airflow DAG: the same pipeline, scheduled and deployed the way it would run in production.

Locally the steps run with `python src/run_pipeline.py`. In a GCP deployment
(Cloud Composer), each task would write to BigQuery instead of DuckDB and raw
files would land in a Cloud Storage bucket; the step logic stays the same.

    fetch_openalex > build_database > entity_resolution > build_views > topics > [llm_classify, quality_checks]
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import build_database  # noqa: E402
import build_views  # noqa: E402
import entity_resolution  # noqa: E402
import fetch_openalex  # noqa: E402
import llm_classify  # noqa: E402
import topics  # noqa: E402
from common import connect  # noqa: E402


def quality_checks() -> None:
    """Fail the run loudly if the data looks wrong, instead of publishing bad numbers."""
    con = connect(read_only=True)
    checks = {
        "works table is not empty": "SELECT COUNT(*) > 0 FROM works",
        "no duplicate work ids": "SELECT COUNT(*) = COUNT(DISTINCT work_id) FROM works",
        "at least 80% of works have an abstract":
            "SELECT AVG(CASE WHEN abstract IS NOT NULL THEN 1 ELSE 0 END) >= 0.8 FROM works",
        "country credit sums to at most 1 per paper":
            "SELECT MAX(s) <= 1.0001 FROM (SELECT SUM(weight) s FROM author_countries GROUP BY work_id)",
        "every work has a cluster":
            "SELECT COUNT(*) = 0 FROM works LEFT JOIN work_topics USING (work_id) WHERE cluster IS NULL",
    }
    failed = [name for name, sql in checks.items() if not con.execute(sql).fetchone()[0]]
    con.close()
    if failed:
        raise ValueError(f"Data quality checks failed: {failed}")


default_args = {"retries": 2, "retry_delay": timedelta(minutes=10)}

with DAG(
    dag_id="cset_sample_ai_biotech_bibliometrics",
    description="OpenAlex ingest, entity resolution, topic emergence and LLM annotation",
    start_date=datetime(2026, 1, 1),
    schedule="@monthly",   # OpenAlex updates continuously; a monthly refresh keeps trends current
    catchup=False,
    default_args=default_args,
    tags=["bibliometrics", "openalex", "biotech"],
) as dag:
    t_fetch = PythonOperator(task_id="fetch_openalex", python_callable=fetch_openalex.main)
    t_db = PythonOperator(task_id="build_database", python_callable=build_database.main)
    t_er = PythonOperator(task_id="entity_resolution", python_callable=entity_resolution.main)
    t_views = PythonOperator(task_id="build_views", python_callable=build_views.main)
    t_topics = PythonOperator(task_id="topics", python_callable=topics.main)
    t_llm = PythonOperator(task_id="llm_classify", python_callable=llm_classify.main)
    t_qc = PythonOperator(task_id="quality_checks", python_callable=quality_checks)

    t_fetch >> t_db >> t_er >> t_views >> t_topics >> [t_llm, t_qc]
