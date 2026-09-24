"""Run the full pipeline locally, in order.

    python src/run_pipeline.py            # every step
    python src/run_pipeline.py topics     # start from a given step

The same steps are wired into an Airflow DAG in dags/cset_sample_dag.py.
"""

import sys

import build_database
import build_views
import entity_resolution
import fetch_openalex
import llm_classify
import topics
from common import get_logger

log = get_logger("pipeline")

STEPS = [
    ("fetch", fetch_openalex.main),
    ("build_db", build_database.main),
    ("entity_resolution", entity_resolution.main),
    ("views", build_views.main),
    ("topics", topics.main),
    ("llm", llm_classify.main),
]


def main() -> None:
    names = [n for n, _ in STEPS]
    start = sys.argv[1] if len(sys.argv) > 1 else names[0]
    if start not in names:
        sys.exit(f"Unknown step '{start}'. Choose from: {', '.join(names)}")
    for name, fn in STEPS[names.index(start):]:
        log.info("=== %s ===", name)
        fn()
    log.info("Pipeline complete. Render the report with: quarto render")


if __name__ == "__main__":
    main()
