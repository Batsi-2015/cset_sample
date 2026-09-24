"""Step 3b: Create reusable SQL views (sql/views.sql) on top of the pipeline tables."""

from common import ROOT, connect, get_logger, load_config

log = get_logger("views")


def main() -> None:
    con = connect(load_config())
    con.execute((ROOT / "sql" / "views.sql").read_text(encoding="utf-8"))
    log.info("Views created: v_country_credit, v_work_collab")
    con.close()


if __name__ == "__main__":
    main()
