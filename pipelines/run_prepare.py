"""DVC stage: raw CSV -> cleaned parquet."""

import structlog

from prodml.config import get_settings
from prodml.data import prepare

log = structlog.get_logger(__name__)

OUTPUT = "data/interim/clean.parquet"


def main():
    settings = get_settings()
    df = prepare(settings.DATA_PATH)
    df.to_parquet(OUTPUT, index=False)
    log.info("prepare_done", rows=len(df), output=OUTPUT)


if __name__ == "__main__":
    main()
