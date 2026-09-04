"""DVC stage: cleaned parquet -> pickled features + fitted transformers."""
import pickle
import pandas as pd
from prodml.data import split, featurize
import structlog
import pathlib
Path = pathlib.Path

log = structlog.get_logger(__name__)

INPUT = "data/interim/clean.parquet"
OUTPUT = "data/processed/features.pkl"
TEST_RAW = "data/processed/test_raw.parquet"


def main():
    df = pd.read_parquet(INPUT)
    df_train, df_val, df_test = split(df)
    bundle = featurize(df_train, df_val, df_test)

    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "wb") as f:
        pickle.dump(bundle, f)

    df_test.to_parquet(TEST_RAW, index=False)

    log.info("featurize_done", train_shape=bundle[0].shape, test_rows=len(df_test))

if __name__ == "__main__":
    main()