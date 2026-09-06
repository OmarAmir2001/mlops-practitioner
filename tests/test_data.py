import numpy as np
import pandas as pd

from prodml.data import NUMERICAL, featurize, prepare, split


def test_prepare_lowercases_columns(raw_csv):
    df = prepare(raw_csv)
    assert "customerid" in df.columns
    assert all(c == c.lower() for c in df.columns)


def test_prepare_lowercases_values(raw_csv):
    df = prepare(raw_csv)
    assert set(df.gender.unique()) <= {"female", "male"}
    assert "month-to-month" in df.contract.unique()


def test_prepare_replaces_spaces_in_values(raw_csv):
    df = prepare(raw_csv)
    assert "no_phone_service" in df.multiplelines.unique()
    assert "electronic_check" in df.paymentmethod.unique()


def test_blank_totalcharges_becomes_zero(raw_csv):
    """Telco has blank TotalCharges for customers with tenure 0."""
    df = prepare(raw_csv)
    assert df.totalcharges.isna().sum() == 0
    assert (df.totalcharges == 0).any()


def test_totalcharges_is_numeric(raw_csv):
    df = prepare(raw_csv)
    assert pd.api.types.is_numeric_dtype(df.totalcharges)


def test_churn_is_binary(raw_csv):
    df = prepare(raw_csv)
    assert set(df.churn.unique()) <= {0, 1}


def test_split_is_deterministic(raw_csv):
    df = prepare(raw_csv)
    a, _, _ = split(df, seed=42)
    b, _, _ = split(df, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_split_different_seeds_differ(raw_csv):
    df = prepare(raw_csv)
    a, _, _ = split(df, seed=42)
    b, _, _ = split(df, seed=7)
    assert not a.index.equals(b.index)


def test_split_preserves_all_rows(raw_csv):
    df = prepare(raw_csv)
    train, val, test = split(df)
    assert len(train) + len(val) + len(test) == len(df)


def test_split_has_no_overlap(raw_csv):
    df = prepare(raw_csv)
    train, val, test = split(df)
    assert set(train.index).isdisjoint(val.index)
    assert set(train.index).isdisjoint(test.index)
    assert set(val.index).isdisjoint(test.index)


def test_featurize_returns_eight_objects(raw_csv):
    df = prepare(raw_csv)
    bundle = featurize(*split(df))
    assert len(bundle) == 8


def test_featurize_consistent_widths(raw_csv):
    df = prepare(raw_csv)
    X_train, X_val, X_test, *_ = featurize(*split(df))
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]


def test_featurize_row_counts_match_splits(raw_csv):
    df = prepare(raw_csv)
    train, val, test = split(df)
    X_train, X_val, X_test, y_train, y_val, y_test, _, _ = featurize(train, val, test)
    assert X_train.shape[0] == len(y_train) == len(train)
    assert X_val.shape[0] == len(y_val) == len(val)
    assert X_test.shape[0] == len(y_test) == len(test)


def test_scaler_fit_on_train_only(raw_csv):
    """No leakage: the scaler must learn from train, not the whole dataset."""
    df = prepare(raw_csv)
    train, val, test = split(df)
    *_, scaler = featurize(train, val, test)
    np.testing.assert_allclose(scaler.mean_, train[NUMERICAL].mean().values, rtol=1e-6)


def test_featurize_does_not_mutate_inputs(raw_csv):
    """featurize() copies; the caller's frames must be untouched."""
    df = prepare(raw_csv)
    train, val, test = split(df)
    before = train[NUMERICAL].copy()
    featurize(train, val, test)
    pd.testing.assert_frame_equal(train[NUMERICAL], before)


def test_dv_feature_names_cover_all_columns(raw_csv):
    df = prepare(raw_csv)
    *_, dv, _ = featurize(*split(df))
    names = list(dv.get_feature_names_out())
    for col in NUMERICAL:
        assert col in names
