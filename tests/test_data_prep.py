"""Testy jednostkowe modułu data_prep.py (uruchomienie: pytest)."""

import numpy as np
import pandas as pd
import pytest

from data_prep import load_data, split_data


@pytest.fixture
def sample_df():
    """Syntetyczny zbiór: 100 wierszy, 3 cechy, 3 klasy (40/30/30)."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(100, 3)), columns=["f1", "f2", "f3"])
    df["target"] = [0] * 40 + [1] * 30 + [2] * 30
    return df


# --- split_data ---------------------------------------------------------------


def test_split_test_size_proportion(sample_df):
    X_train, X_test, y_train, y_test = split_data(sample_df, test_size=0.2, seed=42)
    assert len(X_test) == pytest.approx(len(sample_df) * 0.2, abs=1)


def test_split_sizes_sum_to_total(sample_df):
    X_train, X_test, y_train, y_test = split_data(sample_df, test_size=0.3, seed=42)
    assert len(X_train) + len(X_test) == len(sample_df)
    assert len(y_train) + len(y_test) == len(sample_df)


def test_split_separates_target_from_features(sample_df):
    X_train, X_test, y_train, y_test = split_data(sample_df, test_size=0.2, seed=42)
    assert list(X_train.columns) == ["f1", "f2", "f3"]
    assert "target" not in X_test.columns
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_split_is_reproducible_for_same_seed(sample_df):
    first = split_data(sample_df, test_size=0.2, seed=42)
    second = split_data(sample_df, test_size=0.2, seed=42)
    pd.testing.assert_frame_equal(first[1], second[1])


def test_split_differs_for_different_seed(sample_df):
    _, X_test_a, _, _ = split_data(sample_df, test_size=0.2, seed=1)
    _, X_test_b, _, _ = split_data(sample_df, test_size=0.2, seed=2)
    assert not X_test_a.index.equals(X_test_b.index)


def test_split_preserves_class_proportions(sample_df):
    _, _, _, y_test = split_data(sample_df, test_size=0.2, seed=42)
    # 40/30/30 przy 20% testu => 8/6/6 próbek na klasę
    assert y_test.value_counts().sort_index().tolist() == [8, 6, 6]


def test_split_does_not_modify_input(sample_df):
    original = sample_df.copy()
    split_data(sample_df, test_size=0.2, seed=42)
    pd.testing.assert_frame_equal(sample_df, original)


def test_split_uses_explicit_target_column(sample_df):
    df = sample_df.rename(columns={"target": "label"})
    X_train, _, y_train, _ = split_data(df, test_size=0.2, seed=42, target_column="label")
    assert y_train.name == "label"
    assert "label" not in X_train.columns


# --- split_data: niepoprawne wejście --------------------------------------------


def test_split_empty_dataframe_raises():
    empty = pd.DataFrame(columns=["f1", "target"])
    with pytest.raises(ValueError):
        split_data(empty, test_size=0.2, seed=42)


@pytest.mark.parametrize("bad_test_size", [0, 1, -0.1, 1.5])
def test_split_invalid_test_size_raises(sample_df, bad_test_size):
    with pytest.raises(ValueError):
        split_data(sample_df, test_size=bad_test_size, seed=42)


def test_split_missing_target_column_raises(sample_df):
    with pytest.raises(ValueError):
        split_data(sample_df, test_size=0.2, seed=42, target_column="nie_ma_takiej")


def test_split_wrong_input_type_raises():
    with pytest.raises(TypeError):
        split_data([[1, 2, 0], [3, 4, 1]], test_size=0.2, seed=42)


# --- load_data ------------------------------------------------------------------


def test_load_data_returns_dataframe(tmp_path, sample_df):
    csv_path = tmp_path / "data.csv"
    sample_df.to_csv(csv_path, index=False)

    loaded = load_data(csv_path)

    assert isinstance(loaded, pd.DataFrame)
    pd.testing.assert_frame_equal(loaded, sample_df)


def test_load_data_has_no_side_effects(tmp_path, sample_df):
    csv_path = tmp_path / "data.csv"
    sample_df.to_csv(csv_path, index=False)

    load_data(csv_path)

    # funkcja niczego nie zapisała obok wczytywanego pliku
    assert [p.name for p in tmp_path.iterdir()] == ["data.csv"]


def test_load_data_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "nie_istnieje.csv")


def test_load_data_empty_file_raises(tmp_path):
    empty_path = tmp_path / "empty.csv"
    empty_path.write_text("")
    with pytest.raises(ValueError):
        load_data(empty_path)


def test_load_data_header_only_raises(tmp_path):
    header_only = tmp_path / "header_only.csv"
    header_only.write_text("f1,f2,target\n")
    with pytest.raises(ValueError):
        load_data(header_only)
