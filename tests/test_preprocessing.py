import numpy as np
import pandas as pd
import pytest

from catfood_unsupervised.preprocessing import (
    THAI_LIKERT_SCALE,
    impute_buy_factors,
    map_likert_value,
)


def test_map_likert_value_maps_thai_scale():
    assert map_likert_value("เห็นด้วยที่สุด", THAI_LIKERT_SCALE) == 5


def test_impute_buy_factors_returns_no_nan_values():
    buy_factors = pd.DataFrame(
        {
            "bf_a": [1.0, np.nan, 3.0, 4.0, 5.0],
            "bf_b": [2.0, 2.0, np.nan, 4.0, 5.0],
        }
    )

    imputed = impute_buy_factors(buy_factors, ["bf_a", "bf_b"])

    assert not imputed.isna().any().any()


def test_impute_buy_factors_returns_empty_frame_for_empty_input():
    buy_factors = pd.DataFrame(index=pd.RangeIndex(0), columns=["bf_a", "bf_b"])

    imputed = impute_buy_factors(buy_factors, ["bf_a", "bf_b"])

    assert imputed.empty
    assert imputed.columns.tolist() == ["bf_a", "bf_b"]
    assert imputed.index.tolist() == []


def test_impute_buy_factors_preserves_all_nan_selected_columns():
    buy_factors = pd.DataFrame(
        {
            "bf_a": [1.0, np.nan, 3.0],
            "bf_b": [np.nan, np.nan, np.nan],
        }
    )

    imputed = impute_buy_factors(buy_factors, ["bf_a", "bf_b"])

    assert imputed.columns.tolist() == ["bf_a", "bf_b"]
    assert imputed.index.tolist() == [0, 1, 2]
    assert imputed["bf_b"].isna().all()
    assert imputed["bf_a"].notna().all()
