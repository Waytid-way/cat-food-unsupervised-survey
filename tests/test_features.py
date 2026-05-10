import numpy as np
import pandas as pd
import pytest

from catfood_unsupervised.features import build_vote_features, ipsatize_rows


def test_build_vote_features_applies_rank_weights():
    rankings = pd.DataFrame({"rank1": [3], "rank2": [2], "rank3": [1]})

    vote_features = build_vote_features(
        rankings, rank_columns=("rank1", "rank2", "rank3"), option_count=3
    )

    assert vote_features.loc[0, "vote_03"] == 3
    assert vote_features.loc[0, "vote_02"] == 2
    assert vote_features.loc[0, "vote_01"] == 1


def test_build_vote_features_parses_option_vote_strings():
    rankings = pd.DataFrame(
        {"rank1": ["Option 10"], "rank2": ["Option 2"], "rank3": ["Option 1"]}
    )

    vote_features = build_vote_features(
        rankings, rank_columns=("rank1", "rank2", "rank3"), option_count=10
    )

    assert vote_features.loc[0, "vote_10"] == 3
    assert vote_features.loc[0, "vote_02"] == 2
    assert vote_features.loc[0, "vote_01"] == 1


def test_build_vote_features_rejects_non_integral_numeric_rankings():
    rankings = pd.DataFrame({"rank1": [2.9], "rank2": [2], "rank3": [1]})

    with pytest.raises(ValueError, match="rank1"):
        build_vote_features(rankings, rank_columns=("rank1", "rank2", "rank3"), option_count=3)


def test_build_vote_features_rejects_duplicate_rankings_in_one_row():
    rankings = pd.DataFrame({"rank1": [2], "rank2": [2], "rank3": [1]})

    with pytest.raises(ValueError, match="duplicate"):
        build_vote_features(rankings, rank_columns=("rank1", "rank2", "rank3"), option_count=3)


def test_build_vote_features_rejects_out_of_range_options():
    rankings = pd.DataFrame({"rank1": [10], "rank2": [2], "rank3": [1]})

    with pytest.raises(ValueError, match="out of range"):
        build_vote_features(rankings, rank_columns=("rank1", "rank2", "rank3"), option_count=3)


def test_ipsatize_rows_centers_each_row_mean_at_zero():
    option_matrix = np.array([[5.0, 3.0, 1.0], [4.0, 4.0, 4.0]])

    ipsatized = ipsatize_rows(option_matrix)

    assert ipsatized.mean(axis=1)[0] == pytest.approx(0.0)
    assert ipsatized.mean(axis=1)[1] == pytest.approx(0.0)
