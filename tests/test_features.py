import pandas as pd
import pytest
from train import DATA, load_and_engineer


def test_engineered_features_and_no_leakage():
    df = load_and_engineer()
    assert len(df) == 7043 and df["churn"].isin([0, 1]).all()
    assert not df.isna().any().any()
    assert {"n_services", "charge_gap", "tenure_bucket", "is_month_to_month", "auto_pay"} <= set(df.columns)
    assert "Churn" not in df.columns and "customerID" not in df.columns


@pytest.mark.parametrize("column,value", [("Churn", "Unknown"), ("TotalCharges", "not-a-number")])
def test_invalid_labels_and_nonzero_tenure_charges_are_not_silently_recoded(tmp_path, column, value):
    data = pd.read_csv(DATA)
    row = data[data.tenure > 0].head(1).copy()
    row[column] = value
    path = tmp_path / "bad.csv"
    row.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_and_engineer(path)


def test_duplicate_customers_cannot_enter_multiple_cv_partitions(tmp_path):
    row = pd.read_csv(DATA).head(1)
    path = tmp_path / "duplicate.csv"
    pd.concat([row, row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="identities"):
        load_and_engineer(path)
