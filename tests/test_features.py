from train import load_and_engineer


def test_engineered_features_and_no_leakage():
    df = load_and_engineer()
    assert len(df) == 7043 and df["churn"].isin([0, 1]).all()
    assert not df.isna().any().any()
    assert {"n_services", "charge_gap", "tenure_bucket", "is_month_to_month", "auto_pay"} <= set(df.columns)
    assert "Churn" not in df.columns and "customerID" not in df.columns
