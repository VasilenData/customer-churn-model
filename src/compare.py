"""Model comparison, calibration and a cost-based decision threshold for the churn model.

Run: python src/compare.py        (writes outputs/model_comparison.json and two charts)

The business assumptions (offer cost, offer success rate, customer lifetime value) are
ILLUSTRATIVE parameters, set at the top of this file. Change them to match a real business.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from train import OUT, SEED, load_and_engineer

# --- illustrative business assumptions -------------------------------------------------
OFFER_COST = 60.0         # cost of a retention offer per contacted customer (e.g. a discount)
OFFER_SUCCESS = 0.25      # share of contacted churners the offer actually keeps
VALUE_MONTHS = 12         # value of a saved customer = monthly charge x this many months
# ----------------------------------------------------------------------------------------


def preprocessor(X, scale):
    cat = X.select_dtypes(include="object").columns.tolist()
    num = [c for c in X.columns if c not in cat]
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat),
                              ("num", StandardScaler() if scale else "passthrough", num)])


def models(X):
    with open(os.path.join(OUT, "metrics.json")) as f:
        best = {k.split("__", 1)[1]: v for k, v in json.load(f)["best_params"].items()}
    return {
        "Logistic regression": Pipeline([("pre", preprocessor(X, True)),
                                         ("m", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "Random forest (tuned)": Pipeline([("pre", preprocessor(X, False)),
                                           ("m", RandomForestClassifier(random_state=SEED, n_jobs=-1, **best))]),
        "Gradient boosting": Pipeline([("pre", preprocessor(X, False)),
                                       ("m", HistGradientBoostingClassifier(random_state=SEED, class_weight="balanced",
                                                                            max_depth=4, learning_rate=0.05, max_iter=200))]),
    }


def net_value(y_true, proba, monthly, threshold):
    """Expected net value of contacting every customer scored at or above the threshold."""
    contact = proba >= threshold
    saved = contact & (y_true == 1)
    gain = (saved * monthly * VALUE_MONTHS * OFFER_SUCCESS).sum()
    return float(gain - contact.sum() * OFFER_COST)


def main():
    df = load_and_engineer()
    X, y = df.drop(columns="churn"), df["churn"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    rows, fitted = [], {}
    for name, pipe in models(X).items():
        cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
        pipe.fit(X_tr, y_tr)
        p = pipe.predict_proba(X_te)[:, 1]
        pred = (p >= 0.5).astype(int)
        fitted[name] = (pipe, p)
        rows.append({"model": name, "cv_roc_auc_mean": round(cv_auc.mean(), 4), "cv_roc_auc_std": round(cv_auc.std(), 4),
                     "test_roc_auc": round(roc_auc_score(y_te, p), 4), "test_recall": round(recall_score(y_te, pred), 4),
                     "test_precision": round(precision_score(y_te, pred), 4), "test_f1": round(f1_score(y_te, pred), 4),
                     "brier": round(brier_score_loss(y_te, p), 4)})
    best = max(rows, key=lambda r: r["cv_roc_auc_mean"])["model"]  # chosen on CV, not on the test set
    pipe, p_raw = fitted[best]

    # calibration of the chosen model (fit on training data only)
    cal = CalibratedClassifierCV(models(X)[best], method="isotonic", cv=5).fit(X_tr, y_tr)
    p_cal = cal.predict_proba(X_te)[:, 1]
    calib = {"model": best, "brier_raw": round(brier_score_loss(y_te, p_raw), 4),
             "brier_calibrated": round(brier_score_loss(y_te, p_cal), 4)}

    # cost-based threshold, chosen on training-set out-of-fold scores, evaluated on the test set
    from sklearn.model_selection import cross_val_predict
    oof = cross_val_predict(CalibratedClassifierCV(models(X)[best], method="isotonic", cv=5), X_tr, y_tr, cv=cv,
                            method="predict_proba", n_jobs=-1)[:, 1]
    grid = np.round(np.arange(0.05, 0.95, 0.01), 2)
    tr_vals = [net_value(y_tr.values, oof, X_tr["MonthlyCharges"].values, t) for t in grid]
    t_star = float(grid[int(np.argmax(tr_vals))])
    te_at = lambda t: net_value(y_te.values, p_cal, X_te["MonthlyCharges"].values, t)
    pred_star = (p_cal >= t_star).astype(int)
    decision = {"threshold_chosen_on_train": t_star,
                "test_net_value_at_threshold": round(te_at(t_star), 0),
                "test_net_value_at_0.5": round(te_at(0.5), 0),
                "test_net_value_contact_everyone": round(te_at(0.0), 0),
                "test_recall_at_threshold": round(recall_score(y_te, pred_star), 4),
                "test_precision_at_threshold": round(precision_score(y_te, pred_star), 4),
                "assumptions": {"offer_cost": OFFER_COST, "offer_success_rate": OFFER_SUCCESS, "value_months": VALUE_MONTHS}}

    # model-agnostic importance on held-out data
    pi = permutation_importance(pipe, X_te, y_te, scoring="roc_auc", n_repeats=10, random_state=SEED, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=X_te.columns).sort_values(ascending=False)
    importance = {k: round(float(v), 4) for k, v in imp.head(8).items()}

    result = {"comparison": rows, "chosen_by_cv": best, "calibration": calib, "decision": decision,
              "permutation_importance_auc_drop": importance}
    json.dump(result, open(os.path.join(OUT, "model_comparison.json"), "w"), indent=2)
    print(json.dumps(result, indent=2))

    # charts
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for lbl, p in [("raw", p_raw), ("isotonic calibrated", p_cal)]:
        fr, mp = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
        ax[0].plot(mp, fr, marker="o", label=lbl)
    ax[0].plot([0, 1], [0, 1], "--", color="grey"); ax[0].set_xlabel("Predicted churn probability")
    ax[0].set_ylabel("Observed churn rate"); ax[0].set_title(f"Calibration: {best}"); ax[0].legend()
    te_vals = [te_at(t) for t in grid]
    ax[1].plot(grid, te_vals); ax[1].axvline(t_star, color="grey", ls="--", label=f"threshold {t_star} (chosen on train)")
    ax[1].set_xlabel("Decision threshold"); ax[1].set_ylabel("Illustrative net value (test set)")
    ax[1].set_title("Cost-based threshold"); ax[1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "calibration_and_threshold.png"), dpi=120); plt.close()


if __name__ == "__main__":
    main()
