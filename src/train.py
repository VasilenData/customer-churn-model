"""Customer churn model: feature engineering + Random Forest on the IBM Telco churn dataset.

Run: python src/train.py        (writes outputs/metrics.json and outputs/*.png)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "telco_churn.csv")
OUT = os.path.join(ROOT, "outputs")
SEED = 42
SERVICES = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


def load_and_engineer(path: str = DATA) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["TotalCharges"].isna(), "TotalCharges"] = 0.0  # new customers, tenure == 0
    df["churn"] = (df["Churn"] == "Yes").astype(int)

    # engineered features
    df["n_services"] = sum((df[c] == "Yes").astype(int) for c in SERVICES) + (df["InternetService"] != "No").astype(int)
    df["charge_per_tenure_month"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    df["charge_gap"] = df["MonthlyCharges"] - df["charge_per_tenure_month"]
    df["tenure_bucket"] = pd.cut(df["tenure"], [-1, 6, 12, 24, 48, 100], labels=["0-6", "7-12", "13-24", "25-48", "49+"]).astype(str)
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    df["auto_pay"] = df["PaymentMethod"].str.contains("automatic").astype(int)
    return df.drop(columns=["customerID", "Churn"])


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    cat = X.select_dtypes(include="object").columns.tolist()
    num = [c for c in X.columns if c not in cat]
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat)], remainder="passthrough")
    return Pipeline([("pre", pre), ("rf", RandomForestClassifier(random_state=SEED, n_jobs=-1))]), cat, num


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    df = load_and_engineer()
    X, y = df.drop(columns="churn"), df["churn"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    pipe, cat, num = build_pipeline(X)
    grid = {"rf__n_estimators": [200, 400], "rf__max_depth": [6, 10, None],
            "rf__min_samples_leaf": [1, 5, 10], "rf__class_weight": [None, "balanced"]}
    search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1)
    search.fit(X_tr, y_tr)
    model = search.best_estimator_

    proba = model.predict_proba(X_te)[:, 1]
    # pick the decision threshold on the TRAIN data only (best F1 via CV is overkill; use 0.5 and report both)
    pred = (proba >= 0.5).astype(int)
    base_acc = max(y_te.mean(), 1 - y_te.mean())
    metrics = {
        "dataset_rows": int(len(df)), "churn_rate": round(float(y.mean()), 4),
        "test_rows": int(len(y_te)),
        "majority_class_baseline_accuracy": round(float(base_acc), 4),
        "accuracy": round(accuracy_score(y_te, pred), 4),
        "precision": round(precision_score(y_te, pred), 4),
        "recall": round(recall_score(y_te, pred), 4),
        "f1": round(f1_score(y_te, pred), 4),
        "roc_auc": round(roc_auc_score(y_te, proba), 4),
        "best_params": search.best_params_, "cv_roc_auc": round(float(search.best_score_), 4),
    }
    json.dump(metrics, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_te, pred, target_names=["stay", "churn"]))

    # plots
    names = model.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(model.named_steps["rf"].feature_importances_, index=names).sort_values().tail(12)
    imp.index = [i.replace("cat__", "").replace("remainder__", "") for i in imp.index]
    imp.plot.barh(figsize=(7, 5), color="#4c7a34"); plt.title("Top 12 feature importances"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "feature_importance.png"), dpi=120); plt.close()

    fpr, tpr, _ = roc_curve(y_te, proba)
    plt.figure(figsize=(5, 5)); plt.plot(fpr, tpr, color="#4c7a34", label=f"AUC = {metrics['roc_auc']}")
    plt.plot([0, 1], [0, 1], "--", color="grey"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.legend(); plt.title("ROC curve (test set)"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "roc_curve.png"), dpi=120); plt.close()

    cm = confusion_matrix(y_te, pred)
    plt.figure(figsize=(4.5, 4)); plt.imshow(cm, cmap="Greens")
    for (i, j), v in np.ndenumerate(cm): plt.text(j, i, v, ha="center", va="center")
    plt.xticks([0, 1], ["stay", "churn"]); plt.yticks([0, 1], ["stay", "churn"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion matrix"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "confusion_matrix.png"), dpi=120); plt.close()


if __name__ == "__main__":
    main()
