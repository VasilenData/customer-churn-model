"""Bounded real-data nested logistic-policy evaluation; does not rerun historical RF grids."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from compare import net_value, preprocessor, select_threshold
from nested import nested_policy_scores
from train import DATA, SEED, load_and_engineer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("existing evidence is immutable; choose a new --out directory")
    source_sha = hashlib.sha256(Path(DATA).read_bytes()).hexdigest()
    data = load_and_engineer()
    X, y = data.drop(columns="churn"), data.churn
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)
    candidates = {f"logistic_C={c}": Pipeline([("pre", preprocessor(X_train, True)),
                  ("model", LogisticRegression(C=c, max_iter=2000, random_state=SEED))]) for c in (.1, 1., 10.)}
    oof, audit = nested_policy_scores(candidates, X_train, y_train, outer_splits=3, inner_splits=3, seed=SEED)
    threshold, policy_fit_value = select_threshold(y_train, oof, X_train.MonthlyCharges)
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
    selection = {name: float(np.mean(cross_val_score(estimator, X_train, y_train, scoring="roc_auc", cv=cv,
                                                    n_jobs=1, error_score="raise"))) for name, estimator in candidates.items()}
    selected = max(sorted(selection), key=selection.get)
    fitted = CalibratedClassifierCV(candidates[selected], method="isotonic", cv=cv).fit(X_train, y_train)
    predicted = fitted.predict_proba(X_test)[:, 1]
    predictions = pd.DataFrame({"customer_id": X_test.index, "observed_churn": y_test.to_numpy(),
                                "probability": predicted, "contact": predicted >= threshold,
                                "monthly_charge": X_test.MonthlyCharges.to_numpy()})
    if hashlib.sha256(Path(DATA).read_bytes()).hexdigest() != source_sha:
        raise ValueError("source changed during evaluation; refusing publication")
    report = {"schema_version": 1, "scope": "real full IBM sample; bounded logistic candidates only",
              "rows": len(data), "train_rows": len(X_train), "test_rows": len(X_test), "seed": SEED,
              "selected_model": selected, "selection_auc": selection, "threshold": threshold,
              "test_roc_auc": float(roc_auc_score(y_test, predicted)), "test_brier": float(brier_score_loss(y_test, predicted)),
              "training_oof_policy_fit_value": policy_fit_value,
              "test_illustrative_net_value": net_value(y_test, predicted, X_test.MonthlyCharges, threshold),
              "test_illustrative_contact_everyone": net_value(y_test, predicted, X_test.MonthlyCharges, 0.),
              "test_contact_nobody": 0., "contacted_test_customers": int((predicted >= threshold).sum()),
              "assumptions": {"offer_cost": 60., "offer_success": .25, "value_months": 12},
              "limits": ["Previously inspected fixed random split; not a new blind assessment or temporal validation.",
                         "Cross-fit model selection and calibration exclude each scored customer's labels; threshold maximization still fits the policy on OOF outcomes.",
                         "OOF policy-fit value is not an unbiased estimate of policy value; only final held-out outcomes assess the frozen policy.",
                         "Illustrative utility is not measured savings or retention treatment effect. Historical RF results were not regenerated."],
              "source_sha256": source_sha, "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                                                       "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
              "code_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                              for name in ("nested_demo.py", "nested.py", "compare.py", "train.py")}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".nested-evaluation-", dir=args.out.parent))
    try:
        (stage / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        (stage / "folds.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        predictions.to_csv(stage / "heldout_predictions.csv", index=False)
        pd.DataFrame({"customer_id": X_train.index, "observed_churn": y_train.to_numpy(), "oof_probability": oof,
                      "outer_fold": audit["fold_for_row"]}).to_csv(stage / "training_oof.csv", index=False)
        manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in stage.iterdir()}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.rename(stage, args.out)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
