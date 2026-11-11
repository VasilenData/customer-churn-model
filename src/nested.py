"""Cross-fit the complete model-selection and calibration procedure, preserving row identities."""
from __future__ import annotations

import hashlib
import json

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score


def _identity(index):
    return hashlib.sha256(json.dumps([str(value) for value in index], separators=(",", ":")).encode()).hexdigest()


def nested_policy_scores(candidates, X, y, *, outer_splits=5, inner_splits=3, seed=42):
    """Each held-out probability comes from selection/calibration that never saw its label.

    Candidate estimators may themselves contain GridSearchCV; those searches run
    only inside the fitting partition passed to them. Returned fold membership
    is inspectable, rather than assuming index order happened to align.
    """
    if not candidates or not X.index.is_unique or not y.index.equals(X.index):
        raise ValueError("nonempty candidates and unique aligned feature/label identities are required")
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("both binary classes are required")
    for count in (outer_splits, inner_splits):
        if isinstance(count, bool) or not isinstance(count, int) or count < 2:
            raise ValueError("each cross-validation level requires at least two folds")
    if y.value_counts().min() < outer_splits:
        raise ValueError("each class needs at least outer_splits observations")
    probabilities = np.full(len(y), np.nan)
    memberships = np.full(len(y), -1, dtype=int)
    audit = []
    for fold, (train, heldout) in enumerate(StratifiedKFold(outer_splits, shuffle=True, random_state=seed).split(X, y)):
        X_fit, y_fit = X.iloc[train], y.iloc[train]
        if y_fit.value_counts().min() < inner_splits:
            raise ValueError("an outer training fold has too few examples for inner selection/calibration")
        inner = StratifiedKFold(inner_splits, shuffle=True, random_state=seed + fold + 1)
        selection = {name: float(np.mean(cross_val_score(clone(estimator), X_fit, y_fit, cv=inner,
                                                        scoring="roc_auc", error_score="raise", n_jobs=1)))
                     for name, estimator in candidates.items()}
        if not all(np.isfinite(value) for value in selection.values()):
            raise ValueError("inner model selection produced invalid scores")
        winner = max(sorted(selection), key=selection.get)
        calibrated = CalibratedClassifierCV(clone(candidates[winner]), method="isotonic", cv=inner).fit(X_fit, y_fit)
        scored = calibrated.predict_proba(X.iloc[heldout])[:, 1]
        if len(scored) != len(heldout) or not np.isfinite(scored).all() or ((scored < 0) | (scored > 1)).any():
            raise ValueError("calibration produced invalid held-out probabilities")
        probabilities[heldout], memberships[heldout] = scored, fold
        audit.append({"fold": fold, "selected_model": winner, "selection_auc": selection,
                      "train_row_ids": [str(v) for v in X.index[train]], "heldout_row_ids": [str(v) for v in X.index[heldout]],
                      "train_ids_sha256": _identity(X.index[train]), "heldout_ids_sha256": _identity(X.index[heldout])})
    if (memberships < 0).any() or not np.isfinite(probabilities).all():
        raise ValueError("every training observation must receive exactly one held-out score")
    return probabilities, {"method": "nested selection plus within-fold isotonic calibration",
                           "outer_splits": outer_splits, "inner_splits": inner_splits, "seed": seed,
                           "row_ids": [str(v) for v in X.index], "fold_for_row": memberships.tolist(), "folds": audit}
