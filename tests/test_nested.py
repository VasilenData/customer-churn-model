import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from compare import models
from nested import nested_policy_scores


class RecordingClassifier(ClassifierMixin, BaseEstimator):
    fits = []
    predictions = []

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.seen_ = set(X.index)
        type(self).fits.append(self.seen_)
        return self

    def predict_proba(self, X):
        # Every prediction in cross-validation and calibration must exclude its
        # training identities. Final fitting is not involved in this helper.
        assert not self.seen_ & set(X.index)
        type(self).predictions.append((self.seen_, set(X.index)))
        p = 1 / (1 + np.exp(-X.signal.to_numpy()))
        return np.c_[1 - p, p]


def data():
    X = pd.DataFrame({"signal": np.linspace(-3, 3, 60)}, index=[f"customer-{i}" for i in range(60)])
    y = pd.Series([0, 1] * 30, index=X.index)
    return X, y


def test_every_score_excludes_its_own_identity_during_selection_and_calibration():
    X, y = data()
    probabilities, audit = nested_policy_scores({"recording": RecordingClassifier()}, X, y, outer_splits=3, inner_splits=2)
    assert np.isfinite(probabilities).all() and len(probabilities) == 60
    assert sorted(v for fold in audit["folds"] for v in fold["heldout_row_ids"]) == sorted(X.index)
    for fold in audit["folds"]:
        assert not set(fold["train_row_ids"]) & set(fold["heldout_row_ids"])
        assert len(fold["train_row_ids"]) + len(fold["heldout_row_ids"]) == 60


def test_outer_heldout_labels_cannot_change_their_probability_with_fixed_membership(monkeypatch):
    # Test explicit selection boundary independently of random stratification:
    # each outer fold changes when y changes, so fix folds for this metamorphism.
    import nested
    from sklearn.model_selection import KFold
    monkeypatch.setattr(nested, "StratifiedKFold", KFold)
    X, y = data()
    p, audit = nested_policy_scores({"lr": LogisticRegression()}, X, y, outer_splits=3, inner_splits=2)
    heldout = audit["folds"][0]["heldout_row_ids"]
    changed = y.copy()
    changed.loc[heldout] = 1 - changed.loc[heldout]
    other, _ = nested_policy_scores({"lr": LogisticRegression()}, X, changed, outer_splits=3, inner_splits=2)
    indices = X.index.get_indexer(heldout)
    np.testing.assert_array_equal(p[indices], other[indices])


def test_forest_search_is_local_not_loaded_from_historical_global_tuning():
    from sklearn.model_selection import GridSearchCV
    forest = models(data()[0])["Random forest (nested tuning)"]
    assert isinstance(forest, GridSearchCV)
    assert len(forest.param_grid["m__max_depth"]) > 1


@pytest.mark.parametrize("mutation", [lambda X, y: (X, y.iloc[::-1]),
                                      lambda X, y: (X.set_axis(["same"] * len(X)), y.set_axis(["same"] * len(y)))])
def test_identity_alignment_is_required(mutation):
    X, y = mutation(*data())
    with pytest.raises(ValueError, match="identities"):
        nested_policy_scores({"lr": LogisticRegression()}, X, y)
