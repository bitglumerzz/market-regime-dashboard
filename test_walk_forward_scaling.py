"""Walk-forward (OOS) posteriors must not depend on future bars.

Perturbing bars at index >= k must leave every walk-forward posterior for
bars < k untouched — including the feature scaling, which is refitted on
X[:r] at each refit point instead of on the full sample.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hmm_model import walk_forward_classify  # noqa: E402

MIN_TRAIN = 200
REFIT = 20
K = 250  # mid-block: refit points are 200, 220, 240, 260, …


def _regime_features(T: int = 400, seed: int = 0) -> np.ndarray:
    """Three-feature series switching between three volatility regimes."""
    rng = np.random.default_rng(seed)
    means = np.array([[0.5, 0.0, 1.0], [1.5, 0.3, 2.0], [3.0, -0.5, 4.0]])
    rows, state = [], 0
    for _ in range(T):
        if rng.random() < 0.05:
            state = int(rng.integers(3))
        rows.append(means[state] + rng.normal(scale=0.3, size=3))
    return np.asarray(rows)


def _perturb_after(X: np.ndarray, k: int) -> np.ndarray:
    """Large shift + scale of every bar from k on — moves full-sample stats a lot."""
    Y = X.copy()
    Y[k:] = Y[k:] * 5.0 + 10.0
    return Y


def _wf(X: np.ndarray, **kw) -> np.ndarray:
    post, _, _ = walk_forward_classify(
        X, n_components=3, min_train_size=MIN_TRAIN, refit_period=REFIT, **kw
    )
    return post


def test_future_perturbation_does_not_change_past_posteriors():
    X = _regime_features()
    before = _wf(X)[: K - MIN_TRAIN]
    after = _wf(_perturb_after(X, K))[: K - MIN_TRAIN]
    assert np.isfinite(before).all()
    np.testing.assert_allclose(before, after, atol=1e-10, rtol=0)


def test_full_sample_scaling_leaks_future_into_past():
    """Control: the old path (scaler fitted on all bars) fails the same check,
    so the test above actually detects this kind of leak."""
    X = _regime_features()
    Y = _perturb_after(X, K)
    before = _wf(StandardScaler().fit_transform(X), standardize=False)[: K - MIN_TRAIN]
    after = _wf(StandardScaler().fit_transform(Y), standardize=False)[: K - MIN_TRAIN]
    assert not np.allclose(before, after, atol=1e-6)
