"""Hidden Markov Model training and *online* posterior inference.

The forward algorithm here is the single most important piece of the codebase.
At time t the posterior P(state_t | obs_{1..t}) is computed using ONLY the
observations available up to and including t — never future bars. We never
call `model.predict()` or `model.predict_proba()` (both use the full sequence,
which leaks future information into past timesteps).
"""
from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


# BIC search bounds.
DEFAULT_N_RANGE: tuple[int, int] = (3, 7)
HMM_N_ITER: int = 200
HMM_COVARIANCE_TYPE: str = "full"
HMM_RANDOM_STATE: int = 42

# Walk-forward defaults.
DEFAULT_MIN_TRAIN_SIZE: int = 250   # ≈ 1 trading year before the first OOS bar
DEFAULT_REFIT_PERIOD: int = 20      # refit roughly monthly

# Numerical floor for log-space arithmetic.
_LOG_EPSILON: float = 1e-300


# ---------------------------------------------------------------------------
# Training — information-criterion-based model selection
# ---------------------------------------------------------------------------
def _n_params(model: GaussianHMM, n_features: int) -> int:
    """Free-parameter count used by both BIC and AIC.

    Components:
      - Transition matrix: n * (n - 1) free entries (rows sum to 1)
      - Initial distribution: n - 1 free entries (sums to 1)
      - Gaussian means: n * n_features
      - Full covariances: n * n_features * (n_features + 1) / 2
    """
    n = model.n_components
    n_trans = n * (n - 1)
    n_init = n - 1
    n_means = n * n_features
    n_covars = n * n_features * (n_features + 1) // 2
    return n_trans + n_init + n_means + n_covars


def _bic(model: GaussianHMM, X: np.ndarray) -> float:
    """Standard Bayesian Information Criterion.

        BIC = -2 * log_likelihood + k * log(N)

    Lower is better. Penalty grows with log(N), so for large N the
    likelihood term dominates and BIC will pick more components only when
    they truly help fit.
    """
    ll = model.score(X)             # total log-likelihood (not per-sample)
    n_obs = len(X)
    k = _n_params(model, X.shape[1])
    return -2.0 * ll + k * np.log(n_obs)


def _aic(model: GaussianHMM, X: np.ndarray) -> float:
    """Akaike Information Criterion.

        AIC = -2 * log_likelihood + 2 * k

    Lower is better. AIC penalizes parameters less harshly than BIC, so it
    tends to pick slightly more components. Useful as a second opinion.
    """
    ll = model.score(X)
    k = _n_params(model, X.shape[1])
    return -2.0 * ll + 2.0 * k


_CRITERIA = {"bic": _bic, "aic": _aic}


def train_best_hmm(
    X: np.ndarray,
    n_range: Tuple[int, int] = DEFAULT_N_RANGE,
    criterion: str = "bic",
) -> Tuple[GaussianHMM, int]:
    """Fit GaussianHMMs across `n_range` and return the best one.

    Parameters
    ----------
    X : np.ndarray
        (T, n_features) feature matrix, ideally already standardized.
    n_range : (low, high), inclusive
        Range of n_components to search.
    criterion : {"bic", "aic"}
        Information criterion used to compare candidate models. Default BIC.

    Returns
    -------
    (best_model, best_n)
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}.")
    if len(X) < 60:
        raise ValueError(
            f"Need at least 60 rows to fit an HMM reliably; got {len(X)}."
        )

    low, high = n_range
    if low < 2 or high < low:
        raise ValueError(f"Invalid n_range {n_range}.")

    crit = criterion.lower()
    if crit not in _CRITERIA:
        raise ValueError(f"criterion must be 'bic' or 'aic'; got {criterion!r}.")
    score_fn = _CRITERIA[crit]

    best_model: GaussianHMM | None = None
    best_n: int = -1
    best_score: float = np.inf

    # hmmlearn emits "Model is not converging" warnings in degenerate cases;
    # we capture those without spamming the user.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n in range(low, high + 1):
            try:
                model = GaussianHMM(
                    n_components=n,
                    covariance_type=HMM_COVARIANCE_TYPE,
                    n_iter=HMM_N_ITER,
                    random_state=HMM_RANDOM_STATE,
                    tol=1e-3,
                )
                model.fit(X)
                value = score_fn(model, X)
            except Exception:
                # Skip degenerate fits (singular covariance, no convergence).
                continue

            if np.isfinite(value) and value < best_score:
                best_score = value
                best_model = model
                best_n = n

    if best_model is None:
        raise RuntimeError(
            "No HMM in the requested range fit successfully. "
            "Try widening the date range or simplifying features."
        )

    return best_model, best_n


# ---------------------------------------------------------------------------
# Forward filtering — the anti-bias core
# ---------------------------------------------------------------------------
def _log_gaussian_pdf(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Per-timestep log N(x_t; mean, cov) for a full-covariance Gaussian.

    Vectorized across all rows of X. Returns shape (T,).

    Robustness: EM on small windows sometimes produces nearly-singular
    covariance matrices that fail Cholesky. We retry with increasing diagonal
    jitter — a tiny regularization that is invisible in well-conditioned
    cases but rescues us from numerical singularities.
    """
    d = X.shape[1]
    diff = X - mean  # (T, d)

    L = None
    eye = np.eye(d, dtype=cov.dtype)
    # Start with no jitter; if Cholesky fails, escalate.
    for jitter in (0.0, 1e-10, 1e-8, 1e-6, 1e-4):
        try:
            L = np.linalg.cholesky(cov + jitter * eye)
            break
        except np.linalg.LinAlgError:
            continue
    if L is None:
        # Last-ditch: shrink toward the identity to guarantee psd.
        L = np.linalg.cholesky(0.5 * cov + 0.5 * eye * float(np.trace(cov)) / d)

    y = np.linalg.solve(L, diff.T)            # (d, T)
    mahalanobis_sq = np.sum(y * y, axis=0)    # (T,)
    log_det_cov = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d * np.log(2.0 * np.pi) + log_det_cov + mahalanobis_sq)


def _emission_log_prob(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """(T, n_components) matrix of log emission probabilities."""
    T = len(X)
    n = model.n_components
    out = np.empty((T, n), dtype=np.float64)
    covars = model.covars_
    for i in range(n):
        out[:, i] = _log_gaussian_pdf(X, model.means_[i], covars[i])
    return out


def _logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Numerically stable log-sum-exp."""
    a_max = np.max(a, axis=axis, keepdims=True)
    a_max = np.where(np.isfinite(a_max), a_max, 0.0)
    out = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)) + a_max
    return np.squeeze(out, axis=axis) if axis is not None else out.squeeze()


def forward_filter(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """Causal forward-algorithm posteriors.

    Parameters
    ----------
    model : GaussianHMM
        A *fitted* model — uses model.startprob_, model.transmat_,
        model.means_, model.covars_.
    X : np.ndarray
        (T, n_features) observation matrix.

    Returns
    -------
    np.ndarray
        (T, n_components) posterior probabilities. Row t depends only on
        X[0:t+1], never on X[t+1:].
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}.")
    T = len(X)
    n = model.n_components

    log_emit = _emission_log_prob(model, X)                                  # (T, n)
    log_start = np.log(np.clip(model.startprob_, _LOG_EPSILON, None))        # (n,)
    log_trans = np.log(np.clip(model.transmat_, _LOG_EPSILON, None))         # (n, n)

    log_alpha = np.empty((T, n), dtype=np.float64)

    # t = 0 — initial state distribution.
    log_alpha[0] = log_start + log_emit[0]

    # t = 1..T-1 — recursion.
    # log_alpha[t, j] = log_emit[t, j] + logsumexp_i(log_alpha[t-1, i] + log_trans[i, j])
    for t in range(1, T):
        # broadcast: prev (n, 1) + trans (n, n) → (n, n), reduce axis=0 → (n,)
        prev = log_alpha[t - 1][:, None] + log_trans
        log_alpha[t] = log_emit[t] + _logsumexp(prev, axis=0)

    # Normalize each row to a probability distribution (posterior, not joint).
    log_norm = _logsumexp(log_alpha, axis=1)[:, None]
    posterior = np.exp(log_alpha - log_norm)

    # Clip tiny negatives from floating error and renormalize.
    posterior = np.clip(posterior, 0.0, 1.0)
    row_sums = posterior.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    posterior = posterior / row_sums
    return posterior


# ---------------------------------------------------------------------------
# Look-ahead bias verification
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Walk-forward (out-of-sample) classification
# ---------------------------------------------------------------------------
def walk_forward_classify(
    X: np.ndarray,
    *,
    n_components: int,
    min_train_size: int = DEFAULT_MIN_TRAIN_SIZE,
    refit_period: int = DEFAULT_REFIT_PERIOD,
    standardize: bool = True,
    progress_callback=None,
) -> tuple[np.ndarray, list[int], list[GaussianHMM]]:
    """Rolling-window HMM refit with strict no-look-ahead.

    Algorithm
    ---------
    For bars t in [min_train_size, T):
        - Let r = largest refit point <= t (refit at min_train_size, then
          every `refit_period` bars).
        - Fit a StandardScaler and a fresh GaussianHMM on X[:r] (data
          strictly before bar t).
        - Use forward_filter on scaler(X[:t+1]) to compute posterior for
          bar t, then KEEP only the posterior at row t (the most recent bar).
    The forward filter is already causal, so this gives a fully OOS
    classification: bar t's regime depends only on observations up to t and
    on parameters (scaler mean/variance included) estimated from
    observations strictly before the refit point that t falls into.

    Parameters
    ----------
    X : (T, n_features) — RAW (unscaled) features. Do not pass a matrix
        standardized on the full sample: its mean/variance already contain
        future bars and the result is no longer out-of-sample.
    n_components : fixed n for all refits (caller decides via BIC/AIC on
                   the initial training window).
    min_train_size : number of bars needed before the first OOS prediction.
    refit_period : how many bars between successive HMM refits. Smaller =
                   more responsive but slower.
    standardize : fit a StandardScaler on X[:r] at every refit point
                  (default). False only if X is already scaled causally.

    Returns
    -------
    posteriors : (T - min_train_size, n_components) np.ndarray
        Posterior probabilities for bars [min_train_size:].
    state_orderings : list of length len(refit_points)
        For each refit, argsort of model.means_[:, 0] — used by the caller
        to map raw state ids to volatility-ordered labels per refit window.
    models : list of fitted GaussianHMM, one per refit point.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}.")
    T = len(X)
    if T < min_train_size + refit_period:
        raise ValueError(
            f"Need at least min_train_size + refit_period = "
            f"{min_train_size + refit_period} bars; got {T}."
        )
    if refit_period < 1:
        raise ValueError("refit_period must be ≥ 1")

    # Refit points: the bar index at which a new model becomes "current".
    # First refit uses bars [:min_train_size], then every refit_period bars.
    refit_points: list[int] = []
    r = min_train_size
    while r < T:
        refit_points.append(r)
        r += refit_period

    n_oos = T - min_train_size
    posteriors = np.full((n_oos, n_components), np.nan, dtype=np.float64)
    state_orderings: list[np.ndarray] = []
    models: list[GaussianHMM] = []
    scaler: StandardScaler | None = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for i, r in enumerate(refit_points):
            # Train scaler + HMM on the strictly-prior window [:r].
            try:
                new_scaler = StandardScaler().fit(X[:r]) if standardize else None
                model = GaussianHMM(
                    n_components=n_components,
                    covariance_type=HMM_COVARIANCE_TYPE,
                    n_iter=HMM_N_ITER,
                    random_state=HMM_RANDOM_STATE,
                    tol=1e-3,
                )
                model.fit(X[:r] if new_scaler is None else new_scaler.transform(X[:r]))
                scaler = new_scaler
            except Exception:
                # Fall back to the previous model (and its scaler) if this fit failed.
                if not models:
                    raise RuntimeError(
                        f"Initial walk-forward fit at r={r} failed; "
                        "widen the date range or relax n_components."
                    )
                model = models[-1]

            models.append(model)
            state_orderings.append(np.argsort(model.means_[:, 0]))

            # Window of OOS bars this model is responsible for.
            next_r = refit_points[i + 1] if i + 1 < len(refit_points) else T
            # Run the causal forward filter over [:next_r] and keep rows [r:next_r].
            X_block = X[:next_r] if scaler is None else scaler.transform(X[:next_r])
            post_block = forward_filter(model, X_block)  # shape (next_r, n)
            # Map model's raw state order to canonical (vol-sorted) order so
            # column j of `posteriors` always means "j-th vol regime".
            order = state_orderings[-1]
            post_block_sorted = post_block[:, order]
            posteriors[r - min_train_size : next_r - min_train_size] = (
                post_block_sorted[r:next_r]
            )

            if progress_callback is not None:
                progress_callback((i + 1) / len(refit_points))

    return posteriors, state_orderings, models


def verify_no_lookahead(model: GaussianHMM, X: np.ndarray) -> bool:
    """Confirm forward_filter is causal.

    Runs forward_filter on X[:50] and X[:100], and checks that posteriors
    for t<50 are identical across both runs. A non-causal implementation
    would produce different posteriors because the longer sequence reveals
    future observations.

    Returns
    -------
    True if causal. Raises AssertionError otherwise.
    """
    if len(X) < 100:
        raise ValueError(
            f"verify_no_lookahead needs >=100 rows; got {len(X)}."
        )

    post_short = forward_filter(model, X[:50])
    post_long = forward_filter(model, X[:100])

    if not np.allclose(post_short, post_long[:50], atol=1e-10, rtol=1e-8):
        # Find the first row that diverges for a useful error message.
        diff = np.max(np.abs(post_short - post_long[:50]), axis=1)
        first_bad = int(np.argmax(diff > 1e-8))
        raise AssertionError(
            "Look-ahead bias detected: forward_filter posteriors differ "
            f"between the 50-row and 100-row runs starting at t={first_bad} "
            f"(max abs diff = {diff.max():.2e}). The forward algorithm "
            "implementation is NOT causal."
        )
    return True
