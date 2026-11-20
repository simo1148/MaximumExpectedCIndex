"""Core computations for the EC index.
Throughout we will use T as a vector of times,
D as a vector of events (true or false, 0 or 1)
S will be survival function S(t) = P(T>t)
G will be the censoring distribution.
"""

from __future__ import annotations

import numpy as np
from lifelines import KaplanMeierFitter


def fit_censoring_distribution(T, C):
    """Fit the censoring distribution."""
    kmf = KaplanMeierFitter().fit(T, ~C)  # fit KM to censored events
    return kmf


def fit_survival_function(T, C):
    "Fit the baseline survival function if needed"
    return KaplanMeierFitter().fit(T, C)  # fit KM for survival fn S


def calculate_hazard(t, S, max_h=9e5, min_h=0, t_star=None, use_median=False):
    """Calculates the hazard for time t and survival function S."""
    if use_median:
        return np.log(0.5) / np.log(S.survival_function_at_times(t).iloc[0])

    # first drops and the boundary would be missed.
    values, widths = _survival_steps(S, t_star)
    lower = float(np.sum(widths[values >= 1.0]))
    upper = float(np.sum(widths))
    if t <= lower:
        return max_h
    if t >= upper:
        return min_h
    return find_hazard_from_mean(t, S, t_star=t_star, max_h=max_h)


def _survival_steps(
    S: KaplanMeierFitter, t_star: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """calculates step sizes and values for expectation based"""
    timeline = S.timeline
    values = S.survival_function_.values.ravel()

    if t_star is None:
        if values[-1] > 0.0:
            raise ValueError("S does not reach 0. Pass `t_star` to integrate ")
        # Past the last event S is exactly 0 and contributes nothing.
        t_star = float(timeline[-1])

    edges = np.concatenate([timeline, [t_star]])
    widths = np.diff(edges)
    return values, widths


def find_hazard_from_mean(
    t: float,
    S: KaplanMeierFitter,
    t_star: float | None = None,
    tol: float = 1e-8,
    max_h: float = 9e5,
    max_iter: int = 200,
) -> float:

    """optional to remove first and last event. """
    
    values, widths = _survival_steps(S, t_star)

    def integral(h: float) -> float:
        return float(np.sum(values**h * widths))

    upper = integral(0.0)  # == t_star: the h -> 0 limit
    lower = float(np.sum(widths[values >= 1.0]))  # the h -> infinity limit

    if not lower < t < upper:
        raise ValueError(
            f"target mean {t:.6g} is outside the achievable range "
            f"({lower:.6g}, {upper:.6g}). The upper end is t_star; the lower "
            "end is the time to the first event, where S is still 1."
        )

    # Bracket by doubling. The root is in (lo, hi) once integral(hi) < t.
    lo, hi = 0.0, 1.0
    while integral(hi) > t:
        lo = hi
        hi *= 2.0
        if hi > max_h:
            raise ValueError(
                f"could not bracket a hazard below max_h={max_h:g} for target "
                f"mean {t:.6g}; the target is very close to the first event time"
            )

    for _ in range(max_iter):
        if hi - lo <= tol:
            break
        mid = 0.5 * (lo + hi)
        # integral is decreasing in h, so overshooting the mean means h is
        # still too small.
        if integral(mid) > t:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


def ecindex(T, C, S=None, t_star=None, use_median=False):
    assert isinstance(C, np.ndarray) and C.dtype == bool, (
        "Censorin flags are not boolean"
    )
    assert isinstance(T, np.ndarray), "T is not array"
    assert T.shape[0] == C.shape[0], (
        "Events and censoring arrays have different lengths"
    )

    n_prevalent = int(np.sum(C & (T <= 0)))
    if n_prevalent:
        raise ValueError(
            f"{n_prevalent} subject(s) have an event at time <= 0 (prevalent "
            "cases: the outcome was already present at baseline). They have no "
            "incident time to contribute, so no hazard is defined for them. "
            "Exclude them before calling, e.g. `keep = T > 0`."
        )

    cens = fit_censoring_distribution(T, C)
    if S is None:
        S = fit_survival_function(T, C)
    sort_mask = np.argsort(T)
    events = T[sort_mask][C[sort_mask]]  # get sorted not censored events
    vcalculate_hazard = np.vectorize(
        lambda t: calculate_hazard(t, S, t_star=t_star, use_median=use_median)
    )
    hazards = vcalculate_hazard(events)
    weights = 1 / cens.survival_function_at_times(events).values
    numerator, denominator = 0, 0
    for i, h in enumerate(hazards[:-1]):
        denom_h = h + hazards[i + 1 :]
        # Both hazards 0 (both events at/after t_star) score = 0.5 like a 
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(denom_h == 0, 0.5, h / denom_h)
        numerator += np.sum(weights[i] * weights[i + 1 :] * ratio)
        denominator += np.sum(weights[i] * weights[i + 1 :])
    return numerator / denominator, events, hazards, weights



