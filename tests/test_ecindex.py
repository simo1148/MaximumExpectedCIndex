import numpy as np
import pytest

from ecindex.main import ecindex


def make_cohort(n: int = 150, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """A censored cohort with strictly positive event times."""
    rng = np.random.default_rng(seed)
    latent, cens = rng.exponential(10.0, n), rng.exponential(12.0, n)
    times = np.minimum(latent, cens)
    observed = latent <= cens
    keep = times > 0
    return times[keep], observed[keep]


def test_runs_via_the_mean_route() -> None:
    times, observed = make_cohort()
    index, events, hazards, weights = ecindex(
        times, observed, t_star=float(times.max())
    )
    assert 0.0 < index < 1.0
    assert len(events) == len(hazards) == len(weights) == int(observed.sum())


def test_runs_via_the_median_route() -> None:
    times, observed = make_cohort()
    index, *_ = ecindex(times, observed, use_median=True)
    assert 0.0 < index < 1.0


def test_the_two_routes_broadly_agree() -> None:
    """Mean- and median-matching are different estimands, but not wildly so."""
    times, observed = make_cohort()
    mean_index, *_ = ecindex(times, observed, t_star=float(times.max()))
    median_index, *_ = ecindex(times, observed, use_median=True)
    assert mean_index == pytest.approx(median_index, abs=0.15)


def test_prevalent_cases_are_rejected() -> None:
    times, observed = make_cohort()
    times[0], observed[0] = 0.0, True
    with pytest.raises(ValueError, match="prevalent"):
        ecindex(times, observed, t_star=float(times.max()))


def test_prevalent_cases_are_rejected_on_the_median_route() -> None:
    """The median route returns -inf rather than raising, so guard it too."""
    times, observed = make_cohort()
    times[0], observed[0] = 0.0, True
    with pytest.raises(ValueError, match="prevalent"):
        ecindex(times, observed, use_median=True)


def test_censored_observation_at_zero_is_allowed() -> None:
    """t=0 only breaks things for *events*; a censored 0 contributes no hazard."""
    times, observed = make_cohort()
    times[0], observed[0] = 0.0, False
    index, *_ = ecindex(times, observed, t_star=float(times.max()))
    assert 0.0 < index < 1.0
