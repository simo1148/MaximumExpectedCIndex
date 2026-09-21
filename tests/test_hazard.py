import numpy as np
import pytest
from lifelines import KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time as rmst

from ecindex.main import find_hazard_from_mean, integrate_survival


@pytest.fixture
def uncensored() -> KaplanMeierFitter:
    """A KM curve that reaches 0, so the unrestricted integral converges."""
    rng = np.random.default_rng(0)
    times = rng.exponential(10.0, 300)
    return KaplanMeierFitter().fit(times, np.ones(300, dtype=bool))


@pytest.fixture
def censored() -> KaplanMeierFitter:
    """A KM curve with mass left at t_max, as under real censoring."""
    rng = np.random.default_rng(1)
    latent, cens = rng.exponential(10.0, 200), rng.exponential(12.0, 200)
    return KaplanMeierFitter().fit(np.minimum(latent, cens), latent <= cens)


def test_h_equals_one_is_the_restricted_mean(uncensored: KaplanMeierFitter) -> None:
    t_star = float(uncensored.timeline[-1])
    assert integrate_survival(1.0, uncensored, t_star) == pytest.approx(
        rmst(uncensored, t=t_star)
    )


def test_integral_decreases_with_h(uncensored: KaplanMeierFitter) -> None:
    means = [integrate_survival(h, uncensored, 20.0) for h in (0.1, 0.5, 1.0, 5.0)]
    assert means == sorted(means, reverse=True)


@pytest.mark.parametrize("h_true", [0.25, 1.0, 2.0, 7.5, 50.0])
def test_round_trip_recovers_the_hazard(
    uncensored: KaplanMeierFitter, h_true: float
) -> None:
    t_star = float(uncensored.timeline[-1])
    mean = integrate_survival(h_true, uncensored, t_star)
    assert find_hazard_from_mean(mean, uncensored, t_star) == pytest.approx(
        h_true, abs=1e-6
    )


@pytest.mark.parametrize("t_star", [5.0, 15.0, 25.0])
def test_round_trip_under_censoring(censored: KaplanMeierFitter, t_star: float) -> None:
    mean = integrate_survival(2.0, censored, t_star)
    assert find_hazard_from_mean(mean, censored, t_star) == pytest.approx(2.0, abs=1e-6)


def test_infinite_integral_needs_t_star(censored: KaplanMeierFitter) -> None:
    with pytest.raises(ValueError, match="does not reach 0"):
        integrate_survival(1.0, censored)


def test_target_above_t_star_is_rejected(censored: KaplanMeierFitter) -> None:
    with pytest.raises(ValueError, match="outside the achievable range"):
        find_hazard_from_mean(30.0, censored, t_star=25.0)


def test_target_below_first_event_is_rejected(censored: KaplanMeierFitter) -> None:
    too_small = float(censored.timeline[1]) * 0.5
    with pytest.raises(ValueError, match="outside the achievable range"):
        find_hazard_from_mean(too_small, censored, t_star=25.0)
