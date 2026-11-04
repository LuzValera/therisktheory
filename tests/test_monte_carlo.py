"""Unit tests for Monte Carlo collective aggregate loss simulation."""

from __future__ import annotations

import numpy as np
import pytest

from therisktheory.severity_aggregate import simulate_aggregate_loss


def test_simulate_aggregate_loss_poisson_exponential() -> None:
    # E[N] = 10, E[X] = 100 -> E[S] = 1000
    # Var(N) = 10, Var(X) = 10000 -> Var(S) = 10*10000 + 10*10000 = 200000
    sims = simulate_aggregate_loss(
        frequency_dist="poisson",
        severity_dist="exponential",
        num_simulations=20_000,
        frequency_params={"lam": 10.0},
        severity_params={"scale": 100.0},
        random_state=42,
    )
    assert sims.shape == (20_000,)
    assert np.all(sims >= 0.0)
    assert np.mean(sims) == pytest.approx(1000.0, rel=0.05)


def test_simulate_aggregate_loss_negative_binomial_lognormal() -> None:
    sims = simulate_aggregate_loss(
        frequency_dist="negative_binomial",
        severity_dist="lognormal",
        num_simulations=10_000,
        frequency_params={"n": 5.0, "p": 0.5},
        severity_params={"mean": 4.0, "sigma": 0.5},
        random_state=123,
    )
    assert sims.shape == (10_000,)
    assert np.all(sims >= 0.0)


def test_simulate_aggregate_loss_geometric_gamma() -> None:
    sims = simulate_aggregate_loss(
        frequency_dist="geometric",
        severity_dist="gamma",
        num_simulations=5_000,
        frequency_params={"p": 0.25},
        severity_params={"shape": 2.0, "scale": 50.0},
        random_state=99,
    )
    assert sims.shape == (5_000,)
    assert np.all(sims >= 0.0)


def test_simulate_aggregate_loss_pareto_and_weibull() -> None:
    sims_pareto = simulate_aggregate_loss(
        frequency_dist="poisson",
        severity_dist="pareto",
        num_simulations=5_000,
        frequency_params={"lam": 2.0},
        severity_params={"a": 3.0, "scale": 50.0},
        random_state=42,
    )
    assert sims_pareto.shape == (5_000,)

    sims_weibull = simulate_aggregate_loss(
        frequency_dist="poisson",
        severity_dist="weibull",
        num_simulations=5_000,
        frequency_params={"lam": 2.0},
        severity_params={"a": 1.5, "scale": 100.0},
        random_state=42,
    )
    assert sims_weibull.shape == (5_000,)


def test_simulate_aggregate_loss_custom_callables() -> None:
    def custom_freq(size: int, rng: np.random.Generator) -> np.ndarray:
        return rng.binomial(n=10, p=0.3, size=size)

    def custom_sev(size: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(low=50.0, high=150.0, size=size)

    sims = simulate_aggregate_loss(
        frequency_dist=custom_freq,
        severity_dist=custom_sev,
        num_simulations=5_000,
        random_state=7,
    )
    assert sims.shape == (5_000,)
    # E[N] = 3.0, E[X] = 100.0 -> E[S] = 300.0
    assert np.mean(sims) == pytest.approx(300.0, rel=0.08)


def test_simulate_aggregate_loss_reproducibility() -> None:
    sims_1 = simulate_aggregate_loss(
        frequency_dist="poisson",
        severity_dist="exponential",
        num_simulations=1000,
        frequency_params={"lam": 5.0},
        severity_params={"scale": 20.0},
        random_state=12345,
    )
    sims_2 = simulate_aggregate_loss(
        frequency_dist="poisson",
        severity_dist="exponential",
        num_simulations=1000,
        frequency_params={"lam": 5.0},
        severity_params={"scale": 20.0},
        random_state=12345,
    )
    np.testing.assert_array_equal(sims_1, sims_2)


def test_simulate_aggregate_loss_zero_claims() -> None:
    # Custom frequency returning zero claims
    sims = simulate_aggregate_loss(
        frequency_dist=lambda size, rng: np.zeros(size, dtype=np.int64),
        severity_dist="exponential",
        num_simulations=500,
    )
    assert np.all(sims == 0.0)


def test_simulate_aggregate_loss_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="num_simulations"):
        simulate_aggregate_loss("poisson", "exponential", num_simulations=0)

    with pytest.raises(ValueError, match="Unsupported frequency"):
        simulate_aggregate_loss("unknown_freq", "exponential", num_simulations=10)

    with pytest.raises(ValueError, match="Unsupported severity"):
        simulate_aggregate_loss("poisson", "unknown_sev", num_simulations=10)

    with pytest.raises(ValueError, match="lam"):
        simulate_aggregate_loss("poisson", "exponential", frequency_params={"lam": -1.0})

    with pytest.raises(ValueError, match="Negative Binomial"):
        simulate_aggregate_loss(
            "negative_binomial", "exponential", frequency_params={"n": -1.0, "p": 0.5}
        )

    with pytest.raises(ValueError, match="Geometric"):
        simulate_aggregate_loss("geometric", "exponential", frequency_params={"p": -0.5})

    with pytest.raises(ValueError, match="scale"):
        simulate_aggregate_loss("poisson", "exponential", severity_params={"scale": -10.0})

    with pytest.raises(ValueError, match="sigma"):
        simulate_aggregate_loss(
            "poisson", "lognormal", severity_params={"mean": 0.0, "sigma": -1.0}
        )

    with pytest.raises(ValueError, match="Pareto"):
        simulate_aggregate_loss("poisson", "pareto", severity_params={"a": -2.0, "scale": 1.0})

    with pytest.raises(ValueError, match="Gamma"):
        simulate_aggregate_loss("poisson", "gamma", severity_params={"shape": -1.0, "scale": 1.0})

    with pytest.raises(ValueError, match="Weibull"):
        simulate_aggregate_loss("poisson", "weibull", severity_params={"a": -1.0, "scale": 1.0})

    with pytest.raises(ValueError, match="negative claim counts"):
        simulate_aggregate_loss(
            frequency_dist=lambda size, rng: np.array([-1, 2]),
            severity_dist="exponential",
            num_simulations=2,
        )

    with pytest.raises(ValueError, match="negative claim values"):
        simulate_aggregate_loss(
            frequency_dist=lambda size, rng: np.array([2]),
            severity_dist=lambda size, rng: np.array([-5.0, 10.0]),
            num_simulations=1,
        )
