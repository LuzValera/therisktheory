"""Unit tests for free-parameter credibility, reserving models, and diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from therisktheory.credibility import (
    credibility_reserve_estimate,
    free_parameter_credibility,
    reserve_model_diagnostics,
)

# ============================================================================
# Tests for free_parameter_credibility
# ============================================================================


def test_free_parameter_credibility_kaas_example() -> None:
    """Benchmark test against Kaas et al. (2008), Example 8.2.1 & 8.2.5.

    3 groups observed for 5 years with equal exposure:
    Group means: 100.0, 110.0, 120.0; overall mean: 110.0.
    s^2 = 108.97, a = 78.206, Z = 0.78206.
    Credibility premiums: [102.18, 110.00, 117.82].
    """
    kaas_data = np.array(
        [
            [99.3, 93.7, 103.9, 92.5, 110.6],
            [112.5, 108.3, 118.0, 99.4, 111.8],
            [129.2, 140.9, 108.3, 105.0, 116.6],
        ]
    )

    res = free_parameter_credibility(kaas_data)

    # 1. Overall mean: (100.0 + 110.0 + 120.0) / 3 = 110.0
    assert res["overall_sample_mean"] == pytest.approx(110.0)
    assert res["collective_mean"] == pytest.approx(110.0)

    # 2. Individual sample means
    expected_means = np.array([100.0, 110.0, 120.0])
    np.testing.assert_allclose(res["individual_means"], expected_means)

    # 3. Process variance: s^2 = 108.97
    assert res["process_variance"] == pytest.approx(108.97)

    # 4. Between-risk variance: a = 78.206
    assert res["variance_hypothetical_means"] == pytest.approx(78.206)

    # 5. Parameter ratio: k = 108.97 / 78.206 = 1.39337...
    expected_k = 108.97 / 78.206
    assert res["k_ratio"] == pytest.approx(expected_k)

    # 6. Credibility factor: Z = 5 / (5 + k) = 0.78206...
    expected_z = 5.0 / (5.0 + expected_k)
    np.testing.assert_allclose(res["credibility_factors"], np.full(3, expected_z))

    # 7. Credibility premiums: [102.18, 110.00, 117.82]
    expected_premiums = np.array([102.1794, 110.0, 117.8206])
    np.testing.assert_allclose(res["credibility_premiums"], expected_premiums, atol=1e-4)

    # 8. Portfolio balance: sum(P_i) == sum(X_bar_i) = 330.0
    assert np.sum(res["credibility_premiums"]) == pytest.approx(np.sum(expected_means))


def test_free_parameter_credibility_weighted_buhlmann_straub() -> None:
    """Test non-parametric credibility with unequal exposure weights."""
    # 3 risks, 3 years
    losses = np.array(
        [
            [10.0, 12.0, 14.0],
            [20.0, 22.0, 24.0],
            [30.0, 32.0, 34.0],
        ]
    )
    weights = np.array(
        [
            [100.0, 150.0, 200.0],  # P_1. = 450
            [200.0, 200.0, 200.0],  # P_2. = 600
            [50.0, 50.0, 100.0],  # P_3. = 200
        ]
    )

    res = free_parameter_credibility(losses, weights=weights)

    # Credibility factors should vary with row totals P_{i.}
    z_factors = res["credibility_factors"]
    assert len(z_factors) == 3
    # Risk 2 has highest weight (600) -> highest credibility factor
    assert z_factors[1] > z_factors[0] > z_factors[2]
    assert np.all((z_factors >= 0.0) & (z_factors <= 1.0))

    # Portfolio balance condition: sum P_{i.} * P_i == sum P_{i.} * X_bar_i
    row_weights = res["total_weights"]
    total_fitted = np.sum(row_weights * res["credibility_premiums"])
    total_observed = np.sum(row_weights * res["individual_means"])
    assert total_fitted == pytest.approx(total_observed)


def test_free_parameter_credibility_zero_between_variance_truncation() -> None:
    """When risks are identical, between-risk variance estimator can be negative.

    Check that truncation a_hat = max(0, a_raw) yields a=0, k=inf, Z=0,
    and all premiums equal the collective mean.
    """
    # 4 risks with identical random noise around same mean 50.0
    data = np.array(
        [
            [50.1, 49.9, 50.0, 50.0],
            [49.9, 50.1, 50.0, 50.0],
            [50.0, 50.0, 50.1, 49.9],
            [50.0, 50.0, 49.9, 50.1],
        ]
    )
    res = free_parameter_credibility(data)

    # Sample means are identical (50.0), so raw between variance is negative
    assert res["raw_variance_hypothetical_means"] < 0.0
    assert res["variance_hypothetical_means"] == 0.0
    assert np.isposinf(res["k_ratio"])
    np.testing.assert_allclose(res["credibility_factors"], 0.0)
    # When Z = 0, all premiums collapse to collective mean
    np.testing.assert_allclose(res["credibility_premiums"], res["collective_mean"])


# ============================================================================
# Tests for credibility_reserve_estimate
# ============================================================================


def test_credibility_reserve_estimate_bornhuetter_ferguson_default() -> None:
    """Test default Bornhuetter-Ferguson credibility bridge (Z = 1/f)."""
    # C = 60,000, M = 100,000, f = 1.25 -> p = 0.80, Z = 0.80
    # CL ultimate = 60,000 * 1.25 = 75,000
    # BF ultimate = 0.80 * 75,000 + 0.20 * 100,000 = 60,000 + 20,000 = 80,000
    # Reserve = 80,000 - 60,000 = 20,000
    res = credibility_reserve_estimate(
        reported_claims=60000.0,
        expected_claims=100000.0,
        development_factor=1.25,
        credibility_factor=None,
    )

    assert np.asarray(res["emergence_percentage"]).item() == pytest.approx(0.80)
    assert np.asarray(res["credibility_factors"]).item() == pytest.approx(0.80)
    assert np.asarray(res["chain_ladder_ultimate"]).item() == pytest.approx(75000.0)
    assert np.asarray(res["ultimate_claims"]).item() == pytest.approx(80000.0)
    assert np.asarray(res["reserve"]).item() == pytest.approx(20000.0)


def test_credibility_reserve_estimate_explicit_extremes() -> None:
    # 1. Full credibility (Z = 1.0) -> Pure Chain Ladder
    res_cl = credibility_reserve_estimate(
        reported_claims=50000.0,
        expected_claims=120000.0,
        development_factor=2.0,
        credibility_factor=1.0,
    )
    assert np.asarray(res_cl["ultimate_claims"]).item() == pytest.approx(100000.0)
    assert np.asarray(res_cl["reserve"]).item() == pytest.approx(50000.0)

    # 2. Zero credibility (Z = 0.0) -> Pure prior expectation
    res_prior = credibility_reserve_estimate(
        reported_claims=50000.0,
        expected_claims=120000.0,
        development_factor=2.0,
        credibility_factor=0.0,
    )
    assert np.asarray(res_prior["ultimate_claims"]).item() == pytest.approx(120000.0)
    assert np.asarray(res_prior["reserve"]).item() == pytest.approx(70000.0)


def test_credibility_reserve_estimate_vectorized_and_premiums() -> None:
    reported = np.array([10000.0, 30000.0, 70000.0])
    expected = np.array([50000.0, 50000.0, 80000.0])
    cdf = np.array([4.0, 2.0, 1.1])
    premiums = np.array([80000.0, 80000.0, 100000.0])

    res = credibility_reserve_estimate(
        reported_claims=reported,
        expected_claims=expected,
        development_factor=cdf,
        premiums=premiums,
    )

    assert "implied_loss_ratio" in res
    np.testing.assert_allclose(res["implied_loss_ratio"], res["ultimate_claims"] / premiums)
    assert np.all(res["reserve"] >= 0.0)


# ============================================================================
# Tests for reserve_model_diagnostics
# ============================================================================


def test_reserve_model_diagnostics_metrics() -> None:
    obs = np.array([100.0, 110.0, 120.0, 130.0])
    fit = np.array([102.0, 111.0, 118.0, 129.0])
    z = np.array([0.7, 0.8, 0.6, 0.9])
    coll_mean = 115.0

    diag = reserve_model_diagnostics(
        observed_values=obs,
        credibility_estimates=fit,
        collective_mean=coll_mean,
        credibility_factors=z,
        process_variance=25.0,
        hypothetical_mean_variance=30.0,
    )

    assert "portfolio_imbalance" in diag
    assert "relative_imbalance" in diag
    assert "weighted_mae" in diag
    assert "weighted_rmse" in diag
    assert diag["mean_credibility"] == pytest.approx(0.75)
    assert diag["high_credibility_ratio"] == 1.0  # all Z >= 0.5

    # Theoretical MSE = (1 - Z) * a
    expected_mse = (1.0 - z) * 30.0
    np.testing.assert_allclose(diag["theoretical_mse"], expected_mse)

    # Standardized residuals check
    assert "standardized_residuals" in diag
    expected_std_resid = (obs - fit) / np.sqrt(25.0)
    np.testing.assert_allclose(diag["standardized_residuals"], expected_std_resid)


# ============================================================================
# Tests for validation and domain bounds
# ============================================================================


def test_free_parameter_credibility_invalid_inputs() -> None:
    # 1D array
    with pytest.raises(ValueError, match="2D"):
        free_parameter_credibility(np.array([10.0, 20.0, 30.0]))

    # Less than 2 risks
    with pytest.raises(ValueError, match="at least 2 risks"):
        free_parameter_credibility(np.array([[10.0, 20.0]]))

    # Less than 2 periods
    with pytest.raises(ValueError, match="at least 2 periods"):
        free_parameter_credibility(np.array([[10.0], [20.0]]))

    # Non-finite inputs
    with pytest.raises(ValueError, match="non-finite"):
        free_parameter_credibility(np.array([[10.0, np.nan], [20.0, 30.0]]))

    # Mismatched weights shape
    with pytest.raises(ValueError, match="shape"):
        free_parameter_credibility(
            np.array([[10.0, 20.0], [30.0, 40.0]]),
            weights=np.array([[1.0], [2.0]]),
        )

    # Negative weights
    with pytest.raises(ValueError, match="non-negative"):
        free_parameter_credibility(
            np.array([[10.0, 20.0], [30.0, 40.0]]),
            weights=np.array([[-1.0, 1.0], [1.0, 1.0]]),
        )

    # Risk with zero total weight
    with pytest.raises(ValueError, match="strictly positive total weight"):
        free_parameter_credibility(
            np.array([[10.0, 20.0], [30.0, 40.0]]),
            weights=np.array([[0.0, 0.0], [1.0, 1.0]]),
        )

    # Insufficient degrees of freedom (e.g. only 1 active period per risk)
    with pytest.raises(ValueError, match="Insufficient degrees of freedom"):
        free_parameter_credibility(
            np.array([[10.0, 20.0], [30.0, 40.0]]),
            weights=np.array([[1.0, 0.0], [0.0, 1.0]]),
        )


def test_credibility_reserve_estimate_invalid_inputs() -> None:
    # Negative reported claims
    with pytest.raises(ValueError, match="reported_claims"):
        credibility_reserve_estimate(-10.0, 100.0, 1.5)

    # Non-positive expected claims
    with pytest.raises(ValueError, match="expected_claims"):
        credibility_reserve_estimate(10.0, 0.0, 1.5)

    # Development factor < 1.0
    with pytest.raises(ValueError, match="development_factor"):
        credibility_reserve_estimate(10.0, 100.0, 0.9)

    # Credibility factor outside [0, 1]
    with pytest.raises(ValueError, match="credibility_factor"):
        credibility_reserve_estimate(10.0, 100.0, 1.5, credibility_factor=1.2)

    # Non-positive premium
    with pytest.raises(ValueError, match="premiums"):
        credibility_reserve_estimate(10.0, 100.0, 1.5, premiums=-50.0)


def test_reserve_model_diagnostics_invalid_inputs() -> None:
    # 2D array instead of 1D
    with pytest.raises(ValueError, match="1D"):
        reserve_model_diagnostics([[10.0, 20.0]], [[10.0, 20.0]], 15.0, [0.5, 0.5])

    # Mismatched lengths
    with pytest.raises(ValueError, match="identical length"):
        reserve_model_diagnostics([10.0, 20.0], [10.0], 15.0, [0.5, 0.5])

    # Factor outside [0, 1]
    with pytest.raises(ValueError, match="credibility_factors"):
        reserve_model_diagnostics([10.0], [10.0], 10.0, [-0.1])

    # Weights shape mismatch
    with pytest.raises(ValueError, match="weights must be non-negative"):
        reserve_model_diagnostics([10.0, 20.0], [10.0, 20.0], 15.0, [0.5, 0.5], weights=[1.0])

    # Negative weights
    with pytest.raises(ValueError, match="weights must be non-negative"):
        reserve_model_diagnostics([10.0, 20.0], [10.0, 20.0], 15.0, [0.5, 0.5], weights=[-1.0, 1.0])

    # Total weight zero
    with pytest.raises(ValueError, match="total weight must be strictly positive"):
        reserve_model_diagnostics([10.0, 20.0], [10.0, 20.0], 15.0, [0.5, 0.5], weights=[0.0, 0.0])

    # Negative hypothetical mean variance
    with pytest.raises(ValueError, match="hypothetical_mean_variance"):
        reserve_model_diagnostics([10.0], [10.0], 10.0, [0.5], hypothetical_mean_variance=-1.0)

    # Negative process variance
    with pytest.raises(ValueError, match="process_variance"):
        reserve_model_diagnostics([10.0], [10.0], 10.0, [0.5], process_variance=-1.0)
