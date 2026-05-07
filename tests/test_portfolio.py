import numpy as np
import pandas as pd
import pytest

from src.markowitz.portfolio import PortfolioOptimization


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=500)
    prices = np.cumprod(1 + rng.normal(0.0005, 0.02, (500, 3)), axis=0) * 100
    return pd.DataFrame(prices, index=dates, columns=["AAPL", "GOOGL", "AMZN"])


@pytest.fixture
def portfolio(synthetic_prices: pd.DataFrame) -> PortfolioOptimization:
    return PortfolioOptimization(synthetic_prices, risk_free_rate=0.045, random_seed=42)


def test_retornos_shape(portfolio: PortfolioOptimization, synthetic_prices: pd.DataFrame) -> None:
    retornos = portfolio.calcular_retornos()
    assert retornos.shape == (len(synthetic_prices) - 1, 3)


def test_retornos_sem_nan(portfolio: PortfolioOptimization) -> None:
    assert not portfolio.calcular_retornos().isna().any().any()


def test_sharpe_sem_nan_ou_inf(portfolio: PortfolioOptimization) -> None:
    result = portfolio.calcular_fronteira_eficiente(n_portfolios=200)
    assert not np.any(np.isnan(result.sharpe_ratios))
    assert not np.any(np.isinf(result.sharpe_ratios))


def test_pesos_somam_1(portfolio: PortfolioOptimization) -> None:
    result = portfolio.calcular_fronteira_eficiente(n_portfolios=200)
    assert abs(result.optimal_weights.sum() - 1.0) < 1e-6


def test_fronteira_eficiente_smoke(portfolio: PortfolioOptimization) -> None:
    result = portfolio.calcular_fronteira_eficiente(n_portfolios=200)
    assert len(result.frontier_x) == 50
    assert result.optimal_weights_df.shape == (3, 2)


def test_reproducibilidade(synthetic_prices: pd.DataFrame) -> None:
    p1 = PortfolioOptimization(synthetic_prices, random_seed=7)
    p2 = PortfolioOptimization(synthetic_prices, random_seed=7)
    r1 = p1.calcular_fronteira_eficiente(n_portfolios=100)
    r2 = p2.calcular_fronteira_eficiente(n_portfolios=100)
    np.testing.assert_array_equal(r1.optimal_weights, r2.optimal_weights)
