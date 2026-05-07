from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class EfficientFrontierResult:
    """Resultado completo da otimização de portfólio."""

    tickers: list[str]
    volatilities: np.ndarray
    returns_arithmetic: np.ndarray
    sharpe_ratios: np.ndarray
    frontier_x: list[float]
    frontier_y: np.ndarray
    all_weights: np.ndarray
    optimal_index: int = field(repr=False)

    @property
    def optimal_weights(self) -> np.ndarray:
        return self.all_weights[self.optimal_index]

    @property
    def optimal_weights_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"Ação": self.tickers, "Peso no Portfólio": self.optimal_weights}
        )


class PortfolioOptimization:
    """Otimização de portfólio pelo método de Markowitz."""

    ANNUALIZATION = 252

    def __init__(
        self,
        prices: pd.DataFrame,
        risk_free_rate: float = 0.0,
        random_seed: int | None = None,
        annualization_factor: int = 252,
    ) -> None:
        self.prices = prices
        self.tickers = list(prices.columns)
        self.risk_free_rate = risk_free_rate
        self.random_seed = random_seed
        self.annualization_factor = annualization_factor

    def calcular_retornos(self) -> pd.DataFrame:
        """Retorna log-retornos diários, removendo linhas com NaN."""
        return self.prices.pct_change().apply(lambda x: np.log(1 + x)).dropna()

    def calcular_fronteira_eficiente(
        self, n_portfolios: int = 100000
    ) -> EfficientFrontierResult:
        """Gera a fronteira eficiente via Monte Carlo + SLSQP."""
        retornos = self.calcular_retornos()
        media_retornos = retornos.mean()
        matriz_cov = retornos.cov()
        n_ativos = len(self.tickers)
        ann = self.annualization_factor

        rng = np.random.default_rng(self.random_seed)
        pesos = rng.random((n_portfolios, n_ativos))
        pesos /= pesos.sum(axis=1, keepdims=True)

        ret_log = (pesos @ media_retornos.values) * ann
        ret_arit = np.exp(ret_log) - 1

        cov_anual = matriz_cov.values * ann
        vols = np.sqrt(np.einsum("ij,jk,ik->i", pesos, cov_anual, pesos))

        # Proteção contra divisão por zero
        sharpe = np.where(
            vols > 1e-8,
            (ret_log - self.risk_free_rate) / vols,
            0.0,
        )

        optimal_idx = int(sharpe.argmax())

        frontier_y = np.linspace(ret_arit.min(), ret_arit.max(), 50)
        frontier_x = self._calcular_curva_fronteira(
            media_retornos, matriz_cov, frontier_y, n_ativos, ann
        )

        return EfficientFrontierResult(
            tickers=self.tickers,
            volatilities=vols,
            returns_arithmetic=ret_arit,
            sharpe_ratios=sharpe,
            frontier_x=frontier_x,
            frontier_y=frontier_y,
            all_weights=pesos,
            optimal_index=optimal_idx,
        )

    def _calcular_curva_fronteira(
        self,
        media_retornos: pd.Series,
        matriz_cov: pd.DataFrame,
        frontier_y: np.ndarray,
        n_ativos: int,
        ann: int,
    ) -> list[float]:
        """Calcula a curva da fronteira eficiente via minimização de volatilidade."""
        cov_anual = matriz_cov.values * ann
        peso_inicial = [1.0 / n_ativos] * n_ativos
        limites = tuple((0.0, 1.0) for _ in range(n_ativos))

        def _vol(w: np.ndarray) -> float:
            w = np.asarray(w)
            return float(np.sqrt(w @ cov_anual @ w))

        def _ret_arit(w: np.ndarray) -> float:
            return float(np.exp(np.sum(media_retornos.values * w) * ann) - 1)

        frontier_x: list[float] = []
        for alvo in frontier_y:
            restricoes = (
                {"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0},
                {"type": "eq", "fun": lambda w, a=alvo: _ret_arit(w) - a},
            )
            res = minimize(
                _vol,
                peso_inicial,
                method="SLSQP",
                bounds=limites,
                constraints=restricoes,
            )
            frontier_x.append(float(res["fun"]))

        return frontier_x
