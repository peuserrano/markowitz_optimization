from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

from .portfolio import EfficientFrontierResult


def plot_efficient_frontier(
    result: EfficientFrontierResult,
    title: str = "Portfólio Ótimo (Markowitz)",
    save_path: str | None = None,
) -> plt.Figure:
    """Plota a fronteira eficiente e destaca o portfólio de Sharpe máximo.

    Se save_path for fornecido, salva o gráfico como PNG em vez de exibir.
    Retorna o objeto Figure para uso em notebooks ou testes.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        result.volatilities,
        result.returns_arithmetic,
        c=result.sharpe_ratios,
        cmap="viridis",
        alpha=0.4,
        s=5,
    )
    fig.colorbar(scatter, ax=ax, label="Índice de Sharpe")

    opt_vol = result.volatilities[result.optimal_index]
    opt_ret = result.returns_arithmetic[result.optimal_index]
    ax.scatter(opt_vol, opt_ret, c="red", s=80, zorder=5, label="Sharpe máximo")

    ax.plot(result.frontier_x, result.frontier_y, "b-", linewidth=2, label="Fronteira eficiente")

    ax.set_xlabel("Volatilidade esperada (a.a.)", labelpad=12)
    ax.set_ylabel("Retorno esperado (a.a.)", labelpad=12)
    ax.set_title(title)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend()
    fig.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filepath = os.path.join(save_path, "efficient_frontier.png")
        fig.savefig(filepath, dpi=150)
        print(f"Gráfico salvo em: {filepath}")
    else:
        plt.show()

    return fig
