from __future__ import annotations

import yaml

from src.markowitz.data import download_prices
from src.markowitz.portfolio import PortfolioOptimization
from src.markowitz.visualization import plot_efficient_frontier


def main() -> None:
    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    prices = download_prices(cfg["tickers"], cfg["start_date"], cfg["end_date"])

    portfolio = PortfolioOptimization(
        prices=prices,
        risk_free_rate=cfg.get("risk_free_rate", 0.0),
        random_seed=cfg.get("random_seed"),
        annualization_factor=cfg.get("annualization_factor", 252),
    )

    result = portfolio.calcular_fronteira_eficiente(n_portfolios=cfg.get("n_portfolios", 100000))

    save_path = cfg["output_path"] if cfg.get("save_plot") else None
    title = f"Portfólio Ótimo (Markowitz) — {cfg['start_date']} a {cfg['end_date']}"
    plot_efficient_frontier(result, title=title, save_path=save_path)

    print("\nAlocação ótima (Sharpe máximo):")
    print(result.optimal_weights_df.to_string(index=False))


if __name__ == "__main__":
    main()
