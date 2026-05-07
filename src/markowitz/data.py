from __future__ import annotations

import pandas as pd
import yfinance as yf


def validate_tickers(tickers: list[str]) -> None:
    """Lança ValueError se a lista de tickers for inválida."""
    if not tickers:
        raise ValueError("A lista de tickers não pode ser vazia.")
    if any(not isinstance(t, str) or not t.strip() for t in tickers):
        raise ValueError("Todos os tickers devem ser strings não-vazias.")


def validate_dates(start: str, end: str) -> None:
    """Lança ValueError se as datas forem inválidas ou em ordem errada."""
    try:
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
    except Exception as exc:
        raise ValueError(f"Formato de data inválido: {exc}") from exc
    if start_dt >= end_dt:
        raise ValueError(f"start_date ({start}) deve ser anterior a end_date ({end}).")


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Baixa preços de fechamento ajustados e retorna um DataFrame sem NaN.

    Tickers brasileiros da B3 devem incluir o sufixo '.SA' (ex: 'PETR4.SA').
    """
    validate_tickers(tickers)
    validate_dates(start, end)

    try:
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    except Exception as exc:
        raise ConnectionError(f"Falha ao baixar dados do Yahoo Finance: {exc}") from exc

    prices: pd.DataFrame = raw["Close"]

    # yfinance retorna Series quando há um único ticker
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    prices = prices.ffill().bfill()

    if prices.empty or prices.isna().all().any():
        raise ValueError(
            f"Dados insuficientes para os tickers {tickers} no período {start}–{end}. "
            "Verifique os símbolos e o intervalo de datas."
        )

    return prices
