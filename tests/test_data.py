import pytest

from src.markowitz.data import validate_tickers, validate_dates


def test_validate_tickers_vazio() -> None:
    with pytest.raises(ValueError, match="vazia"):
        validate_tickers([])


def test_validate_tickers_string_vazia() -> None:
    with pytest.raises(ValueError, match="strings não-vazias"):
        validate_tickers(["AAPL", ""])


def test_validate_dates_ordem_errada() -> None:
    with pytest.raises(ValueError, match="anterior"):
        validate_dates("2022-01-01", "2015-01-01")


def test_validate_dates_iguais() -> None:
    with pytest.raises(ValueError, match="anterior"):
        validate_dates("2020-01-01", "2020-01-01")


def test_validate_dates_formato_invalido() -> None:
    with pytest.raises(ValueError):
        validate_dates("not-a-date", "2022-01-01")


def test_validate_datas_validas() -> None:
    validate_dates("2015-01-01", "2022-12-31")


def test_validate_tickers_validos() -> None:
    validate_tickers(["AAPL", "PETR4.SA", "GOOGL"])
