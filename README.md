# Markowitz Portfolio Optimization

Implementação da teoria de portfólio de Markowitz (1952) em Python, com otimização via Monte Carlo e programação quadrática (SLSQP).

## Funcionalidades

- Download automático de dados históricos via `yfinance`
- Geração de fronteira eficiente por simulação Monte Carlo
- Identificação do portfólio de Índice de Sharpe máximo
- Visualização interativa ou exportação do gráfico como PNG
- Configuração completa por arquivo YAML sem alterar o código

## Pré-requisitos

- Python 3.10+

## Instalação

```bash
git clone https://github.com/seu-usuario/markowitz_optimization.git
cd markowitz_optimization
pip install -r requirements.txt
```

## Configuração

Edite o arquivo `config.yaml` para definir os parâmetros da análise:

```yaml
tickers:
  - "AAPL"
  - "NKE"
  - "GOOGL"
  - "AMZN"

start_date: "2015-01-01"
end_date:   "2022-12-31"

n_portfolios: 100000      # carteiras simuladas pelo Monte Carlo
risk_free_rate: 0.045     # taxa livre de risco anualizada (4,5% a.a.)
random_seed: 42           # null para resultado aleatório a cada execução
annualization_factor: 252 # dias úteis por ano

save_plot: false          # true = salva PNG em vez de exibir
output_path: "results/"
```

> **Ações da B3:** use o sufixo `.SA` no ticker (ex: `"PETR4.SA"`, `"VALE3.SA"`).

## Execução

```bash
python main.py
```

O programa exibe o gráfico da fronteira eficiente e imprime no terminal a alocação ótima:

```
Alocação ótima (Sharpe máximo):
   Ação  Peso no Portfólio
   AAPL             0.3214
    NKE             0.1587
  GOOGL             0.2891
   AMZN             0.2308
```

## Teoria

### Fronteira Eficiente
A fronteira eficiente representa o conjunto de portfólios que oferecem o **maior retorno esperado para cada nível de risco** (volatilidade). Portfólios fora dessa fronteira são sub-ótimos.

### Índice de Sharpe
Mede o retorno ajustado pelo risco:

```
Sharpe = (E[R] - Rf) / σ
```

Onde `E[R]` é o retorno anualizado esperado, `Rf` é a taxa livre de risco e `σ` é a volatilidade anualizada.

### Log-retornos
O programa utiliza log-retornos diários por serem aditivos no tempo e produzirem estimativas de risco mais estáveis:

```
r_t = ln(P_t / P_{t-1})
```

O fator de anualização padrão é **252 dias úteis**.

## Estrutura do Projeto

```
markowitz_optimization/
├── src/
│   └── markowitz/
│       ├── data.py          # download e validação de dados
│       ├── portfolio.py     # cálculos de retorno, risco e fronteira
│       └── visualization.py # geração do gráfico
├── tests/
│   ├── test_portfolio.py
│   └── test_data.py
├── main.py       # entry point
├── config.yaml   # parâmetros de configuração
└── requirements.txt
```

## Testes

```bash
python -m pytest tests/ -v
```

## Limitações

- **Backtesting ≠ predição:** desempenho histórico não garante resultados futuros.
- **Pesos longos apenas:** o modelo não permite posições vendidas (short selling).
- **Taxa livre de risco:** configure `risk_free_rate` em `config.yaml` de acordo com o mercado analisado.
- **Dados ausentes:** o programa preenche falhas com `forward fill / backward fill`. Períodos longos sem cotação podem distorcer os resultados.
