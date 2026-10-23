# Simple Trading Strategy Backtester

A beginner-friendly backtest of two classic strategies on any Yahoo Finance ticker
(index, commodity future, FX pair, etc.).

## Question
Do a moving-average crossover or a z-score mean-reversion strategy beat buy & hold
after transaction costs, on data the parameters were *not* tuned on?

## Method
- **Data:** daily adjusted closes via `yfinance`
- **Strategies:** MA crossover (trend) and z-score mean reversion
- **No look-ahead:** a signal at today's close is traded from tomorrow
- **Costs:** charged in basis points on every position change
- **Validation:** parameters are optimised on the first 70% of data (train),
  then evaluated on the last 30% (test)
- **Metrics:** total return, CAGR, volatility, Sharpe, max drawdown, trades, win rate

## Run
```
pip install -r requirements.txt
python backtest.py --ticker "CL=F" --start 2012-01-01 --strategy both
python backtest.py --ticker "^GSPC" --cost-bps 10 --allow-short
```
Outputs `results.csv` and `results.png`. Use `--synthetic` to test offline.
Useful tickers: `^GSPC` (S&P 500), `CL=F` (WTI crude), `GC=F` (gold),
`EURUSD=X`, `^TNX` (US 10Y yield).

## Results
_Paste your results table and chart here, then write 3-4 lines on what you saw._

## Limitations (fill in with your own findings)
- Parameter search on a small grid still risks overfitting; a good train Sharpe
  often disappears in test
- One train/test split is a single sample; try walk-forward windows
- Costs ignore slippage and bid-ask spread, which matter more in illiquid markets
- Daily closes only; no intraday behaviour, no position sizing or risk limits
- Results depend heavily on the asset and period (regime change)

## Ideas to extend
- Add position sizing by volatility targeting
- Compare long-only vs long/short
- Run across 5 instruments and compare which strategy suits which market
- Add a stop-loss and see whether it helps or hurts
