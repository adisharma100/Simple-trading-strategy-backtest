"""
Simple trading strategy backtester (beginner project).

Strategies:
  1. Moving-average crossover (trend following)
  2. Z-score mean reversion

Key honesty features:
  - Signals are shifted by one day (no look-ahead bias)
  - Transaction costs are charged on every position change
  - Parameters are chosen on a TRAIN period and judged on an unseen TEST period
  - Everything is compared against buy & hold

Usage:
  python backtest.py --ticker "CL=F" --start 2015-01-01 --strategy both
  python backtest.py --ticker "^GSPC" --cost-bps 5 --allow-short
  python backtest.py --synthetic          # run offline with fake data
"""
import argparse
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ----------------------------------------------------------------- data
def load_data(ticker, start, end, synthetic=False):
    if not synthetic:
        try:
            import yfinance as yf
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=True, progress=False)
            if df is not None and not df.empty:
                close = df["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                return close.dropna().rename(ticker)
            print(f"[warn] No data returned for {ticker}.")
        except Exception as exc:
            print(f"[warn] Could not download data: {exc}")
        print("[warn] Falling back to SYNTHETIC data. Results are meaningless.\n")
    rng = np.random.default_rng(42)
    idx = pd.bdate_range(start, end or pd.Timestamp.today())
    rets = rng.normal(0.0003, 0.01, len(idx))
    return pd.Series(100 * np.exp(np.cumsum(rets)), index=idx,
                     name=f"{ticker} (synthetic)")


# ------------------------------------------------------------ strategies
def ma_crossover(close, fast, slow, allow_short=False):
    """Long when fast MA > slow MA. Flat (or short) otherwise."""
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    pos = pd.Series(0, index=close.index, dtype=float)
    pos[fast_ma > slow_ma] = 1
    if allow_short:
        pos[fast_ma < slow_ma] = -1
    pos[slow_ma.isna()] = 0
    return pos


def mean_reversion(close, window, entry, exit_=0.0, allow_short=False):
    """Buy when price is `entry` std-devs below its rolling mean; exit near the mean."""
    z = (close - close.rolling(window).mean()) / close.rolling(window).std()
    zv = z.to_numpy()
    pos = np.zeros(len(close))
    state = 0
    for i in range(len(close)):
        if np.isnan(zv[i]):
            state = 0
        elif state == 0:
            if zv[i] < -entry:
                state = 1
            elif allow_short and zv[i] > entry:
                state = -1
        elif state == 1 and zv[i] >= -exit_:
            state = 0
        elif state == -1 and zv[i] <= exit_:
            state = 0
        pos[i] = state
    return pd.Series(pos, index=close.index)


# -------------------------------------------------------------- backtest
def run_backtest(close, position, cost_bps):
    """Position decided at today's close is held from tomorrow (shift by 1)."""
    pos = position.shift(1).fillna(0)
    ret = close.pct_change().fillna(0)
    turnover = pos.diff().abs().fillna(pos.abs())
    strat_ret = pos * ret - turnover * cost_bps / 1e4
    return strat_ret, pos


def trade_returns(strat_ret, pos):
    """Return of each completed holding period (one entry to exit)."""
    run_id = (pos != pos.shift()).cumsum()
    out = []
    for _, grp in strat_ret.groupby(run_id):
        if pos.loc[grp.index[0]] != 0:
            out.append((1 + grp).prod() - 1)
    return pd.Series(out, dtype=float)


def metrics(strat_ret, pos):
    if len(strat_ret) < 2:
        return {}
    equity = (1 + strat_ret).cumprod()
    years = len(strat_ret) / TRADING_DAYS
    vol = strat_ret.std() * np.sqrt(TRADING_DAYS)
    trades = trade_returns(strat_ret, pos)
    return {
        "Total return": equity.iloc[-1] - 1,
        "CAGR": equity.iloc[-1] ** (1 / years) - 1,
        "Volatility": vol,
        "Sharpe": (strat_ret.mean() * TRADING_DAYS) / vol if vol > 0 else np.nan,
        "Max drawdown": (equity / equity.cummax() - 1).min(),
        "Trades": len(trades),
        "Win rate": (trades > 0).mean() if len(trades) else np.nan,
    }


def fmt(m):
    pct = {"Total return", "CAGR", "Volatility", "Max drawdown", "Win rate"}
    return {k: (f"{v:.1%}" if k in pct else f"{v:.2f}" if k == "Sharpe" else f"{v:.0f}")
            for k, v in m.items()}


# ------------------------------------------------------- parameter search
GRIDS = {
    "ma": {"fast": [10, 20, 30, 50], "slow": [50, 100, 150, 200]},
    "meanrev": {"window": [10, 20, 30, 60], "entry": [1.0, 1.5, 2.0]},
}


def make_signal(name, close, params, allow_short):
    if name == "ma":
        return ma_crossover(close, params["fast"], params["slow"], allow_short)
    return mean_reversion(close, params["window"], params["entry"],
                          allow_short=allow_short)


def candidate_params(name):
    g = GRIDS[name]
    keys = list(g)
    for combo in itertools.product(*g.values()):
        p = dict(zip(keys, combo))
        if name == "ma" and p["fast"] >= p["slow"]:
            continue
        yield p


def optimise(name, close, split, cost_bps, allow_short):
    """Pick the parameters with the best Sharpe on the TRAIN period only."""
    best, best_sharpe = None, -np.inf
    for p in candidate_params(name):
        ret, pos = run_backtest(close, make_signal(name, close, p, allow_short), cost_bps)
        m = metrics(ret.iloc[:split], pos.iloc[:split])
        if m and m["Sharpe"] > best_sharpe:
            best, best_sharpe = p, m["Sharpe"]
    return best


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticker", default="^GSPC")
    ap.add_argument("--start", default="2012-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--strategy", choices=["ma", "meanrev", "both"], default="both")
    ap.add_argument("--cost-bps", type=float, default=5.0,
                    help="cost per unit of position change, in basis points")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--allow-short", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    close = load_data(args.ticker, args.start, args.end, args.synthetic)
    split = int(len(close) * args.train_frac)
    print(f"{close.name}: {len(close)} days, {close.index[0].date()} to "
          f"{close.index[-1].date()}")
    print(f"Train: up to {close.index[split-1].date()} | "
          f"Test: from {close.index[split].date()}\n")

    names = ["ma", "meanrev"] if args.strategy == "both" else [args.strategy]
    labels = {"ma": "MA crossover", "meanrev": "Mean reversion"}
    curves, rows = {}, {}

    bh_ret = close.pct_change().fillna(0)
    bh_pos = pd.Series(1.0, index=close.index)
    curves["Buy & hold"] = (1 + bh_ret.iloc[split:]).cumprod()
    rows["Buy & hold (test)"] = fmt(metrics(bh_ret.iloc[split:], bh_pos.iloc[split:]))

    for name in names:
        params = optimise(name, close, split, args.cost_bps, args.allow_short)
        ret, pos = run_backtest(close, make_signal(name, close, params, args.allow_short),
                                args.cost_bps)
        print(f"{labels[name]}: best train params = {params}")
        rows[f"{labels[name]} (train)"] = fmt(metrics(ret.iloc[:split], pos.iloc[:split]))
        rows[f"{labels[name]} (test)"] = fmt(metrics(ret.iloc[split:], pos.iloc[split:]))
        curves[labels[name]] = (1 + ret.iloc[split:]).cumprod()

    table = pd.DataFrame(rows).T
    print("\n" + table.to_string() + "\n")
    table.to_csv("results.csv")

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, curve in curves.items():
        ax.plot(curve.index, curve.values, label=label,
                lw=2 if label == "Buy & hold" else 1.5)
    ax.set_title(f"Out-of-sample equity curves: {close.name} "
                 f"(cost {args.cost_bps:g} bps)")
    ax.set_ylabel("Growth of 1 unit")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("results.png", dpi=150)
    print("Saved results.csv and results.png")


if __name__ == "__main__":
    main()
