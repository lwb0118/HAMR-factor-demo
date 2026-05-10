"""
portfolio_backtest.py
=====================
TopN portfolio construction with transaction costs.

Per HAMR v2.0 P0:
  - Weekly/bi-weekly rebalance
  - Top 30 long / Bottom 30 short (market-neutral)
  - Turnover + cost model (0.05%-0.30% per side)
  - Gross vs net equity curve
  - Sharpe, Sortino, max drawdown
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent


# ======================================================================
# Rebalance schedule
# ======================================================================

def get_rebalance_dates(panel, freq='W-FRI'):
    """
    Get rebalance dates from panel.

    Args:
        freq: 'W-FRI' (weekly Fri), '2W-FRI' (bi-weekly), 'M' (monthly)
    """
    all_dates = pd.Series(sorted(panel['date'].unique()))
    if 'W' in freq:
        # Pick last trading day of each week
        weeks = all_dates.dt.isocalendar().week
        year_week = all_dates.dt.strftime('%Y-%W')
        last_per_week = all_dates.groupby(year_week).transform('max')
        dates = sorted(set(last_per_week))
    elif 'M' in freq:
        months = all_dates.dt.strftime('%Y-%m')
        last_per_month = all_dates.groupby(months).transform('max')
        dates = sorted(set(last_per_month))
    else:
        dates = all_dates.tolist()

    # Bi-weekly: take every other
    if '2W' in freq:
        dates = dates[::2]

    return [pd.to_datetime(d) for d in dates]


# ======================================================================
# Portfolio construction
# ======================================================================

def build_topn_portfolio(panel, factor_col='hamr_zscore', top_n=30,
                          rebalance_freq='2W-FRI', side='long_only'):
    """
    TopN portfolio. side='long_only' for A-share reality.
    """

    Long top_n stocks by factor, short bottom_n stocks.
    Equal weight within each leg.
    Rebalances on rebalance_freq schedule.

    Returns DataFrame with per-period returns.
    """
    dates = get_rebalance_dates(panel, rebalance_freq)
    if len(dates) < 2:
        return pd.DataFrame(columns=['entry_date', 'exit_date', 'gross_ret',
                                      'turnover', 'cost', 'net_ret'])

    periods = []
    prev_long = set()
    prev_short = set()

    for i, entry_date in enumerate(dates[:-1]):
        exit_date = dates[i + 1]

        # Current day's stocks
        day = panel[panel['date'] == entry_date].dropna(subset=[factor_col])
        if len(day) < top_n * 3:
            continue

        # Only tradable stocks
        if 'tradable' in day.columns:
            day = day[day['tradable'] == 1]
            if len(day) < top_n * 2:
                continue

        n = min(top_n, len(day) // 3)
        top = day.nlargest(n, factor_col)

        # Weights based on side
        if side == 'long_only':
            weights = {c: 1.0 / n for c in top['code'].tolist()}
        else:
            bot = day.nsmallest(n, factor_col)
            long_w = {c: 1.0 / n for c in top['code'].tolist()}
            short_w = {c: -1.0 / n for c in bot['code'].tolist()}
            weights = {**long_w, **short_w}

        n_long = n
        n_short = n if side != 'long_only' else 0

        # Turnover
        new_stocks = set(weights.keys())
        old_stocks = prev_long | (prev_short if side != 'long_only' else prev_long)
        turnover = len(new_stocks - old_stocks) + len(old_stocks - new_stocks)
        turnover_frac = turnover / max(1, len(new_stocks))

        # Gross return
        gross_ret = 0.0
        n_valid = 0
        for code, w in weights.items():
            code_data = panel[(panel['code'] == code) &
                              (panel['date'].isin([entry_date, exit_date]))]
            if code_data['date'].nunique() == 2:
                prices = code_data.sort_values('date')['close']
                ret = prices.iloc[-1] / prices.iloc[0] - 1
                gross_ret += w * ret
                n_valid += 1

        if n_valid > 0:
            gross_ret = gross_ret * (len(long_w) + len(short_w)) / n_valid
        else:
            gross_ret = 0.0

        periods.append({
            'entry_date': entry_date,
            'exit_date': exit_date,
            'gross_ret': gross_ret,
            'turnover': turnover_frac,
            'n_long': n,
            'n_short': n,
        })

        prev_long = set(long_w.keys())
        prev_short = set(short_w.keys())

    if not periods:
        return pd.DataFrame()

    perf = pd.DataFrame(periods)
    return perf


# ======================================================================
# Cost model
# ======================================================================

def apply_costs(perf, cost_per_side=0.001):
    """
    Apply transaction costs to portfolio returns.

    cost = turnover × cost_per_side
    net_ret = gross_ret - cost

    Tests multiple cost levels: [0.0005, 0.001, 0.002, 0.003]
    """
    perf['cost'] = perf['turnover'] * cost_per_side
    perf['net_ret'] = perf['gross_ret'] - perf['cost']
    return perf


def cost_sensitivity(perf, cost_levels=(0.0005, 0.001, 0.002, 0.003)):
    """Test portfolio across cost levels."""
    results = {}
    for c in cost_levels:
        pf = apply_costs(perf.copy(), c)
        metrics = compute_portfolio_metrics(pf)
        results[f'{c:.2%}'] = metrics
    return results


# ======================================================================
# Performance metrics
# ======================================================================

def compute_portfolio_metrics(perf, periods_per_year=52):
    """
    Compute comprehensive portfolio metrics.

    Returns: sharpe, sortino, ann_ret, ann_vol, max_dd, win_rate,
             calmar, n_periods, total_ret, avg_turnover
    """
    if perf.empty or len(perf) < 3:
        return {'sharpe': 0, 'ann_ret': 0, 'max_dd': 0, 'n_periods': 0}

    net = perf['net_ret'].values
    n = len(net)

    # Equity curve
    cum = np.cumprod(1 + net)
    total_ret = float(cum[-1] - 1)

    # Annualized metrics
    ann_ret = float((1 + total_ret) ** (periods_per_year / n) - 1)
    ann_vol = float(np.std(net, ddof=1) * np.sqrt(periods_per_year))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    # Sortino (downside-only volatility)
    downside = net[net < 0]
    sortino_vol = float(np.std(downside, ddof=1) * np.sqrt(periods_per_year)) if len(downside) > 1 else ann_vol
    sortino = ann_ret / sortino_vol if sortino_vol > 0 else 0.0

    # Max drawdown
    peak = np.maximum.accumulate(cum)
    max_dd = float(np.min(cum / peak - 1))

    # Calmar ratio
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0.0

    # Win rate
    win_rate = float(np.mean(net > 0))

    return {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'ann_ret': round(ann_ret, 4),
        'ann_vol': round(ann_vol, 4),
        'max_dd': round(max_dd, 4),
        'calmar': round(calmar, 3),
        'win_rate': round(win_rate, 3),
        'total_ret': round(total_ret, 4),
        'n_periods': n,
        'avg_turnover': round(float(perf['turnover'].mean()), 3) if 'turnover' in perf else 0,
    }


def run_portfolio_backtest(panel, factor_col='hamr_zscore', top_n=30,
                            rebalance_freq='W-FRI', cost_levels=(0.0005, 0.001, 0.002, 0.003)):
    """
    Complete portfolio backtest pipeline.

    1. Build TopN portfolio
    2. Apply cost sensitivity
    3. Report metrics per cost level

    Returns dict with perf_df, cost_results, summary.
    """
    print(f'\n[Portfolio] Top{top_n}, rebalance={rebalance_freq}')

    perf = build_topn_portfolio(panel, factor_col, top_n, rebalance_freq)
    if perf.empty:
        print('  No valid periods')
        return None

    cost_results = cost_sensitivity(perf, cost_levels)

    print(f'\n  {"Cost":>8s} | {"Sharpe":>6s} | {"AnnRet":>8s} | {"MaxDD":>8s} | {"WinRate":>7s} | Turnover')
    print(f'  {"-"*60}')
    for cost_label, m in cost_results.items():
        print(f'  {cost_label:>8s} | {m["sharpe"]:>6.2f} | {m["ann_ret"]:>8.2%} | '
              f'{m["max_dd"]:>8.2%} | {m["win_rate"]:>7.1%} | {m.get("avg_turnover",0):.2f}')

    return {'perf': perf, 'cost_results': cost_results, 'n_periods': len(perf)}
