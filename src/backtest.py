"""
backtest.py
===========
HAMR portfolio backtest — quintile analysis + control comparison.

Per HAMR Framework section 11-13:
  - Quintile portfolio (Q1-Q5) by HAMR_Final
  - Monotonicity test (Spearman ρ)
  - Control comparison: Quality / Reversal / Quality×Reversal
  - Multi-horizon analysis (5d, 10d, 20d)
  - AIHeat state-conditional analysis
"""

import numpy as np
import pandas as pd
from scipy import stats


# ======================================================================
# Quintile Portfolio Test
# ======================================================================

def quintile_test(panel, factor_col='hamr_zscore', return_col='fwd_5d',
                   n_groups=5, min_stocks_per_date=10):
    """
    Per-date cross-sectional quintile test.

    Each trading day, sort stocks into n_groups by factor value,
    compute equal-weighted forward return per group.
    Then average group returns across dates.

    This is the standard academic methodology — NEVER mix dates.
    """
    valid = panel[[factor_col, return_col, 'date']].dropna().copy()

    # Per-date group returns
    date_group_returns = []
    for date, grp in valid.groupby('date'):
        if len(grp) < n_groups * 2:  # need enough stocks
            continue
        try:
            grp['group'] = pd.qcut(
                grp[factor_col].rank(method='first'),
                n_groups, labels=list(range(n_groups)), duplicates='drop'
            )
            gret = grp.groupby('group')[return_col].mean()
            for g in gret.index:
                date_group_returns.append({'date': date, 'group': g,
                                           'n_stocks': len(grp[grp['group'] == g]),
                                           'return': gret[g]})
        except Exception:
            pass

    if not date_group_returns:
        raise ValueError('No valid date-groups')

    df_grp = pd.DataFrame(date_group_returns)

    # Average across dates
    gstats = df_grp.groupby('group')['return'].agg(['mean', 'std', 'count'])
    groups = sorted(gstats.index)

    group_means = {f'Q{g+1}': float(gstats.loc[g, 'mean']) for g in groups}
    group_stds = {f'Q{g+1}': float(gstats.loc[g, 'std']) for g in groups}
    group_counts = {f'Q{g+1}': int(gstats.loc[g, 'count']) for g in groups}

    n_groups_valid = len(groups)
    spread = group_means[f'Q{n_groups_valid}'] - group_means['Q1']

    # Monotonicity
    ranks = list(range(1, n_groups_valid + 1))
    rets = [group_means[f'Q{i+1}'] for i in range(n_groups_valid)]
    mono, _ = stats.spearmanr(ranks, rets)

    return {
        'group_means': group_means,
        'group_stds': group_stds,
        'group_counts': group_counts,
        'spread': spread,
        'spread_annualized': (1 + spread) ** (252 / 20) - 1,  # 20d horizon
        'monotonicity': float(mono)
    }


# ======================================================================
# Control Factor Comparison
# ======================================================================

def compare_controls(panel, horizons=(5, 10)):
    """
    Compare HAMR vs control factors.

    Controls:
      - QualityScore only
      - ResidualWeakness only (short-term reversal proxy)
      - Quality × ResidualWeakness (naive combo)
    """
    results = {}
    df = panel.copy()

    # Build control factor columns
    df['quality_z'] = df.groupby('date')['QualityScore'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    df['reversal_z'] = df.groupby('date')['ResidualWeakness'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
    df['q_x_r'] = df['QualityScore'] * df['ResidualWeakness']
    df['q_x_r_z'] = df.groupby('date')['q_x_r'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )

    for h in horizons:
        ret_col = f'fwd_{h}d'
        if ret_col not in df.columns:
            continue

        comps = {}
        for name, col in [
            ('HAMR_Final', 'hamr_zscore'),
            ('QualityScore', 'quality_z'),
            ('ReversalOnly', 'reversal_z'),
            ('Quality_x_Reversal', 'q_x_r_z')
        ]:
            try:
                r = quintile_test(df.dropna(subset=[ret_col]),
                                  factor_col=col, return_col=ret_col)
                comps[name] = {
                    'spread': r['spread'],
                    'spread_ann': r['spread_annualized'],
                    'monotonicity': r['monotonicity'],
                    'group_means': r['group_means']
                }
            except Exception:
                comps[name] = {'spread': 0, 'monotonicity': 0}

        results[f'{h}d'] = comps

    return results


# ======================================================================
# AIHeat State-Conditional Analysis
# ======================================================================

def conditional_ic_by_aistate(panel, factor_col='hamr_zscore', horizon=10):
    """
    Compare IC in high vs low AIHeat states.

    Splits dates by median AIStateScore.
    """
    ret_col = f'fwd_{horizon}d'
    if ret_col not in panel.columns or 'AIStateScore' not in panel.columns:
        return None

    daily_ai = panel.groupby('date')['AIStateScore'].first()
    median_ai = daily_ai.median()

    high_dates = daily_ai[daily_ai >= median_ai].index
    low_dates = daily_ai[daily_ai < median_ai].index

    result = {}
    for label, dates in [('High AIHeat', high_dates), ('Low AIHeat', low_dates)]:
        sub = panel[panel['date'].isin(dates)].dropna(subset=[factor_col, ret_col])
        if len(sub) < 100:
            result[label] = {'ic_mean': 0, 'n_dates': 0}
            continue

        ics = []
        for _, grp in sub.groupby('date'):
            if len(grp) < 10:
                continue
            ic, _ = stats.spearmanr(grp[factor_col], grp[ret_col])
            ics.append(ic)

        ic_arr = np.array(ics)
        ic_mean = float(ic_arr.mean()) if len(ic_arr) > 0 else 0
        icir = ic_mean / ic_arr.std() if len(ic_arr) > 1 and ic_arr.std() > 0 else 0

        result[label] = {
            'ic_mean': ic_mean,
            'icir': icir,
            'n_dates': len(ics)
        }

    return result


# ======================================================================
# Performance Metrics
# ======================================================================

def compute_performance(returns, periods_per_year=252):
    """
    Compute standard performance metrics from return series.
    """
    r = np.array(returns)
    if len(r) < 2:
        return {'sharpe': 0, 'ann_ret': 0, 'max_dd': 0, 'win_rate': 0}

    ann_ret = float((1 + np.mean(r)) ** periods_per_year - 1)
    ann_vol = float(np.std(r, ddof=1) * np.sqrt(periods_per_year))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    cum = np.cumprod(1 + r)
    peak = np.maximum.accumulate(cum)
    max_dd = float(np.min(cum / peak - 1))

    win_rate = float(np.mean(r > 0))

    return {
        'sharpe': sharpe,
        'ann_ret': ann_ret,
        'ann_vol': ann_vol,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'n_periods': len(r),
        'total_ret': float(np.prod(1 + r) - 1)
    }
