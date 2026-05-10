"""
ic_test.py
==========
IC (Information Coefficient) testing suite.

Implements:
  - Cross-sectional Rank IC (Spearman)
  - ICIR (IC Information Ratio)
  - Newey-West robust t-statistic (Bartlett kernel)
  - Multi-horizon analysis
"""

import numpy as np
import pandas as pd
from scipy import stats


def compute_daily_rank_ic(panel, factor_col, return_col, min_stocks=10):
    """Compute daily cross-sectional Rank IC."""
    results = []
    for date, group in panel.groupby('date'):
        valid = group[[factor_col, return_col]].dropna()
        n = len(valid)
        if n < min_stocks:
            continue
        f_ranks = stats.rankdata(valid[factor_col].values)
        r_ranks = stats.rankdata(valid[return_col].values)
        ic, p_val = stats.pearsonr(f_ranks, r_ranks)
        results.append({'date': date, 'ic': ic, 'p_value': p_val, 'n_stocks': n})

    if not results:
        raise ValueError('No valid IC dates.')
    return pd.DataFrame(results).sort_values('date').reset_index(drop=True)


def compute_ic_statistics(ic_series):
    """Compute IC statistics with Newey-West adjustment."""
    n = len(ic_series)
    ic_mean = float(np.mean(ic_series))
    ic_std = float(np.std(ic_series, ddof=1))
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    t_stat = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 else 0.0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))
    ic_pos_ratio = float(np.sum(ic_series > 0) / n)
    nw_tstat = _newey_west_tstat(ic_series)

    return {
        'ic_mean': ic_mean, 'ic_std': ic_std, 'icir': icir,
        't_stat': t_stat, 'p_value': p_value,
        'ic_pos_ratio': ic_pos_ratio, 'n_periods': n,
        'nw_tstat': nw_tstat,
        'cumulative_ic': float(np.sum(ic_series))
    }


def _newey_west_tstat(series, lags=None):
    """Newey-West HAC with Bartlett kernel."""
    n = len(series)
    if n < 3:
        return 0.0
    if lags is None:
        lags = int(n ** 0.25)
    lags = min(lags, n - 2)
    mean_s = np.mean(series)
    gamma0 = np.sum((series - mean_s) ** 2) / n
    nw_var = gamma0
    for j in range(1, lags + 1):
        cov = np.sum((series[:-j] - mean_s) * (series[j:] - mean_s)) / n
        weight = 1.0 - j / (lags + 1)
        nw_var += 2.0 * weight * cov
    nw_se = np.sqrt(nw_var / n) if nw_var > 0 else 1e-10
    return float(mean_s / nw_se)


def full_ic_analysis(panel, factor_col='hamr_zscore', horizons=(1, 5, 10, 20)):
    """Full IC analysis across horizons."""
    results = {}
    for h in horizons:
        ret_col = f'fwd_{h}d'
        if ret_col not in panel.columns:
            continue
        ic_df = compute_daily_rank_ic(panel, factor_col, ret_col)
        stats_dict = compute_ic_statistics(ic_df['ic'].values)
        nw_t = stats_dict['nw_tstat']
        stats_dict['verdict'] = (
            'SIGNIFICANT' if abs(nw_t) > 1.96 else
            'MARGINAL' if abs(nw_t) > 1.28 else 'INSIGNIFICANT'
        )
        results[str(h)] = {'stats': stats_dict, 'ic_series': ic_df}
    return results
