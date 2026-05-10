"""
mechanism.py
============
Mechanism tests — verify HAMR's economic logic is actually driving returns.

Per HAMR Framework section 14.2:
  1. FutureAttentionRecovery → does high HAMR attract attention later?
  2. FutureValuationRepair → does high HAMR lead to valuation recovery?
  3. FutureDrawdown → does high HAMR avoid further significant drawdown?

Key test: HAMR should NOT be just buying recent losers (short-term reversal).
"""

import numpy as np
import pandas as pd
from scipy import stats


def test_attention_recovery(panel, horizon=10):
    """
    Test attention recovery: do high-HAMR stocks see future attention increase?

    Uses turnover/dollar_volume change as attention proxy.
    """
    ret_col = f'fwd_{horizon}d'
    if ret_col not in panel.columns:
        return None

    valid = panel.dropna(subset=['hamr_zscore', 'turnover', 'dollar_volume'])
    if len(valid) < 50:
        return None

    # Future turnover change
    valid['fwd_turnover'] = valid.groupby('code')['turnover'].shift(-horizon)
    valid['turnover_change'] = valid['fwd_turnover'] / (valid['turnover'] + 0.01)

    # IC between HAMR and future attention change
    ics = []
    for date, grp in valid.groupby('date'):
        if len(grp) < 10:
            continue
        from scipy import stats
        ic, _ = stats.spearmanr(grp['hamr_zscore'], grp['turnover_change'])
        ics.append(ic)

    if not ics:
        return None
    ics = np.array(ics)
    return {
        'mean_ic': float(ics.mean()),
        'icir': float(ics.mean() / ics.std()) if ics.std() > 0 else 0,
        'pct_positive': float(np.mean(ics > 0)),
        'n_dates': len(ics),
        'attention_variable': 'turnover_change'
    }


def test_reversal_vs_hamr(panel, horizon=10):
    """
    Critical test: Does HAMR predict returns AFTER controlling for
    short-term reversal?

    If HAMR is just buying recent losers, it should not have
    incremental predictive power beyond ResidualWeakness.
    """
    ret_col = f'fwd_{horizon}d'
    valid = panel[['date', 'code', 'hamr_zscore', 'ResidualWeakness', ret_col]].dropna()

    # Double sort: within each ResidualWeakness quintile, test HAMR IC
    try:
        valid['rev_q'] = valid.groupby('date')['ResidualWeakness'].transform(
            lambda x: pd.qcut(x, 3, labels=['Low', 'Mid', 'High'], duplicates='drop')
        )
    except Exception:
        # If qcut fails, split by quantile manually
        valid['rev_q'] = valid.groupby('date')['ResidualWeakness'].transform(
            lambda x: pd.cut(x, bins=[-np.inf, x.quantile(0.33), x.quantile(0.67), np.inf],
                           labels=['Low', 'Mid', 'High'])
        )

    results = {}
    for q in ['Low', 'Mid', 'High']:
        sub = valid[valid['rev_q'] == q]
        if len(sub) < 50:
            continue
        ics = []
        for date, grp in sub.groupby('date'):
            if len(grp) < 8:
                continue
            ic, _ = stats.spearmanr(grp['hamr_zscore'], grp[ret_col])
            ics.append(ic)
        if ics:
            ics = np.array(ics)
            results[q] = {
                'ic_mean': float(ics.mean()),
                'n_dates': len(ics),
                't_stat': float(ics.mean() / ics.std() * np.sqrt(len(ics)))
                    if ics.std() > 0 else 0
            }

    return results


def test_factor_orthogonality(panel):
    """
    Test whether HAMR is orthogonal to standard factors.

    Computes correlations between HAMR_Final and common factor proxies:
      - Quality (QualityScore)
      - Reversal (ResidualWeakness)
      - Momentum (ret_20d)
      - Size (LiquidityCapacity as inverse size proxy)
      - Low Volatility (volatility_20d inverse)
    """
    proxies = {
        'Quality': 'QualityScore',
        'Reversal': 'ResidualWeakness',
        'Momentum': 'ret_20d',
        'Liquidity': 'LiquidityCapacity',
        'Mismatch': 'MismatchScore',
    }
    available = {k: v for k, v in proxies.items()
                 if v in panel.columns}

    valid = panel[['date', 'hamr_zscore'] + list(available.values())].dropna()

    correlations = {}
    for name, col in available.items():
        cors = []
        for _, grp in valid.groupby('date'):
            if len(grp) < 10:
                continue
            cor, _ = stats.spearmanr(grp['hamr_zscore'], grp[col])
            cors.append(cor)
        correlations[name] = float(np.nanmean(cors)) if cors else float('nan')

    return correlations


def generate_mechanism_report(panel):
    """Generate complete mechanism test report."""
    report = {}

    # 1. Factor orthogonality
    report['factor_correlations'] = test_factor_orthogonality(panel)

    # 2. Reversal vs HAMR double-sort
    report['reversal_double_sort'] = test_reversal_vs_hamr(panel)

    # 3. Attention recovery
    report['attention_recovery'] = test_attention_recovery(panel)

    # 4. AIHeat conditional
    from .backtest import conditional_ic_by_aistate
    report['aistate_conditional'] = conditional_ic_by_aistate(panel)

    # 4. Exclusion test: remove negative forecast stocks
    if 'forecast_neg' in panel.columns:
        clean = panel[panel['forecast_neg'].fillna(0) == 0]
        if len(clean) > 100:
            from .backtest import conditional_ic_by_aistate
            report['exclusion_test'] = conditional_ic_by_aistate(clean)

    return report


def run_double_sort(panel, horizon=10):
    """
    Double sort: Quality × Mismatch/ResidualWeakness.
    Verify only 'high quality + high mismatch + residual weakness' works.
    """
    ret_col = f'fwd_{horizon}d'
    valid = panel[['date', 'code', 'QualityScore', 'MismatchScore',
                   'ResidualWeakness', ret_col]].dropna()
    if len(valid) < 50:
        return {}

    results = {}
    for q_var in ['QualityScore']:
        for m_var in ['MismatchScore', 'ResidualWeakness']:
            try:
                valid['q_high'] = valid.groupby('date')[q_var].transform(
                    lambda x: (x > x.median()).astype(int))
                valid['m_high'] = valid.groupby('date')[m_var].transform(
                    lambda x: (x > x.median()).astype(int))

                label = f'{q_var[:2]}+{m_var[:2]}'
                for q in [0, 1]:
                    for m in [0, 1]:
                        sub = valid[(valid['q_high'] == q) & (valid['m_high'] == m)]
                        if len(sub) < 50:
                            continue
                        key = f'{label}_Q{q}_M{m}'
                        ret = sub.groupby('date')[ret_col].mean().mean()
                        n = len(sub)
                        results[key] = {'return': float(ret), 'n': n}
            except Exception:
                pass
    return results
