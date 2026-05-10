"""
regression.py
=============
Panel regression & mechanism testing for HAMR factor.

Per HAMR Framework section 14:

  FutureResidualReturn(i,t+N) = α + β1*HAMR_Final(i,t) + Controls + FE + ε

  Mechanism regressions:
    FutureAttentionRecovery = α + β1*HAMR_Final + Controls + FE
    FutureValuationRepair   = α + β1*HAMR_Final + Controls + FE
    FutureDrawdown          = α + β1*HAMR_Final + Controls + FE
"""

import numpy as np
import pandas as pd
from scipy import stats


def run_return_regression(panel, horizon=10, controls=True):
    """
    Panel regression: future return ~ HAMR_Final + controls.

    Simple Fama-MacBeth style: cross-sectional regression per date,
    then average coefficients across time.

    Returns:
        dict with beta_hamr, t_stat, p_value, n_dates
    """
    ret_col = f'fwd_{horizon}d'
    needed = ['date', 'hamr_zscore', ret_col]
    if controls:
        needed += ['QualityScore', 'LiquidityCapacity']

    valid = panel[needed].dropna().copy()
    if len(valid) < 50:
        return {'beta_hamr': 0, 't_stat': 0, 'p_value': 1, 'n_dates': 0}

    betas = []
    for date, grp in valid.groupby('date'):
        if len(grp) < 10:
            continue
        # Simple cross-sectional regression
        X = grp['hamr_zscore'].values
        y = grp[ret_col].values
        beta, _, _, _, _ = stats.linregress(X, y)
        betas.append(beta)

    if not betas:
        return {'beta_hamr': 0, 't_stat': 0, 'p_value': 1, 'n_dates': 0}

    betas = np.array(betas)
    mean_beta = float(betas.mean())
    se = betas.std(ddof=1) / np.sqrt(len(betas)) if len(betas) > 1 else 1
    t_stat = mean_beta / se if se > 0 else 0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(betas) - 1))

    return {
        'beta_hamr': mean_beta,
        't_stat': t_stat,
        'p_value': float(p_value),
        'n_dates': len(betas),
        'betas': betas
    }


def regression_with_controls(panel, horizon=10):
    """
    Panel regression with multiple controls.

    Compares HAMR_Final coefficient when adding control variables
    (QualityScore, size, volatility, momentum).
    """
    ret_col = f'fwd_{horizon}d'
    models = {}

    # Model 1: HAMR only
    r1 = run_return_regression(panel[['date', 'hamr_zscore', ret_col]],
                                horizon, controls=False)
    models['HAMR_only'] = r1

    # Model 2: HAMR + Quality + Size proxy (LiquidityCapacity)
    needed = ['date', 'hamr_zscore', 'QualityScore', 'LiquidityCapacity', ret_col]
    valid = panel[needed].dropna()
    betas = []
    for _, grp in valid.groupby('date'):
        if len(grp) < 15:
            continue
        X = grp[['hamr_zscore', 'QualityScore', 'LiquidityCapacity']].values
        y = grp[ret_col].values
        try:
            # OLS: (X'X)^-1 X'y
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
            betas.append(coef[0])  # HAMR coefficient
        except Exception:
            pass
    if betas:
        betas = np.array(betas)
        m = float(betas.mean())
        se = betas.std(ddof=1) / np.sqrt(len(betas)) if len(betas) > 1 else 1
        t = m / se if se > 0 else 0
        models['HAMR_with_controls'] = {
            'beta_hamr': m, 't_stat': t,
            'p_value': float(2 * (1 - stats.t.cdf(abs(t), df=len(betas) - 1))),
            'n_dates': len(betas)
        }
    else:
        models['HAMR_with_controls'] = {'beta_hamr': 0, 't_stat': 0}

    return models
