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
    Fama-MacBeth: cross-sectional regression per date, average coefficients.

    Controls: QualityScore, ResidualWeakness, dollar_volume (size proxy),
              ret_20d (momentum), volatility_20d.
    """
    ret_col = f'fwd_{horizon}d'
    base_vars = ['hamr_zscore']
    control_vars = ['QualityScore', 'ResidualWeakness', 'dollar_volume',
                    'ret_20d', 'volatility_20d']
    needed = ['date'] + base_vars + [ret_col]
    if controls:
        needed += [c for c in control_vars if c in panel.columns]

    valid = panel[needed].dropna().copy()
    if len(valid) < 50:
        return {'beta_hamr': 0, 't_stat': 0, 'p_value': 1}

    betas = []
    for date, grp in valid.groupby('date'):
        if len(grp) < 10:
            continue
        X_cols = base_vars + [c for c in control_vars if c in grp.columns]
        X = grp[X_cols].values
        y = grp[ret_col].values
        try:
            coef, _, _, _ = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)
            betas.append(coef[1])  # HAMR_Final coefficient
        except Exception:
            pass

    if not betas:
        return {'beta_hamr': 0, 't_stat': 0, 'p_value': 1}

    betas = np.array(betas)
    mean_beta = float(betas.mean())
    se = betas.std(ddof=1) / np.sqrt(len(betas)) if len(betas) > 1 else 1
    se = max(se, 0.0001)
    t_stat = mean_beta / se
    p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=len(betas) - 1)))

    return {
        'beta_hamr': mean_beta, 't_stat': t_stat, 'p_value': p_value,
        'n_dates': len(betas),
        'fm_tstat': _newey_west_fm_tstat(betas),
    }


def _newey_west_fm_tstat(betas):
    """Newey-West t-stat for Fama-MacBeth coefficient time series."""
    n = len(betas)
    if n < 3:
        return 0.0
    lags = min(int(n ** 0.25), n - 2)
    m = np.mean(betas)
    gamma0 = np.sum((betas - m) ** 2) / n
    nw_var = gamma0
    for j in range(1, lags + 1):
        cov = np.sum((betas[:-j] - m) * (betas[j:] - m)) / n
        nw_var += 2.0 * (1 - j / (lags + 1)) * cov
    se = np.sqrt(nw_var / n) if nw_var > 0 else 1e-10
    return float(m / se) if se > 0 else 0.0


def regression_with_controls(panel, horizon=10):
    """
    4-model Fama-MacBeth: progressively add controls.

    Model 1: HAMR_Final only
    Model 2: + QualityScore + ResidualWeakness
    Model 3: + Size + Momentum controls
    Model 4: + Industry FE (when available)
    """
    ret_col = f'fwd_{horizon}d'
    models = {}

    # Model 1: HAMR only
    m1 = run_return_regression(panel[['date', 'hamr_zscore', ret_col]].dropna(),
                                horizon, controls=False)
    models['M1_HAMR_only'] = {
        'beta': m1['beta_hamr'], 't': m1['t_stat'],
        'n_dates': m1.get('n_dates', 0)
    }

    # Model 2: HAMR + Quality + Reversal
    cols = ['date', 'hamr_zscore', 'QualityScore', 'ResidualWeakness', ret_col]
    m2 = run_return_regression(panel[cols].dropna(), horizon, controls=True)
    models['M2_+Quality+Reversal'] = {
        'beta': m2['beta_hamr'], 't': m2['t_stat'],
        'n_dates': m2.get('n_dates', 0)
    }

    # Model 3: + Size + Momentum controls
    cols = ['date', 'hamr_zscore', 'QualityScore', 'ResidualWeakness',
            'dollar_volume', 'ret_20d', 'volatility_20d', ret_col]
    available = [c for c in cols if c in panel.columns]
    m3 = run_return_regression(panel[available].dropna(), horizon, controls=True)
    models['M3_+Size+Momentum'] = {
        'beta': m3['beta_hamr'], 't': m3['t_stat'],
        'n_dates': m3.get('n_dates', 0)
    }

    return models
