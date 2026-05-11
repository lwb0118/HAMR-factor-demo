"""
hamr_factor.py
==============
HAMR factor assembly — combines all sub-variables into final signal.

Per HAMR Framework (section 10):

  MispricingPressure(i,t) = MismatchScore × ResidualWeakness × NonFundamentalOK
  HAMR_Core(i,t) = QualityScore × MispricingPressure × FundingVacuum
  HAMR_Final(i,t) = AIStateScore × HAMR_Core × LiquidityCapacity × TrapGuard

All sub-variables ∈ [0, 1], direction-unified (higher = stronger signal).
"""

import numpy as np
import pandas as pd
from .data_process import cross_rankpct


def compute_hamr(panel, ai_state, mismatch, quality, mispricing, vacuum,
                 liquidity, trap_guard, verbose=True):
    """
    Assemble the full HAMR factor.

    Parameters
    ----------
    panel : pd.DataFrame — base panel with [date, code, close]
    ai_state : pd.DataFrame — [date, AIStateScore]
    mismatch : pd.DataFrame — [date, code, MismatchScore]
    quality : pd.DataFrame — [date, code, QualityScore]
    mispricing : pd.DataFrame — [date, code, MispricingPressure]
    vacuum : pd.DataFrame — [date, code, FundingVacuum]
    liquidity : pd.DataFrame — [date, code, LiquidityCapacity]
    trap_guard : pd.DataFrame — [date, code, TrapGuard]

    Returns
    -------
    pd.DataFrame with all factor columns + HAMR_Final.
    """
    if verbose:
        print('  Assembling HAMR factor...')

    df = panel[['date', 'code', 'close']].copy()

    # Merge all components
    df = df.merge(mismatch[['date', 'code', 'MismatchScore']],
                  on=['date', 'code'], how='left')
    df = df.merge(quality[['date', 'code', 'QualityScore']],
                  on=['date', 'code'], how='left')
    df = df.merge(mispricing[['date', 'code', 'MispricingPressure',
                               'ResidualWeakness', 'NonFundamentalOK']],
                  on=['date', 'code'], how='left')
    df = df.merge(vacuum[['date', 'code', 'FundingVacuum']],
                  on=['date', 'code'], how='left')
    df = df.merge(liquidity[['date', 'code', 'LiquidityCapacity']],
                  on=['date', 'code'], how='left')
    df = df.merge(trap_guard[['date', 'code', 'TrapGuard']],
                  on=['date', 'code'], how='left')
    df = df.merge(ai_state[['date', 'AIStateScore']],
                  on='date', how='left')

    # Fill missing
    for col in ['MismatchScore', 'QualityScore', 'MispricingPressure',
                'ResidualWeakness', 'NonFundamentalOK', 'FundingVacuum',
                'LiquidityCapacity', 'TrapGuard']:
        df[col] = df[col].fillna(0.5).clip(0, 1)
    df['AIStateScore'] = df['AIStateScore'].fillna(0.5).clip(0, 1)

    # ================================================================
    # Core HAMR Formula — per HAMR document specification
    # ================================================================

    # HAMR_Core: quality + mispricing + vacuum
    df['HAMR_Core'] = (
        df['QualityScore'] *
        df['MispricingPressure'] *
        df['FundingVacuum']
    ).clip(0, 1)

    # HAMR_Final: state-adjusted + tradability-filtered
    df['HAMR_Final'] = (
        df['AIStateScore'] *
        df['HAMR_Core'] *
        df['LiquidityCapacity'] *
        df['TrapGuard']
    ).clip(0, 1)

    # Z-score for IC testing
    df['hamr_zscore'] = df.groupby('date')['HAMR_Final'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )

    if verbose:
        n_valid = df.dropna(subset=['HAMR_Final'])['code'].nunique()
        print(f'  HAMR_Final: [{df["HAMR_Final"].min():.4f}, '
              f'{df["HAMR_Final"].max():.4f}] across {n_valid} stocks')

    return df


def compute_forward_returns(panel, horizons=(1, 5, 10, 20)):
    """Compute forward returns. fwd_{h}d = close(t+h)/close(t) - 1."""
    df = panel.copy()
    df.sort_values(['code', 'date'], inplace=True)

    for h in horizons:
        col = f'fwd_{h}d'
        df[col] = df.groupby('code')['close'].transform(
            lambda x: x.shift(-h) / x - 1
        )
    return df


def decompose_hamr(df):
    """
    Decompose HAMR_Final into sub-component contributions.

    Returns attribution dict for each sub-variable.
    """
    components = {
        'AIStateScore': 'AIStateScore',
        'QualityScore': 'QualityScore',
        'MismatchScore': 'MismatchScore',
        'ResidualWeakness': 'ResidualWeakness',
        'NonFundamentalOK': 'NonFundamentalOK',
        'FundingVacuum': 'FundingVacuum',
        'LiquidityCapacity': 'LiquidityCapacity',
        'TrapGuard': 'TrapGuard',
    }

    valid = df.dropna(subset=['HAMR_Final'])
    attribution = {}

    for name, col in components.items():
        if col in valid.columns:
            attribution[name] = {
                'mean': float(valid[col].mean()),
                'std': float(valid[col].std()),
                'corr_with_hamr': float(valid[col].corr(valid['HAMR_Final']))
            }

    return attribution
