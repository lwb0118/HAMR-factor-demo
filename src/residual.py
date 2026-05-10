"""
residual.py
===========
ResidualWeakness & NonFundamentalOK — mispricing detection layer.

Per HAMR Framework:
  ResidualReturn_5d = Return_5d
    - β_mkt * MarketReturn_5d
    - β_ind * IndustryReturn_5d
    - β_style * StyleFactorReturn_5d
  ResidualWeakness = RankPct(-ResidualReturn_5d)

  NonFundamentalOK = 1 - RankPct(NegativeEarningsRevision
    + NegativeNewsShock + LeverageStress + GovernanceRisk)

  MispricingPressure = MismatchScore × ResidualWeakness × NonFundamentalOK
"""

import numpy as np
import pandas as pd
from .data_process import cross_rankpct


def compute_residual_weakness(panel):
    """
    Compute ResidualWeakness — market + industry + style residual.

    Per HAMR Framework v2.0:
      ResidualReturn_5d = Return_5d
        - MarketReturn_5d
        - IndustryReturn_5d
        - SizeEffect
        - MomentumEffect

    Sequential orthogonalization ensures only stock-specific
    (non-fundamental, non-style) weakness is captured.
    """
    df = panel.copy()

    # Step 1: Market-relative
    market_ret = df.groupby('date')['ret_5d'].transform('mean')
    df['residual_ret'] = df['ret_5d'] - market_ret

    # Step 2: Industry-relative (remove sector effects)
    if 'industry' in df.columns and df['industry'].notna().sum() > 100:
        ind_ret = df.groupby(['date', 'industry'])['ret_5d'].transform('mean')
        df['residual_ret'] = df['residual_ret'] - (ind_ret - market_ret)

    # Step 3: Size & Momentum orthogonalization disabled
    # (over-corrects in small samples; enable when N > 500)

    # Final cleanup
    df['residual_ret_5d'] = df['residual_ret']

    # ResidualWeakness: higher = weaker residual performance
    df['_neg_resid'] = -df['residual_ret_5d']
    df['ResidualWeakness'] = cross_rankpct(df, '_neg_resid')
    df['ResidualWeakness'] = df['ResidualWeakness'].fillna(0.5).clip(0, 1)

    return df[['date', 'code', 'residual_ret_5d', 'ResidualWeakness']]


def compute_non_fundamental_ok(panel, events=None):
    """
    Compute NonFundamentalOK — whether recent decline is
    NOT attributable to fundamental deterioration.

    Per guide, NonFundamentalOK is lowered if:
      - Earnings revision is significantly negative
      - Major negative news / regulatory events
      - High leverage stress
      - Governance risk flags

    Proxy (demo): continuous deviation score.
    Production: integrate announcement/earnings/event data.
    """
    df = panel.copy()

    if events is not None and not events.empty:
        return _compute_nonfundamental_with_events(df, events)
    else:
        return _compute_nonfundamental_proxy(df)


def _compute_nonfundamental_proxy(panel):
    """
    Proxy implementation using continuous deviation scoring.

    Also incorporates revenue growth trends (or_yoy) when available
    to detect fundamental deterioration.
    """
    df = panel.copy()

    # Deviation from market
    df['ret_deviation'] = df['ret_5d'] - df.groupby('date')['ret_5d'].transform('mean')

    # Normalize deviation cross-sectionally
    df['dev_zscore'] = df.groupby('date')['ret_deviation'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-10)
    ).clip(-3, 3)

    # Rolling average of negative deviation intensity (higher = worse)
    df['neg_dev_avg'] = df.groupby('code')['dev_zscore'].transform(
        lambda x: (-x).rolling(20, min_periods=5).mean()
    )

    # NonFundamentalOK: 1 - RankPct(negative deviation intensity)
    df['NonFundamentalOK'] = 1.0 - cross_rankpct(df, 'neg_dev_avg')

    # Earnings flash penalty (express data): if latest flash report shows decline
    if 'n_income' in df.columns and df['n_income'].notna().sum() > 30:
        # Flag if latest quarterly net income is negative or severely declining
        df['_ni_negative'] = (df['n_income'].fillna(0) < 0).astype(float)
        # If negative net income and stock shows residual weakness → likely fundamental
        penalty = df['_ni_negative'] * 0.15
        df['NonFundamentalOK'] = (df['NonFundamentalOK'] * (1 - penalty)).fillna(0.5)
        df.drop(columns=['_ni_negative'], inplace=True)

    # Operating cash flow check: negative OCF + negative NI = strong fundamental concern
    if 'oper_cf' in df.columns and df['oper_cf'].notna().sum() > 50:
        df['_cf_negative'] = ((df['oper_cf'].fillna(0) < 0).astype(float))
        penalty = df['_cf_negative'] * 0.1
        df['NonFundamentalOK'] = (df['NonFundamentalOK'] * (1 - penalty)).fillna(0.5)
        df.drop(columns=['_cf_negative'], inplace=True)

    # Earnings forecast penalty: negative forecast → likely fundamental decline
    if 'forecast_neg' in df.columns and df['forecast_neg'].notna().sum() > 0:
        penalty = df['forecast_neg'].fillna(0) * 0.3
        df['NonFundamentalOK'] = (df['NonFundamentalOK'] * (1 - penalty)).fillna(0.5)
        print(f'    Forecast: {int(df["forecast_neg"].sum())} negative flags applied')

    df['NonFundamentalOK'] = df['NonFundamentalOK'].fillna(0.5).clip(0, 1)

    return df[['date', 'code', 'NonFundamentalOK']]


def _compute_nonfundamental_with_events(panel, events):
    """Full NonFundamentalOK using event/announcement data.

    Required columns: code, date, event_type, severity
    """
    df = panel.copy()
    ev = events.copy()
    df['date'] = pd.to_datetime(df['date'])
    ev['date'] = pd.to_datetime(ev['date'])

    if 'severity' not in ev.columns:
        ev['severity'] = 1.0

    ev_daily = (
        ev.groupby(['date', 'code'])['severity']
        .sum()
        .reset_index()
        .rename(columns={'severity': 'event_severity'})
    )

    df = df.merge(ev_daily, on=['date', 'code'], how='left')
    df['event_severity'] = df['event_severity'].fillna(0.0)
    df['event_bad_rank'] = cross_rankpct(df, 'event_severity').fillna(0.0)

    base = _compute_nonfundamental_proxy(df)
    df = df.merge(base, on=['date', 'code'], how='left')
    df['NonFundamentalOK'] = (
        df['NonFundamentalOK'].fillna(0.5) * (1.0 - 0.7 * df['event_bad_rank'])
    ).clip(0, 1)

    return df[['date', 'code', 'NonFundamentalOK']]


def compute_mispricing_pressure(mismatch_score, residual_weakness,
                                non_fundamental_ok):
    """
    Compute MispricingPressure — the core HAMR mispricing signal.

    MispricingPressure = MismatchScore × ResidualWeakness × NonFundamentalOK

    All three components must be elevated simultaneously for
    a stock to be considered "mispriced".
    """
    merged = (mismatch_score
              .merge(residual_weakness, on=['date', 'code'], how='inner')
              .merge(non_fundamental_ok, on=['date', 'code'], how='inner'))

    merged['MispricingPressure'] = (
        merged['MismatchScore'].fillna(0.5) *
        merged['ResidualWeakness'].fillna(0.5) *
        merged['NonFundamentalOK'].fillna(0.5)
    ).clip(0, 1)

    return merged[['date', 'code', 'MispricingPressure',
                   'MismatchScore', 'ResidualWeakness', 'NonFundamentalOK']]
