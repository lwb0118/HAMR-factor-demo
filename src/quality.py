"""
quality.py
==========
QualityScore — fundamental quality barrier.

Per HAMR Framework:
  QualityScore = Avg(ProfitabilityRank, StabilityRank, CashFlowRank,
                     LeverageSafetyRank, AccrualQualityRank,
                     GrowthResilienceRank)

  Stocks with QualityScore < 40th percentile are excluded
  (hard filter or non-linear penalty via TrapGuard).

Production: Uses PIT-aligned financial data (ROE, ROA, gross margin,
OCF/NI, D/E, accruals, earnings revisions).

Proxy (demo): Price-based quality proxies.
"""

import numpy as np
import pandas as pd
from .data_process import cross_rankpct


def compute_quality_score(panel, financials=None):
    """
    Compute QualityScore from available data.

    If financial data columns (roe, roa, etc.) are in panel → use fundamentals.
    Otherwise → fall back to price-based proxies.
    """
    fin_cols = ['roe', 'roa', 'debt_to_assets', 'grossprofit_margin']
    has_fin = all(c in panel.columns for c in fin_cols[:2])  # at least ROE/ROA

    if has_fin and panel['roe'].notna().sum() > 50:
        return _compute_quality_fundamental(panel)
    else:
        return _compute_quality_proxy(panel)


def _compute_quality_proxy(panel):
    """
    Price-based QualityScore proxy.

    Normal investors prefer:
      - Lower volatility (more stable)
      - Higher liquidity (more investable)
      - More stable returns (less extreme moves)
      - Lower Amihud illiquidity (easier to trade)
    """
    df = panel.copy()

    # 1. Volatility inverse
    if 'volatility_20d' in df.columns:
        q_vol = 1.0 - cross_rankpct(df, 'volatility_20d')
    else:
        q_vol = pd.Series(0.5, index=df.index)

    # 2. Liquidity
    if 'dollar_volume' in df.columns:
        q_liq = cross_rankpct(df, 'dollar_volume')
    else:
        q_liq = pd.Series(0.5, index=df.index)

    # 3. Return stability (inverse of return dispersion)
    df['ret_stability'] = df.groupby('code')['ret_1d'].transform(
        lambda x: 1.0 / (x.rolling(20).std() + 0.01)
    )
    q_stability = cross_rankpct(df, 'ret_stability')

    # 4. Amihud illiquidity inverse
    if 'amihud' in df.columns:
        q_amihud = 1.0 - cross_rankpct(df, 'amihud')
    else:
        q_amihud = pd.Series(0.5, index=df.index)

    # Composite (equal weight)
    df['QualityScore'] = (
        0.25 * q_vol.fillna(0.5) +
        0.25 * q_liq.fillna(0.5) +
        0.25 * q_stability.fillna(0.5) +
        0.25 * q_amihud.fillna(0.5)
    )
    df['QualityScore'] = cross_rankpct(df, 'QualityScore')
    df['QualityScore'] = df['QualityScore'].fillna(0.5).clip(0, 1)

    # Quality flags for filtering
    df['quality_ok'] = (df['QualityScore'] >= 0.4).astype(int)

    return df[['date', 'code', 'QualityScore', 'quality_ok']]


def _compute_quality_fundamental(panel, financials=None):
    """
    Compute QualityScore from financial statement data (PIT-aligned).

    Uses columns already merged into panel:
      - roe, roa → profitability
      - grossprofit_margin → margin stability
      - debt_to_assets → leverage safety (inverse)
      - current_ratio → short-term safety
      - bps → book value growth
      - ocf_to_operating_ni → cash flow quality
    """
    df = panel.copy()

    scores = []

    # 1. ROE (higher = better)
    if 'roe' in df.columns and df['roe'].notna().sum() > 20:
        scores.append(cross_rankpct(df, 'roe').fillna(0.5))

    # 2. ROA (higher = better)
    if 'roa' in df.columns and df['roa'].notna().sum() > 20:
        scores.append(cross_rankpct(df, 'roa').fillna(0.5))

    # 3. Gross margin (higher = better)
    if 'grossprofit_margin' in df.columns and df['grossprofit_margin'].notna().sum() > 20:
        scores.append(cross_rankpct(df, 'grossprofit_margin').fillna(0.5))

    # 4. Debt ratio (lower = better → inverse)
    if 'debt_to_assets' in df.columns and df['debt_to_assets'].notna().sum() > 20:
        scores.append((1 - cross_rankpct(df, 'debt_to_assets')).fillna(0.5))

    # 5. Current ratio (higher = better)
    if 'current_ratio' in df.columns and df['current_ratio'].notna().sum() > 20:
        scores.append(cross_rankpct(df, 'current_ratio').fillna(0.5))

    # 6. BPS growth (higher = better)
    if 'bps' in df.columns and df['bps'].notna().sum() > 20:
        scores.append(cross_rankpct(df, 'bps').fillna(0.5))

    # 7. OCF/Operating NI (enable with consistent coverage)
    # if 'ocf_to_operating_ni' in df.columns and df['ocf_to_operating_ni'].notna().sum() > 100:
    #     scores.append(cross_rankpct(df, 'ocf_to_operating_ni').fillna(0.5))

    # 8. Revenue YoY (enable with consistent coverage)
    # if 'or_yoy' in df.columns and df['or_yoy'].notna().sum() > 100:
    #     scores.append(cross_rankpct(df, 'or_yoy').fillna(0.5))

    # 9. Operating cash flow (from income/cashflow merge, higher = more cash generation)
    if 'oper_cf' in df.columns and df['oper_cf'].notna().sum() > 100:
        scores.append(cross_rankpct(df, 'oper_cf').fillna(0.5))

    # 10. Net income (from income statement, higher = more profitable)
    if 'n_income' in df.columns and df['n_income'].notna().sum() > 100:
        scores.append(cross_rankpct(df, 'n_income').fillna(0.5))

    if scores:
        df['QualityScore'] = sum(scores) / len(scores)
    else:
        df['QualityScore'] = 0.5

    df['QualityScore'] = cross_rankpct(df, 'QualityScore')
    df['QualityScore'] = df['QualityScore'].fillna(0.5).clip(0, 1)
    df['quality_ok'] = (df['QualityScore'] >= 0.4).astype(int)

    return df[['date', 'code', 'QualityScore', 'quality_ok']]
