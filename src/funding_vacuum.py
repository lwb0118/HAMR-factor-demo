"""
funding_vacuum.py
=================
FundingVacuum — execution filter ensuring stock is NOT crowded.

Per HAMR Data Guide (formulas):
  AbTurnover(i,t) = turnover(i,t) / mean(turnover(i,t-60:t-1))
  AbAmount(i,t)   = amount(i,t) / mean(amount(i,t-60:t-1))
  IntradayVol(i,t)= (high(i,t) - low(i,t)) / pre_close(i,t)
  Crowding(i,t)   = AvgRank(AbTurnover, AbAmount, IntradayVol, VolumeSpike)
  FundingVacuum(i,t) = 1 - AvgRank(Crowding, StockAttention, PositiveAIFlow)

  LiquidityCapacity(i,t) = RankPct(dollar_volume)
  TrapGuard(i,t) = I(QualityScore > q40) × I(LiquidityCapacity > q30)
                    × I(No severe negative announcement)
"""

import numpy as np
import pandas as pd
from .data_process import cross_rankpct


def compute_funding_vacuum(panel):
    """
    Compute FundingVacuum — higher = less crowded.

    Uses main_net (institutional net flow from moneyflow) when available.
    """
    df = panel.copy()

    crowd_parts = []

    # Institutional flow signal (from moneyflow)
    if 'main_net' in df.columns and df['main_net'].notna().sum() > 50:
        crowd_parts.append(cross_rankpct(df, 'main_net').fillna(0.5))

    # Abnormal turnover
    if 'ab_turnover' in df.columns and df['ab_turnover'].notna().sum() > 50:
        crowd_parts.append(cross_rankpct(df, 'ab_turnover').fillna(0.5))
    elif 'turnover' in df.columns and df['turnover'].notna().sum() > 50:
        crowd_parts.append(cross_rankpct(df, 'turnover').fillna(0.5))

    # Abnormal amount
    if 'ab_amount' in df.columns and df['ab_amount'].notna().sum() > 50:
        crowd_parts.append(cross_rankpct(df, 'ab_amount').fillna(0.5))

    # Intraday volatility
    if 'intraday_vol' in df.columns and df['intraday_vol'].notna().sum() > 50:
        crowd_parts.append(cross_rankpct(df, 'intraday_vol').fillna(0.5))

    # Return magnitude
    crowd_parts.append(cross_rankpct(df, 'ret_5d').fillna(0.5))

    if not crowd_parts:
        df['FundingVacuum'] = 0.5
    else:
        crowd_composite = sum(p for p in crowd_parts) / len(crowd_parts)
        df['CrowdingScore'] = crowd_composite
        df['FundingVacuum'] = 1.0 - cross_rankpct(df, 'CrowdingScore')

    df['FundingVacuum'] = df['FundingVacuum'].fillna(0.5).clip(0, 1)

    return df[['date', 'code', 'FundingVacuum', 'CrowdingScore']]


def compute_liquidity_capacity(panel):
    """Compute LiquidityCapacity = RankPct(dollar_volume_20d)."""
    df = panel.copy()

    if 'dollar_volume' in df.columns:
        df['LiquidityCapacity'] = cross_rankpct(df, 'dollar_volume')
    else:
        df['LiquidityCapacity'] = 0.5

    df['LiquidityCapacity'] = df['LiquidityCapacity'].fillna(0.5).clip(0, 1)
    return df[['date', 'code', 'LiquidityCapacity']]


def compute_trap_guard(quality_score, liquidity_capacity, panel=None):
    """
    Compute TrapGuard — soft value-trap filter.

    Per framework:
      TrapGuard = I(QualityScore > q40) × I(LiquidityCapacity > q30)
                  × I(No severe negative announcement)

    Uses sigmoid-style soft boundary for robustness.
    """
    merged = quality_score[['date', 'code', 'QualityScore']].merge(
        liquidity_capacity[['date', 'code', 'LiquidityCapacity']],
        on=['date', 'code'], how='inner'
    )

    q = merged['QualityScore'].values
    l = merged['LiquidityCapacity'].values

    # Sigmoid soft thresholds
    q_filter = 1.0 / (1.0 + np.exp(-15 * (q - 0.35)))
    l_filter = 1.0 / (1.0 + np.exp(-15 * (l - 0.25)))

    merged['TrapGuard'] = (q_filter * l_filter).clip(0, 1)

    return merged[['date', 'code', 'TrapGuard']]
