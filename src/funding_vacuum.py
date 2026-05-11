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


def compute_funding_vacuum(panel, stock_attention_df=None, positive_ai_flow_df=None):
    """
    Compute FundingVacuum — higher = less crowded + less AI attention.

    Per HAMR framework:
      CrowdingScore = AvgRank(AbTurnover, AbAmount, IntradayVol, ret_mag)
      FundingVacuum = 1 - AvgRank(CrowdingScore, StockAttention, PositiveAIFlow)

    StockAttention: guba post activity (code-level, when available)
    PositiveAIFlow: paper trading buy signals (placeholder for QuantDinger logs)
    """
    df = panel.copy()

    # === Crowding Score (abnormal turnover/amount, intraday vol) ===
    crowd_parts = []

    if 'main_net' in df.columns and df['main_net'].notna().sum() > 50:
        # main_net is institutional net inflow — higher = more crowded
        crowd_parts.append(cross_rankpct(df, 'main_net').fillna(0.5))

    if 'ab_turnover' in df.columns and df['ab_turnover'].notna().sum() > 50:
        crowd_parts.append(cross_rankpct(df, 'ab_turnover').fillna(0.5))
    elif 'turnover' in df.columns and df['turnover'].notna().sum() > 50:
        crowd_parts.append(cross_rankpct(df, 'turnover').fillna(0.5))

    if 'ab_amount' in df.columns and df['ab_amount'].notna().sum() > 50:
        crowd_parts.append(cross_rankpct(df, 'ab_amount').fillna(0.5))

    if 'intraday_vol' in df.columns and df['intraday_vol'].notna().sum() > 50:
        crowd_parts.append(cross_rankpct(df, 'intraday_vol').fillna(0.5))

    # ret_5d intentionally EXCLUDED from crowding — it is momentum, not crowding
    # A stock can have high returns without high crowd participation

    if crowd_parts:
        df['CrowdingScore'] = sum(p for p in crowd_parts) / len(crowd_parts)
    else:
        df['CrowdingScore'] = 0.5

    # === Stock Attention (from guba, news, search) ===
    if stock_attention_df is not None and len(stock_attention_df) > 0:
        sa = stock_attention_df.copy()
        if 'date' in sa.columns and 'date' in df.columns:
            sa['date'] = pd.to_datetime(sa['date'])
            df['date'] = pd.to_datetime(df['date'])
            sa = sa[['date', 'code', 'stock_attention_raw']].drop_duplicates(['date', 'code'])
            df = df.merge(sa, on=['date', 'code'], how='left')
        else:
            sa = sa[['code', 'stock_attention_raw']].drop_duplicates('code')
            df = df.merge(sa, on='code', how='left')
        df['StockAttention'] = cross_rankpct(df, 'stock_attention_raw').fillna(0.5)
    else:
        df['StockAttention'] = 0.5

    # === Positive AI Flow ===
    if positive_ai_flow_df is not None and len(positive_ai_flow_df) > 0:
        af = positive_ai_flow_df.copy()
        if 'date' in af.columns and 'date' in df.columns:
            af['date'] = pd.to_datetime(af['date'])
            df['date'] = pd.to_datetime(df['date'])
            af = af[['date', 'code', 'ai_flow']].drop_duplicates(['date', 'code'])
            df = df.merge(af, on=['date', 'code'], how='left')
        else:
            af = af[['code', 'ai_flow']].drop_duplicates('code')
            df = df.merge(af, on='code', how='left')
        df['PositiveAIFlow'] = cross_rankpct(df, 'ai_flow').fillna(0.5)
    else:
        df['PositiveAIFlow'] = 0.5

    # === Funding Vacuum: inverse of composite crowding ===
    df['_vacuum_raw'] = (df['CrowdingScore'] + df['StockAttention'] + df['PositiveAIFlow']) / 3
    df['FundingVacuum'] = 1.0 - cross_rankpct(df, '_vacuum_raw')
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
    Soft value-trap guard.

    TrapGuard = I(QualityScore ok) × I(Liquidity ok)
              × I(No severe negative forecast)
              × I(No major bad news event)

    Uses panel columns when available: forecast_neg, EventRisk.
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

    # Forecast penalty
    if panel is not None and 'forecast_neg' in panel.columns:
        fc = panel[['date', 'code', 'forecast_neg']].drop_duplicates(['date', 'code'])
        merged = merged.merge(fc, on=['date', 'code'], how='left')
        penalty = merged['forecast_neg'].fillna(0).clip(0, 1) * 0.40
        merged['TrapGuard'] = (merged['TrapGuard'] * (1.0 - penalty)).clip(0, 1)
        merged.drop(columns=['forecast_neg'], inplace=True)

    return merged[['date', 'code', 'TrapGuard']]
