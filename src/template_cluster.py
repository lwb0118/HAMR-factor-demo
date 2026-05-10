"""
template_cluster.py
==================
TemplateClustering — identify hot template stocks and measure mismatch.

Per HAMR Framework:
  HotSet(t) = Top stocks by ret_20d, turnover_20d, amount_spike, attention
  TemplateAffinity(i,t) = max_k CosineSimilarity(ZFeature(i,t), Centroid_k(t))
  MismatchScore(i,t) = 1 - RankPct(TemplateAffinity(i,t))

Higher MismatchScore → stock deviates MORE from current hot templates.

Proxy implementation: identifies "hot template" characteristics
via momentum + turnover + volume spike without full clustering.
"""

import numpy as np
import pandas as pd
from .data_process import cross_rankpct


def compute_template_affinity(panel, min_industry_stocks=5):
    """
    Compute how closely each stock matches current hot template characteristics.

    Industry-neutral version:
      - TemplateAffinity: global cross-sectional rank (hot = high mom+turnover)
      - MismatchScore: 1 - RankPct(TemplateAffinity) WITHIN same industry×date
        Falls back to global if industry has < min_industry_stocks.

    This prevents systematic industry bias (e.g. finance always looks 'mismatched'
    due to naturally lower turnover).
    """
    df = panel.copy()

    # Hot template dimensions (global)
    df['hot_mom'] = cross_rankpct(df, 'ret_20d')
    df['hot_turnover'] = cross_rankpct(df, 'turnover_avg_20d')

    if 'turnover_spike' in df.columns:
        df['hot_spike'] = cross_rankpct(df, 'turnover_spike')
        df['TemplateAffinity'] = (
            0.4 * df['hot_mom'].fillna(0.5) +
            0.3 * df['hot_turnover'].fillna(0.5) +
            0.3 * df['hot_spike'].fillna(0.5)
        )
    else:
        df['TemplateAffinity'] = (
            0.5 * df['hot_mom'].fillna(0.5) +
            0.5 * df['hot_turnover'].fillna(0.5)
        )

    # --- Industry-neutral MismatchScore ---
    if 'industry' in df.columns:
        # Rank within industry×date, with size check
        def _industry_rankpct(grp):
            if len(grp) >= min_industry_stocks:
                return 1.0 - grp['TemplateAffinity'].rank(pct=True).fillna(0.5)
            else:
                # Fall back to global within this date
                return grp['_global_mismatch']

        # Pre-compute global mismatch for fallback
        df['_global_mismatch'] = 1.0 - df.groupby('date')['TemplateAffinity'].transform(
            lambda x: x.rank(pct=True)
        )

        df['MismatchScore'] = df.groupby(
            ['date', 'industry'], group_keys=False
        ).apply(_industry_rankpct)

        # Clean up
        df['MismatchScore'] = df['MismatchScore'].fillna(df['_global_mismatch'])
        df.drop(columns=['_global_mismatch'], inplace=True)
    else:
        df['MismatchScore'] = 1.0 - cross_rankpct(df, 'TemplateAffinity')

    df['MismatchScore'] = df['MismatchScore'].clip(0, 1).fillna(0.5)

    return df[['date', 'code', 'TemplateAffinity', 'MismatchScore']]


def identify_hot_template_stocks(panel, top_pct=0.15):
    """
    Identify the set of stocks that define current hot template.

    Per guide: Top by ret_20d, turnover_20d, amount_spike, attention.

    Returns:
        list of stock codes considered "hot template" stocks.
    """
    affinity = compute_template_affinity(panel)
    # Take top stocks by TemplateAffinity across dates
    hot_stocks = (
        affinity.groupby('code')['TemplateAffinity']
        .mean()
        .nlargest(int(affinity['code'].nunique() * top_pct))
        .index.tolist()
    )
    return hot_stocks


def compute_template_features(panel):
    """
    Extract multi-dimensional features for template clustering.

    Production version: cluster on [market_cap, price_level,
    volatility, momentum, reversal, turnover, amount_spike].

    Returns template feature vector per stock per date.
    """
    df = panel.copy()

    features = []

    # Feature 1: Momentum rank
    if 'ret_20d' in df.columns:
        features.append(cross_rankpct(df, 'ret_20d').rename('feat_momentum'))

    # Feature 2: Volatility rank (inverse)
    if 'volatility_20d' in df.columns:
        features.append((1 - cross_rankpct(df, 'volatility_20d')).rename('feat_stability'))

    # Feature 3: Turnover rank
    if 'turnover_avg_20d' in df.columns:
        features.append(cross_rankpct(df, 'turnover_avg_20d').rename('feat_turnover'))

    # Feature 4: Liquidity rank
    if 'dollar_volume' in df.columns:
        features.append(cross_rankpct(df, 'dollar_volume').rename('feat_liquidity'))

    if not features:
        return df

    return pd.concat([df[['date', 'code']]] + features, axis=1)
