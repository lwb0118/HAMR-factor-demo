"""
template_cluster.py
===================
TemplateClustering — KMeans-based hot template identification.

Per HAMR Framework v2.0:
  1. Each trading day t, identify HotSet from past 20 days
  2. Cluster HotSet via KMeans into 1-3 template centroids
  3. TemplateAffinity = max_k cosine_similarity(stock_feature, centroid_k)
  4. MismatchScore = 1 - date/industry RankPct(TemplateAffinity)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from .data_process import cross_rankpct


def compute_template_affinity(
    panel,
    n_clusters=3,
    min_industry_stocks=5,
    recent_only=False,
    recent_days=20,
    min_stocks_per_day=20,
):
    """KMeans-based hot-template clustering.

    recent_only=True  : quick demo, only recent_days.
    recent_only=False : full historical academic backtest.
    """
    df = panel.copy()
    dates = sorted(df['date'].unique())

    # Feature columns for clustering
    feat_cols = ['ret_20d', 'turnover_avg_20d', 'volatility_20d']
    if 'turnover_spike' in df.columns:
        feat_cols.append('turnover_spike')
    if 'dollar_volume' in df.columns:
        feat_cols.append('dollar_volume')

    # Ensure all features exist (fill missing)
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.5

    df['TemplateAffinity'] = 0.5  # default
    df['n_templates'] = 0

    for date in dates[-20:]:  # Only cluster recent dates (computational)
        mask = df['date'] == date
        day_data = df.loc[mask].copy()
        n_stocks = len(day_data)
        if n_stocks < 20:
            continue

        # ---- Step 1: Identify HotSet (top 20% by composite score) ----
        day_data['_hot_score'] = (
            0.4 * day_data['ret_20d'].rank(pct=True).fillna(0.5) +
            0.3 * day_data['turnover_avg_20d'].rank(pct=True).fillna(0.5) +
            0.3 * (day_data.get('turnover_spike', day_data['ret_20d']))
                  .rank(pct=True).fillna(0.5)
        )
        hot_n = max(10, int(n_stocks * 0.20))
        hot_idx = day_data['_hot_score'].nlargest(hot_n).index

        # ---- Step 2: Extract features for HotSet ----
        features = day_data.loc[hot_idx, feat_cols].fillna(0).values
        if len(features) < 3:
            continue

        # Standardize
        scaler = StandardScaler()
        try:
            features_scaled = scaler.fit_transform(features)
        except Exception:
            continue

        # ---- Step 3: KMeans clustering ----
        k = min(n_clusters, len(features) // 3)
        if k < 1:
            k = 1
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
            km.fit(features_scaled)
            centroids = km.cluster_centers_
        except Exception:
            continue

        # ---- Step 4: Cosine similarity for ALL stocks ----
        all_features = day_data[feat_cols].fillna(0).values
        try:
            all_scaled = scaler.transform(all_features)
        except Exception:
            continue

        sim = cosine_similarity(all_scaled, centroids)  # [n_stocks, k]
        affinity = sim.max(axis=1)  # max across clusters

        df.loc[mask, 'TemplateAffinity'] = affinity
        df.loc[mask, 'n_templates'] = k

    # ---- Step 5: MismatchScore (industry-neutral inverse) ----
    if 'industry' in df.columns:
        def _industry_rankpct(grp):
            if len(grp) >= min_industry_stocks:
                return 1.0 - grp['TemplateAffinity'].rank(pct=True).fillna(0.5)
            else:
                return grp['_global_mismatch']
        df['_global_mismatch'] = 1.0 - df.groupby('date')['TemplateAffinity'].transform(
            lambda x: x.rank(pct=True))
        df['MismatchScore'] = df.groupby(['date', 'industry'], group_keys=False).apply(
            _industry_rankpct)
        df['MismatchScore'] = df['MismatchScore'].fillna(df['_global_mismatch']).clip(0, 1)
        df.drop(columns=['_global_mismatch'], inplace=True)
    else:
        df['MismatchScore'] = 1.0 - df.groupby('date')['TemplateAffinity'].transform(
            lambda x: x.rank(pct=True))

    df['MismatchScore'] = df['MismatchScore'].fillna(0.5).clip(0, 1)

    return df[['date', 'code', 'TemplateAffinity', 'MismatchScore']]


def compute_template_affinity_proxy(panel, min_industry_stocks=5):
    """
    Lightweight proxy version (no sklearn dependency).
    Uses momentum + turnover + volume spike weighted score.
    Falls back to this when KMeans fails or sklearn unavailable.
    """
    df = panel.copy()
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

    if 'industry' in df.columns:
        df['_global_mismatch'] = 1.0 - df.groupby('date')['TemplateAffinity'].transform(
            lambda x: x.rank(pct=True))
        df['MismatchScore'] = df.groupby(['date', 'industry'], group_keys=False).apply(
            lambda grp: (1.0 - grp['TemplateAffinity'].rank(pct=True).fillna(0.5))
            if len(grp) >= min_industry_stocks else grp['_global_mismatch']
        )
        df['MismatchScore'] = df['MismatchScore'].fillna(df['_global_mismatch'])
        df.drop(columns=['_global_mismatch'], inplace=True)
    else:
        df['MismatchScore'] = 1.0 - cross_rankpct(df, 'TemplateAffinity')

    df['MismatchScore'] = df['MismatchScore'].fillna(0.5).clip(0, 1)
    return df[['date', 'code', 'TemplateAffinity', 'MismatchScore']]
