"""
template_cluster.py
===================
TemplateClustering — KMeans-based hot template identification.

Per HAMR Framework v2.0:
  1. Each trading day t, identify HotSet from past lookback_days using ONLY data up to t.
  2. Cluster HotSet via KMeans into 1-3 template centroids.
  3. TemplateAffinity = max_k cosine_similarity(stock_feature, centroid_k).
  4. MismatchScore = 1 - date/industry RankPct(TemplateAffinity).
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
    lookback_days=20,
    min_stocks_per_day=20,
):
    """
    Full-history KMeans hot-template clustering.

    For each trading day t:
    1. Use only information up to t (no look-ahead).
    2. Build HotSet from past lookback_days.
    3. Cluster HotSet into 1-3 hot templates.
    4. Compute TemplateAffinity for all stocks on t.
    5. MismatchScore = 1 - date/industry RankPct(TemplateAffinity).
    """
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])

    dates = sorted(df["date"].unique())

    feat_cols = ["ret_20d", "turnover_avg_20d", "volatility_20d"]

    if "turnover_spike" in df.columns:
        feat_cols.append("turnover_spike")

    if "dollar_volume" in df.columns:
        feat_cols.append("dollar_volume")

    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0

    df["TemplateAffinity"] = 0.5
    df["n_templates"] = 0

    if recent_only:
        iter_dates = dates[-recent_days:]
    else:
        iter_dates = dates

    date_to_pos = {d: i for i, d in enumerate(dates)}

    for date in iter_dates:
        pos = date_to_pos[date]

        hist_start = max(0, pos - lookback_days + 1)
        hist_dates = dates[hist_start:pos + 1]

        hist = df[df["date"].isin(hist_dates)].copy()
        today_mask = df["date"] == date
        today = df.loc[today_mask].copy()

        if len(today) < min_stocks_per_day:
            continue

        if len(hist) < min_stocks_per_day * 5:
            continue

        hist["_hot_score"] = (
            0.40 * hist["ret_20d"].rank(pct=True).fillna(0.5)
            + 0.30 * hist["turnover_avg_20d"].rank(pct=True).fillna(0.5)
            + 0.30 * hist.get("turnover_spike", hist["ret_20d"])
            .rank(pct=True)
            .fillna(0.5)
        )

        hot_n = max(30, int(len(hist) * 0.20))
        hot = hist.nlargest(hot_n, "_hot_score").copy()

        if len(hot) < 10:
            continue

        hot_features = hot[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        today_features = (
            today[feat_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

        # Fit scaler on FULL market (all stocks today), not just hot set.
        # Fix: hot-set-only scaler caused extreme out-of-distribution scaling
        # for non-hot stocks, corrupting cosine similarity and MismatchScore.
        scaler = StandardScaler()

        try:
            scaler.fit(today_features.values)
            hot_scaled = scaler.transform(hot_features.values)
            today_scaled = scaler.transform(today_features.values)
        except Exception:
            continue

        k = min(n_clusters, max(1, len(hot_scaled) // 20))

        try:
            km = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10,
                max_iter=200,
            )
            km.fit(hot_scaled)
            centroids = km.cluster_centers_
        except Exception:
            continue

        try:
            sim = cosine_similarity(today_scaled, centroids)
            affinity = sim.max(axis=1)
        except Exception:
            continue

        df.loc[today_mask, "TemplateAffinity"] = affinity
        df.loc[today_mask, "n_templates"] = k

    df["_global_mismatch"] = (
        1.0
        - df.groupby("date")["TemplateAffinity"]
        .transform(lambda x: x.rank(pct=True))
    )

    if "industry" in df.columns:
        def _industry_mismatch(g):
            if len(g) >= min_industry_stocks:
                return 1.0 - g["TemplateAffinity"].rank(pct=True).fillna(0.5)
            return g["_global_mismatch"]

        df["MismatchScore"] = (
            df.groupby(["date", "industry"], group_keys=False)
            .apply(_industry_mismatch)
        )

        df["MismatchScore"] = (
            df["MismatchScore"]
            .fillna(df["_global_mismatch"])
            .clip(0, 1)
        )
    else:
        df["MismatchScore"] = df["_global_mismatch"].fillna(0.5).clip(0, 1)

    return df[["date", "code", "TemplateAffinity", "MismatchScore", "n_templates"]]


