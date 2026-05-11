"""
ai_heat.py
==========
AIHeat_State — market-level condition variable.

Per HAMR Data Source Guide, AIHeat is NOT a stock-level signal.
It is a time-series state variable that gates HAMR activation.

Formula (from guide):
  AIHeat_raw(t) = 0.30*GitHub_AIHeat + 0.30*Search_AIHeat
                + 0.20*Media/News_AIHeat + 0.10*Community_AIHeat
                + 0.10*Market_Proxy
  AIHeat_MA20(t) = MA(AIHeat_raw, 20)
  AIHeat_Z(t) = rolling_zscore(AIHeat_MA20, 252)
  AIHeat_Change(t) = AIHeat_Z(t) - AIHeat_Z(t-20)
  AIStateScore(t) = RankPct(AIHeat_Z, 252) * RankPct(AIHeat_Change, 252)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from .data_process import cross_rankpct, DATA_EXTERNAL


def rolling_rank_pct(series, window=252, min_periods=20):
    """
    Rolling rank percentile using only historical data up to current date.
    Avoid full-sample future leakage.
    """
    s = pd.Series(series).astype(float)
    out = pd.Series(index=s.index, dtype=float)

    for i in range(len(s)):
        start = max(0, i - window + 1)
        win = s.iloc[start:i + 1].dropna()

        if len(win) < min_periods:
            out.iloc[i] = 0.5
        else:
            out.iloc[i] = win.rank(pct=True).iloc[-1]

    return out.fillna(0.5).clip(0, 1)


def compute_ai_heat_from_panel(
    panel,
    github_data=None,
    github_ts=None,
    news_ts=None,
    community=None,
    search_ts=None,
):
    """
    Compute AIStateScore.

    AIHeat_raw =
      0.30 * GitHub_AIHeat
      + 0.30 * Search_AIHeat
      + 0.20 * Media/News_AIHeat
      + 0.10 * Community_AIHeat
      + 0.10 * Market_Proxy
    """
    daily = panel.groupby("date").agg(
        market_ret=("ret_1d", "mean"),
        market_turnover=("turnover", "mean"),
        ret_dispersion=("ret_1d", "std"),
        top20_volume_share=(
            "dollar_volume",
            lambda x: x.nlargest(max(1, int(len(x) * 0.2))).sum() / x.sum()
            if x.sum() > 0 else 0,
        ),
    ).reset_index()

    daily["date"] = pd.to_datetime(daily["date"])

    for col, name in [
        ("market_turnover", "turnover_z"),
        ("ret_dispersion", "dispersion_z"),
        ("top20_volume_share", "concentration_z"),
    ]:
        daily[name] = daily[col].rolling(60, min_periods=10).apply(
            lambda x: (x.iloc[-1] - x.mean()) / x.std()
            if x.std() > 0 else 0,
            raw=False,
        )

    market_proxy_raw = (
        0.35 * daily["turnover_z"].fillna(0)
        + 0.35 * daily["dispersion_z"].fillna(0)
        + 0.30 * daily["concentration_z"].fillna(0)
    )

    daily["market_proxy"] = rolling_rank_pct(market_proxy_raw)

    # --- GitHub AIHeat component ---
    if github_ts is not None and len(github_ts) > 0:
        gh = github_ts.copy()
        gh["date"] = pd.to_datetime(gh["date"])

        daily = daily.merge(
            gh[["date", "github_activity_score"]],
            on="date",
            how="left",
        )

        daily["github_aiheat"] = (
            daily["github_activity_score"]
            .fillna(0.5)
            .clip(0, 1)
        )

    elif github_data and github_data.get("stars", 0) > 0:
        stars = github_data.get("stars", 0)
        forks = github_data.get("forks", 0)
        daily["github_aiheat"] = min(
            1.0,
            max(0.0, 0.7 * stars / 5000 + 0.3 * forks / 1000),
        )
    else:
        daily["github_aiheat"] = 0.5

    # --- Search AIHeat component ---
    if search_ts is not None and len(search_ts) > 0:
        st = search_ts.copy()
        st["date"] = pd.to_datetime(st["date"])

        daily = daily.merge(
            st[["date", "search_score"]],
            on="date",
            how="left",
        )

        daily["search_aiheat"] = (
            daily["search_score"]
            .fillna(0.5)
            .clip(0, 1)
        )
    else:
        daily["search_aiheat"] = 0.5

    # --- News AIHeat component ---
    if news_ts is not None and len(news_ts) > 0:
        nt = news_ts.copy()
        nt["date"] = pd.to_datetime(nt["date"])

        daily = daily.merge(
            nt[["date", "news_intensity"]],
            on="date",
            how="left",
        )

        daily["news_aiheat"] = (
            daily["news_intensity"]
            .fillna(0.02)
            .clip(0, 1)
        )
    else:
        daily["news_aiheat"] = 0.5

    # --- Community AIHeat component ---
    if community and community.get("activity_score", 0) > 0:
        daily["community_aiheat"] = float(community["activity_score"])
    else:
        daily["community_aiheat"] = 0.5

    # --- Combined AIHeat ---
    daily["AIHeat_raw"] = (
        0.30 * daily["github_aiheat"]
        + 0.30 * daily["search_aiheat"]
        + 0.20 * daily["news_aiheat"]
        + 0.10 * daily["community_aiheat"]
        + 0.10 * daily["market_proxy"]
    )

    daily["AIHeat_MA20"] = (
        daily["AIHeat_raw"]
        .rolling(20, min_periods=5)
        .mean()
    )

    # AIHeat_Z (rolling z-score, no look-ahead)
    ma = daily["AIHeat_MA20"]
    z = []

    for i in range(len(ma)):
        start = max(0, i - 252 + 1)
        win = ma.iloc[start:i + 1].dropna()

        if len(win) < 20 or win.std() == 0:
            z.append(0.0)
        else:
            z.append((ma.iloc[i] - win.mean()) / win.std())

    daily["AIHeat_Z"] = z
    daily["AIHeat_Change"] = daily["AIHeat_Z"] - daily["AIHeat_Z"].shift(20)

    daily["AIHeat_Z_rank"] = rolling_rank_pct(daily["AIHeat_Z"])
    daily["AIHeat_Chg_rank"] = rolling_rank_pct(daily["AIHeat_Change"])

    daily["AIStateScore"] = (
        daily["AIHeat_Z_rank"].fillna(0.5)
        * daily["AIHeat_Chg_rank"].fillna(0.5)
    ).clip(0, 1)

    return daily[
        [
            "date",
            "AIStateScore",
            "AIHeat_raw",
            "AIHeat_Z",
            "AIHeat_Change",
            "github_aiheat",
            "search_aiheat",
            "news_aiheat",
            "community_aiheat",
            "market_proxy",
        ]
    ]


def load_search_aiheat(path=None):
    """
    Load manually prepared search AIHeat.

    Expected columns: date, search_score

    search_score can be Google Trends / Baidu Index / WeChat Index,
    normalized to 0-100 or 0-1.
    """
    if path is None:
        path = DATA_EXTERNAL / "search_aiheat.csv"

    path = Path(path)

    if not path.exists():
        return None

    df = pd.read_csv(path)

    if "date" not in df.columns or "search_score" not in df.columns:
        raise ValueError("search_aiheat.csv must contain date, search_score")

    df["date"] = pd.to_datetime(df["date"])
    df["search_score"] = pd.to_numeric(df["search_score"], errors="coerce")

    if df["search_score"].max() > 1.5:
        df["search_score"] = df["search_score"] / 100.0

    df["search_score"] = df["search_score"].clip(0, 1).fillna(0.5)

    return df[["date", "search_score"]]


def fetch_github_aiheat(repo='brokermr810/QuantDinger', token=None):
    """
    Fetch GitHub activity for AIHeat computation.

    Per guide: Δstars, Δforks, issues, PRs, commits, new_repo_count.
    Requires GitHub token for higher rate limits.
    """
    import requests
    import json
    from datetime import datetime

    headers = {'Accept': 'application/vnd.github+json'}
    if token:
        headers['Authorization'] = f'Bearer {token}'

    cache_file = DATA_EXTERNAL / 'github_aiheat.json'

    try:
        meta = requests.get(
            f'https://api.github.com/repos/{repo}', headers=headers, timeout=15
        ).json()
        stars = meta.get('stargazers_count', 0)
        forks = meta.get('forks_count', 0)
        issues = meta.get('open_issues_count', 0)

        result = {
            'repo': repo,
            'stars': stars,
            'forks': forks,
            'open_issues': issues,
            'fetched_at': datetime.now().isoformat(),
            'ai_heat_raw': min(1.0, stars / 5000),
        }

        with open(cache_file, 'w') as f:
            json.dump(result, f)

        return result
    except Exception:
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        return {'stars': 0, 'forks': 0, 'ai_heat_raw': 0}
