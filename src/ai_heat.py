"""
ai_heat.py
==========
AIHeat_State — market-level condition variable.

Per HAMR Data Source Guide, AIHeat is NOT a stock-level signal.
It is a time-series state variable that gates HAMR activation.

Formula (from guide):
  AIHeat_raw(t) = 0.35*GitHub_AIHeat + 0.35*Search_AIHeat
                + 0.20*Media_AIHeat  + 0.10*Community_AIHeat
  AIHeat_MA20(t) = MA(AIHeat_raw, 20)
  AIHeat_Z(t) = rolling_zscore(AIHeat_MA20, 252)
  AIHeat_Change(t) = AIHeat_Z(t) - AIHeat_Z(t-20)
  AIStateScore(t) = RankPct(AIHeat_Z, 252) * RankPct(AIHeat_Change, 252)

Proxy implementation (no external API tokens):
  Uses market turnover + dispersion + concentration as proxy.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from .data_process import cross_rankpct, DATA_EXTERNAL


def compute_ai_heat_from_panel(panel, github_data=None, github_ts=None,
                               news_ts=None, community=None, search_ts=None):
    """
    Compute AIStateScore from market data + GitHub + News + Community.

    AIHeat_raw = 0.35 * GitHub + 0.20 * News + 0.10 * Community + 0.35 * Market_Proxy
    """
    daily = panel.groupby('date').agg(
        market_ret=('ret_1d', 'mean'),
        market_turnover=('turnover', lambda x: x.mean() if 'turnover' in
                         panel.columns else 0),
        ret_dispersion=('ret_1d', 'std'),
        top20_volume_share=('dollar_volume', lambda x:
            x.nlargest(max(1, int(len(x)*0.2))).sum() / x.sum()
            if len(x) > 0 else 0)
    ).reset_index()

    # --- Market proxy components ---
    for col, name in [('market_turnover', 'turnover_z'),
                       ('ret_dispersion', 'dispersion_z'),
                       ('top20_volume_share', 'concentration_z')]:
        daily[name] = daily[col].rolling(60, min_periods=10).apply(
            lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0
        )

    market_proxy = (
        0.35 * daily['turnover_z'].fillna(0) +
        0.35 * daily['dispersion_z'].fillna(0) +
        0.30 * daily['concentration_z'].fillna(0)
    )

    # --- GitHub AIHeat component ---
    if github_ts is not None and len(github_ts) > 10:
        # Merge time series by date
        daily['date_d'] = pd.to_datetime(daily['date']).dt.date
        gh = github_ts.copy()
        gh['date_d'] = pd.to_datetime(gh['date']).dt.date
        daily = daily.merge(
            gh[['date_d', 'github_activity_score']],
            on='date_d', how='left'
        )
        daily['github_aiheat'] = daily['github_activity_score'].fillna(0.5).clip(0, 1)
    elif github_data and github_data.get('stars', 0) > 0:
        # Flat snapshot fallback
        stars = github_data['stars']
        forks = github_data.get('forks', 0)
        gh_level = min(1.0, max(0.0, (stars - 2000) / 3000))
        gh_activity = 0.7 * gh_level + 0.3 * min(1.0, forks / 1000)
        daily['github_aiheat'] = gh_activity
    else:
        daily['github_aiheat'] = 0.5

    # --- News AIHeat component ---
    if news_ts is not None and len(news_ts) > 0:
        daily['date_d'] = pd.to_datetime(daily['date']).dt.date
        nt = news_ts.copy()
        nt['date_d'] = pd.to_datetime(nt['date']).dt.date
        daily = daily.merge(nt[['date_d', 'news_intensity']], on='date_d', how='left')
        # Normalize: typical range 0-10% → scale to 0-1
        daily['news_aiheat'] = (daily['news_intensity'].fillna(0.02) / 0.10).clip(0, 1)
    else:
        daily['news_aiheat'] = 0.02  # low baseline

    # --- Community AIHeat component ---
    if community and community.get('activity_score', 0) > 0:
        daily['community_aiheat'] = community['activity_score']
    else:
        daily['community_aiheat'] = 0.5

    # --- Search AIHeat component ---
    if search_ts is not None and len(search_ts) > 0:
        daily['date_d'] = pd.to_datetime(daily['date']).dt.date
        st = search_ts.copy()
        st['date_d'] = pd.to_datetime(st['date']).dt.date
        daily = daily.merge(st[['date_d', 'search_score']], on='date_d', how='left')
        daily['search_aiheat'] = daily['search_score'].fillna(0).clip(-2, 2)
        daily['search_aiheat'] = (daily['search_aiheat'] + 2) / 4  # [-2,2]→[0,1]
    else:
        daily['search_aiheat'] = 0.3  # low baseline

    # --- Combined AIHeat ---
    daily['AIHeat_raw'] = (
        0.25 * daily['github_aiheat'] +
        0.15 * daily['news_aiheat'] +
        0.10 * daily['search_aiheat'] +
        0.10 * daily['community_aiheat'] +
        0.40 * market_proxy
    )

    # AIHeat_MA20
    daily['AIHeat_MA20'] = daily['AIHeat_raw'].rolling(20, min_periods=5).mean()

    # AIHeat_Z (rolling z-score, 252 window)
    ma_series = daily['AIHeat_MA20'].values
    n = len(ma_series)
    daily['AIHeat_Z'] = 0.0
    for i in range(n):
        start = max(0, i - 252)
        win = ma_series[start:i+1]
        if len(win) >= 10 and win.std() > 0:
            daily.loc[i, 'AIHeat_Z'] = (win[-1] - win.mean()) / win.std()

    # AIHeat_Change
    daily['AIHeat_Change'] = daily['AIHeat_Z'] - daily['AIHeat_Z'].shift(20)

    # AIHeat_Z_rank (rolling, no look-ahead)
    daily['AIHeat_Z_rank'] = _rolling_rank_pct(daily['AIHeat_Z'], 60)
    daily['AIHeat_Chg_rank'] = _rolling_rank_pct(daily['AIHeat_Change'], 60)
    daily['AIStateScore'] = (
        daily['AIHeat_Z_rank'].fillna(0.5) *
        daily['AIHeat_Chg_rank'].fillna(0.5)
    ).clip(0, 1)

    return daily[['date', 'AIStateScore', 'AIHeat_raw', 'AIHeat_Z',
                  'AIHeat_Change']]


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
            'date': datetime.now().isoformat()[:10],
            'stars': stars, 'forks': forks, 'open_issues': issues,
            'repo': repo
        }
        # Cache
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(result, indent=2))
        return result
    except Exception as e:
        if cache_file.exists():
            return json.loads(cache_file.read_text())
        return {'stars': 0, 'forks': 0, 'open_issues': 0}


def fetch_gdelt_aiheat(keywords=None, token=None):
    """
    Fetch GDELT news counts for AIHeat computation.

    Per guide: query for AI trading / ChatGPT trading / LLM trading.
    GDELT API v2 is free (no token required for basic queries).
    """
    import requests
    import json
    from urllib.parse import quote

    if keywords is None:
        keywords = ('AI trading OR AI stock picking OR ChatGPT trading OR LLM trading '
                    'OR algorithmic trading OR quant trading')

    cache_file = DATA_EXTERNAL / 'gdelt_aiheat.json'

    try:
        query = quote(keywords)
        url = (
            f'https://api.gdeltproject.org/api/v2/doc/doc?query={query}'
            f'&mode=timelinevolraw&format=json'
            f'&startdatetime=20250101000000&enddatetime=20251231235959'
        )
        resp = requests.get(url, timeout=30).json()
        # Aggregate daily
        daily_counts = {}
        for item in resp.get('timeline', []):
            d = item.get('date', '')[:8]
            c = item.get('value', 0)
            daily_counts[d] = daily_counts.get(d, 0) + c

        result = {'counts': daily_counts, 'query': keywords}
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(result, indent=2))
        return result
    except Exception:
        if cache_file.exists():
            return json.loads(cache_file.read_text())
        return {'counts': {}, 'query': keywords}


def _rolling_rank_pct(series, window=60):
    """
    Rolling rank percentile — no look-ahead.

    Each value ranked within its trailing window.
    Avoids full-sample rank which leaks future information.
    """
    result = pd.Series(0.5, index=series.index)
    vals = series.values
    for i in range(len(vals)):
        start = max(0, i - window + 1)
        win = vals[start:i+1]
        if len(win) >= 10:
            result.iloc[i] = np.mean(win < vals[i])
    return result.clip(0, 1)


def load_search_aiheat(csv_path=None):
    """
    Load search AIHeat from CSV file.

    Expected CSV format: date,keyword,source,value
    where source in {google_trends, baidu_index, wechat_index}
    and value is relative search intensity.

    If no CSV provided, returns None (search component disabled).
    """
    if csv_path is None:
        csv_path = DATA_EXTERNAL / 'search_aiheat.csv'

    csv_path = Path(csv_path) if not isinstance(csv_path, Path) else csv_path
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = df.pivot_table(index='date', columns=['source', 'keyword'],
                         values='value', aggfunc='mean')
    # Average across sources and keywords
    df['search_score'] = df.mean(axis=1)
    df['search_score'] = (df['search_score'] - df['search_score'].mean()) / \
                          (df['search_score'].std() + 1e-10)
    return df.reset_index()[['date', 'search_score']]
