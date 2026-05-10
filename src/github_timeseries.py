"""
github_timeseries.py
=====================
Build GitHub activity time series for AIHeat computation.

Uses GitHub API to track star/fork/issue history over time,
building a daily activity index for AI quant tool adoption.

Per HAMR Data Guide:
  GitHub_AIHeat(t) = RankPct(Δstars + Δforks + issues + commits)
"""

import requests
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

DATA_EXTERNAL = Path(__file__).parent.parent / 'data' / 'external'
DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)


def fetch_star_history(repo='brokermr810/QuantDinger', token=None):
    """
    Fetch star history (starred_at timestamps) from GitHub API.

    Returns list of dates when stars were added.
    """
    headers = {'Accept': 'application/vnd.github.star+json'}
    if token:
        headers['Authorization'] = f'Bearer {token}'

    cache_file = DATA_EXTERNAL / f'stars_{repo.replace("/", "_")}.json'

    # Try cache first (stars only grow, cache is good for a day)
    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        cache_age = time.time() - cached.get('fetched_at', 0)
        if cache_age < 86400:  # 24 hour cache
            return cached.get('star_dates', [])

    star_dates = []

    # Step 1: Get total star count from repo metadata
    total_stars = 0
    try:
        meta = requests.get(f'https://api.github.com/repos/{repo}',
                           headers={'Authorization': f'Bearer {token}'} if token else {},
                           timeout=15).json()
        total_stars = meta.get('stargazers_count', 0)
    except Exception:
        pass

    # Step 2: Page from the END (most recent stars)
    per_page = 100
    total_pages = (total_stars + per_page - 1) // per_page
    max_pages = min(50, total_pages)
    start_page = total_pages  # start from the very last page
    print(f'    ({total_stars} total, fetching pages {start_page-max_pages+1}→{start_page})', end=' ', flush=True)

    for page in range(start_page, start_page - max_pages, -1):
        url = (f'https://api.github.com/repos/{repo}/stargazers'
               f'?per_page={per_page}&page={page}')
        for attempt in range(3):
            try:
                resp = requests.get(url, headers=headers, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    if not data:
                        continue
                    for item in data:
                        starred = item.get('starred_at', '')
                        if starred:
                            star_dates.append(starred[:10])
                    break
                if resp.status_code == 422:
                    break
            except Exception:
                if attempt == 2:
                    break
                time.sleep(2.0)
        time.sleep(0.3)

    # Cache
    cache_file.write_text(json.dumps({
        'repo': repo,
        'fetched_at': time.time(),
        'star_dates': star_dates,
        'total_stars': len(star_dates)
    }, indent=2))

    return star_dates


def fetch_commit_history(repo='brokermr810/QuantDinger', token=None):
    """
    Fetch commit activity (weekly) from GitHub API.

    Returns list of {week_start, total_commits}.
    """
    headers = {'Accept': 'application/vnd.github+json'}
    if token:
        headers['Authorization'] = f'Bearer {token}'

    cache_file = DATA_EXTERNAL / f'commits_{repo.replace("/", "_")}.json'

    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        cache_age = time.time() - cached.get('fetched_at', 0)
        if cache_age < 3600:
            return cached.get('weeks', [])

    try:
        url = f'https://api.github.com/repos/{repo}/stats/commit_activity'
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            weeks = resp.json()
            result = []
            for w in weeks:
                result.append({
                    'week_start': datetime.fromtimestamp(w['week']).strftime('%Y-%m-%d'),
                    'total': w['total']
                })
            cache_file.write_text(json.dumps({
                'repo': repo,
                'fetched_at': time.time(),
                'weeks': result
            }, indent=2))
            return result
    except Exception:
        pass

    if cache_file.exists():
        return json.loads(cache_file.read_text()).get('weeks', [])
    return []


def build_github_timeseries(repos=None, token=None):
    """
    Build daily GitHub activity time series for AIHeat.

    Tracks star accumulation across multiple repos.
    Add repos to the list as the AI quant ecosystem grows.

    Args:
        repos: list of 'owner/repo' strings. Defaults to QuantDinger.
    """
    if repos is None:
        repos = [
            'brokermr810/QuantDinger',  # AI quant platform, built 2025-12
            'vnpy/vnpy',                 # China's largest quant framework
        ]

    all_star_dates = []
    for repo in repos:
        print(f'  Fetching {repo}...', end=' ')
        star_dates = fetch_star_history(repo, token)
        print(f'{len(star_dates)} stars')
        all_star_dates.extend(star_dates)

    # Truncate all repos to same start date (QuantDinger first star)
    # vnpy/qlib pre-2025 history is irrelevant for current AIHeat signal
    cutoff_date = pd.Timestamp('2025-12-30')  # QuantDinger first star
    date_counts = pd.Series(all_star_dates).value_counts().sort_index()
    date_counts.index = pd.to_datetime(date_counts.index)
    date_counts = date_counts[date_counts.index >= cutoff_date]
    if len(date_counts) == 0:
        return None

    # Full date range
    start = date_counts.index.min()
    end = datetime.now()
    all_dates = pd.date_range(start, end, freq='D')
    daily = pd.DataFrame({'date': all_dates})
    daily['stars_daily'] = daily['date'].map(date_counts).fillna(0)
    daily['stars_cum'] = daily['stars_daily'].cumsum()

    # 20-day delta (star acceleration)
    daily['stars_delta_20d'] = daily['stars_cum'] - daily['stars_cum'].shift(20)

    # Commit activity (weekly, spread to daily)
    commit_weeks = fetch_commit_history(repos[0], token)
    if commit_weeks:
        for w in commit_weeks:
            ws = pd.to_datetime(w['week_start'])
            mask = (daily['date'] >= ws) & (daily['date'] < ws + timedelta(days=7))
            daily.loc[mask, 'commits'] = w['total'] / 7  # daily average
        daily['commits'] = daily['commits'].fillna(0)
    else:
        daily['commits'] = 0

    # Activity score: 90-day rolling percentile of 20d delta
    # QuantDinger shows 3 clear phases: low (Jan-Feb), mid (Mar-Apr), high (May+)
    daily['stars_delta_raw'] = daily['stars_daily'].rolling(20, min_periods=5).mean()
    daily['github_activity_score'] = 0.5
    for i in range(len(daily)):
        if i >= 10:
            win = daily['stars_delta_raw'].iloc[max(0, i-90):i+1]
            if len(win) >= 10:
                daily.iloc[i, daily.columns.get_loc('github_activity_score')] = win.rank(pct=True).iloc[-1]
    daily['github_activity_score'] = daily['github_activity_score'].fillna(0.5).clip(0, 1)
    daily['github_activity_score'] = daily['github_activity_score'].fillna(0.5).clip(0, 1)

    return daily[['date', 'stars_daily', 'stars_cum', 'stars_delta_20d',
                  'commits', 'github_activity_score']]
