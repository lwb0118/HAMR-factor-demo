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


def fetch_star_history(repo='brokermr810/QuantDinger', token=None, since_date=None):
    """
    Fetch star history (starred_at timestamps) from GitHub API.

    Args:
        repo: 'owner/repo' string
        token: GitHub API token
        since_date: 'YYYY-MM-DD' string, only fetch stars after this date

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

    # Step 2: Page from the END (most recent stars first) for recent data only
    per_page = 100
    total_pages = (total_stars + per_page - 1) // per_page
    
    # Without token: 60 requests/hour limit
    # With token: 5000 requests/hour limit
    has_token = token is not None and token.strip()
    if has_token:
        max_pages = min(200, total_pages)  # with token: up to 200 pages
    else:
        max_pages = min(20, total_pages)   # without token: max 20 pages
    
    # Start from the last page (most recent stars)
    start_page = total_pages
    end_page = max(1, start_page - max_pages + 1)
    pages_to_fetch = list(range(start_page, end_page - 1, -1))  # descending: newest first
    
    since_str = f' since {since_date}' if since_date else ''
    print(f'    ({total_stars} total{since_str}, fetching recent pages {end_page}→{start_page})', end=' ', flush=True)

    for i, page in enumerate(pages_to_fetch):
        if (i + 1) % 10 == 0:
            print(f'{i+1}/{len(pages_to_fetch)}', end=' ', flush=True)
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
                            date_str = starred[:10]
                            # Filter by since_date if provided
                            if since_date and date_str < since_date:
                                # We've gone too far back, stop fetching more pages
                                print(f'\n    Reached {date_str} < {since_date}, stopping.', flush=True)
                                # Sort and cache
                                star_dates.sort()
                                cache_file.write_text(json.dumps({
                                    'repo': repo,
                                    'fetched_at': time.time(),
                                    'star_dates': star_dates,
                                    'total_stars': len(star_dates)
                                }, indent=2))
                                return star_dates
                            star_dates.append(date_str)
                    break
                if resp.status_code == 422:
                    break
            except Exception as e:
                if attempt == 2:
                    print(f'[err:{e}]', end=' ', flush=True)
                    break
                time.sleep(2.0)
        time.sleep(0.3)
    print('Done!', flush=True)

    # Sort by date (oldest first) since we fetched newest first
    star_dates.sort()

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


def build_github_timeseries(repos=None, token=None, since_date='2024-05-01'):
    """
    Build daily GitHub activity time series for AIHeat.

    Tracks star accumulation across multiple repos.
    Add repos to the list as the AI quant ecosystem grows.

    Args:
        repos: list of 'owner/repo' strings. Defaults to QuantDinger.
        since_date: 'YYYY-MM-DD' start date for data
    """
    if repos is None:
        repos = [
            'vnpy/vnpy',                 # China's largest quant framework (2015+)
            'brokermr810/QuantDinger',  # AI quant platform, built 2025-12
        ]

    all_star_dates = []
    for repo in repos:
        print(f'  Fetching {repo}...', end=' ')
        star_dates = fetch_star_history(repo, token, since_date=since_date)
        print(f'{len(star_dates)} stars')
        all_star_dates.extend(star_dates)

    # Build daily counts from since_date
    date_counts = pd.Series(all_star_dates).value_counts().sort_index()
    date_counts.index = pd.to_datetime(date_counts.index)
    
    # Full date range from since_date to today
    start = pd.Timestamp(since_date)
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
