"""
guba_data.py
=============
East Money Guba (东方财富股吧) stock discussion data.

Extracts post titles per stock to compute:
  - StockAttention: discussion intensity at stock level
  - Community_AIHeat: aggregate discussion activity (market-wide)

Data Source: guba.eastmoney.com (public HTML, no API key needed)
"""

import requests
import re
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

DATA_EXTERNAL = Path(__file__).parent.parent / 'data' / 'external'
DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Referer': 'https://guba.eastmoney.com/',
}


def fetch_stock_posts(stock_code, page=1, max_retries=2):
    """
    Fetch post titles from East Money Guba for a single stock.

    Returns:
        list of dict: [{'title': str, 'timestamp': str}, ...]
    """
    url = f'https://guba.eastmoney.com/list,{stock_code}_{page}.html'
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=12)
            if resp.status_code != 200:
                if attempt < max_retries:
                    time.sleep(1.0)
                    continue
                return []

            # Extract article_list from JavaScript
            match = re.search(r'var\s+article_list\s*=\s*(\{.*?\});', resp.text, re.DOTALL)
            if not match:
                return []

            data = json.loads(match.group(1))
            posts = []
            for p in data.get('re', []):
                title = p.get('post_title', '').strip()
                ts = p.get('post_last_time', '')
                if title and len(title) > 5:
                    posts.append({'title': title, 'timestamp': ts[:10] if ts else ''})
            return posts
        except Exception:
            if attempt < max_retries:
                time.sleep(1.5)
            else:
                return []


def fetch_universe_posts(codes, max_per_stock=80, max_workers=4):
    """
    Fetch latest page of posts for a list of stocks.

    Returns:
        dict: stock_code -> {'total_posts': N, 'latest_titles': [...],
                              'latest_timestamps': [...]}
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cache_file = DATA_EXTERNAL / 'guba_posts.json'

    # Use cache if recent (< 1 hour)
    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        if time.time() - cached.get('fetched_at', 0) < 3600:
            print(f'    Loaded {len(cached.get("posts",{}))} stocks from cache')
            return cached.get('posts', {})

    results = {}

    def fetch_one(code):
        posts = fetch_stock_posts(code, page=1)
        return code, [p['title'] for p in posts], [p['timestamp'] for p in posts]

    done = 0
    total = len(codes)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_one, c): c for c in codes}
        for future in as_completed(futures):
            done += 1
            code, titles, timestamps = future.result()
            results[code] = {
                'total_posts': len(titles),
                'latest_titles': titles,
                'latest_timestamps': timestamps,
            }
            if done % 50 == 0:
                print(f'    [{done}/{total}] stocks...')
            time.sleep(0.1)

    # Cache
    cache_file.write_text(json.dumps({
        'fetched_at': time.time(),
        'posts': results,
        'total_stocks': len(results),
    }, ensure_ascii=False), encoding='utf-8')

    return results


def compute_stock_attention(panel, guba_posts, trade_date=None):
    """
    Compute StockAttention score per stock per date.

    Args:
        trade_date: reference date (default: today). Used to compute recency.

    Returns DataFrame with [code, stock_attention_raw, guba_posts].
    """
    from datetime import datetime, timedelta
    if trade_date is None:
        trade_date = datetime.now().strftime('%Y-%m-%d')
    cutoff = (datetime.strptime(trade_date, '%Y-%m-%d') - timedelta(days=window_days)).strftime('%Y-%m-%d')
    rows = []
    for code, data in guba_posts.items():
        n_posts = data.get('total_posts', 0)
        if n_posts == 0:
            continue

        # Compute recency: % of posts from last 3 days
        timestamps = data.get('latest_timestamps', [])
        recent = sum(1 for t in timestamps if t >= cutoff)
        recency = recent / n_posts if n_posts > 0 else 0

        # Attention score = post volume normalized * recency
        rows.append({
            'code': code,
            'guba_posts': n_posts,
            'guba_recency': recency,
            'stock_attention_raw': n_posts * (0.5 + 0.5 * recency),
        })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    return df


def compute_community_aiheat(guba_posts):
    """
    Compute aggregate community (股吧) activity level.

    High aggregate post volume = retail investors are actively discussing stocks
    → high community attention regime.

    Returns:
        float: community activity score [0, 1]
    """
    if not guba_posts:
        return None

    total_posts = sum(d.get('total_posts', 0) for d in guba_posts.values())
    n_stocks = len(guba_posts)
    avg_posts = total_posts / n_stocks if n_stocks > 0 else 0

    # Normalize: 50 avg posts = 0.5, 100 = 1.0 (saturates)
    score = min(1.0, avg_posts / 100)

    return {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'total_posts': total_posts,
        'n_stocks': n_stocks,
        'avg_posts': avg_posts,
        'activity_score': score,
    }
