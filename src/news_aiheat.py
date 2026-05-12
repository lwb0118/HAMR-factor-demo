"""
news_aiheat.py
==============
Domestic financial news AI intensity — GDELT replacement.

Uses East Money financial news (akshare stock_info_global_em)
to track daily AI/quant keyword frequency.
"""

import akshare as ak
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

DATA_EXTERNAL = Path(__file__).parent.parent / 'data' / 'external'
DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)

AI_KEYWORDS = ['AI', '量化', '人工智能', 'ChatGPT', '算法交易',
               '大模型', 'DeepSeek', '机器学习', '智能投顾',
               '量化交易', '程序化交易', '策略回测']


def fetch_daily_news_intensity():
    """
    Fetch today's financial news and count AI-related articles.

    Returns:
        dict: {'date': 'YYYY-MM-DD', 'total_news': N, 'ai_news': M,
               'intensity': M/N, 'keywords': {...}}
    """
    try:
        df = ak.stock_info_global_em()
        total = len(df)

        kw_counts = {}
        ai_titles = []
        for kw in AI_KEYWORDS:
            matches = df[df['标题'].str.contains(kw, na=False)]
            kw_counts[kw] = len(matches)
            for _, r in matches.iterrows():
                if r['标题'] not in ai_titles:
                    ai_titles.append(r['标题'])

        ai_count = len(ai_titles)
        intensity = ai_count / total if total > 0 else 0

        return {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'total_news': total,
            'ai_news': ai_count,
            'intensity': round(intensity, 4),
            'keywords': kw_counts,
        }
    except Exception as e:
        return {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'total_news': 0,
            'ai_news': 0,
            'intensity': 0,
            'error': str(e),
        }


def build_news_timeseries():
    """
    Build daily AI news intensity time series.

    Loads cached history and appends today's data.
    Returns DataFrame with [date, news_intensity].
    """
    cache_file = DATA_EXTERNAL / 'news_aiheat.json'
    today_str = datetime.now().strftime('%Y-%m-%d')

    # Load existing history
    history = {}
    if cache_file.exists():
        history = json.loads(cache_file.read_text()).get('daily', {})

    # Fetch today's data (if not already cached)
    if today_str not in history:
        result = fetch_daily_news_intensity()
        history[result['date']] = result
        # Save back
        cache_file.write_text(json.dumps({
            'keywords': AI_KEYWORDS,
            'updated': time.time(),
            'daily': history
        }, indent=2, ensure_ascii=False))
        print(f'  AI News: {result["ai_news"]}/{result["total_news"]} '
              f'({result["intensity"]:.1%}) — saved to cache')

    # Build DataFrame
    rows = []
    for date, data in sorted(history.items()):
        rows.append({
            'date': pd.to_datetime(date),
            'news_intensity': data.get('intensity', 0)
        })

    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values('date')

    # FIX: If only sparse data available, use mean intensity as proxy
    # for all historical dates. Mark as proxy in output.
    if len(df) < 30:
        mean_intensity = df['news_intensity'].mean()
        print(f'    News: only {len(df)} days cached, using mean={mean_intensity:.4f} as proxy')
        df['_is_proxy'] = True
    else:
        df['_is_proxy'] = False

    return df
