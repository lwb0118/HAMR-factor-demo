"""
data_filters.py
===============
Stock universe filters — ST, suspend, limit-up/down, IPO age, industry.

Per HAMR Data Source Guide section 8.1:
  - Remove ST, *ST, delisted, long-suspended stocks
  - Remove stocks < 120 trading days since IPO
  - Remove stocks at limit-up/down on rebalance days
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import time


def build_stock_info(pro):
    """
    Fetch stock basic info: name, industry, IPO date, listing status.

    Returns DataFrame with [code, name, industry, list_date, is_st].
    """
    df = pro.stock_basic(exchange='', list_status='L',
                         fields='ts_code,name,industry,list_date')
    df['code'] = df['ts_code'].str[:6]
    df['list_date'] = pd.to_datetime(df['list_date'], format='%Y%m%d')
    df['is_st'] = df['name'].str.contains(r'\*ST|ST', regex=True).astype(int)
    df['industry'] = df['industry'].fillna('Other')
    return df[['code', 'name', 'industry', 'list_date', 'is_st']]


def build_st_status_history(pro, codes):
    """
    Check historical ST status for each stock via namechange.
    """
    ts_codes = [f'{c}.SH' if c.startswith('6') else f'{c}.SZ' for c in codes]
    all_records = []
    for tc in ts_codes:
        try:
            df = pro.namechange(ts_code=tc)
            if df is not None and len(df) > 0:
                df['code'] = tc[:6]
                all_records.append(df)
        except Exception:
            pass
        time.sleep(0.25)
    if not all_records:
        return pd.DataFrame(columns=['code', 'start_date', 'name'])
    return pd.concat(all_records, ignore_index=True)


def fetch_suspend_dates(pro, start_date, end_date):
    """
    Fetch all suspension records in date range.
    Returns set of (code, date) tuples that are suspended.
    """
    try:
        df = pro.suspend_d(
            suspend_type='S', start_date=start_date, end_date=end_date,
            fields='ts_code,suspend_date,resume_date'
        )
        if df is None or len(df) == 0:
            return set()
        df['code'] = df['ts_code'].str[:6]
        df['suspend_date'] = pd.to_datetime(df['suspend_date'])
        df['resume_date'] = pd.to_datetime(df['resume_date'])

        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        suspended = set()
        for _, row in df.iterrows():
            for d in dates:
                if row['suspend_date'] <= d <= row['resume_date']:
                    suspended.add((row['code'], str(d.date())[:10]))
        return suspended
    except Exception:
        return set()


def fetch_limit_prices(pro, trade_date):
    """
    Fetch limit-up/down prices for a trading date.

    Returns DataFrame with [code, up_limit, down_limit, pre_close].
    """
    try:
        df = pro.stk_limit(trade_date=trade_date,
                           fields='ts_code,trade_date,up_limit,down_limit,pre_close')
        if df is None or len(df) == 0:
            return None
        df['code'] = df['ts_code'].str[:6]
        return df
    except Exception:
        return None


def filter_panel(panel, stock_info, suspended_set=None, limit_data=None):
    """
    Apply all trading constraint filters to the panel.

    Filters:
      1. Remove ST stocks
      2. Remove stocks with IPO < 120 trading days
      3. Remove suspended stocks on each date
      4. Mark (not remove) limit-up/down stocks as untradable

    Returns filtered panel with added columns:
      [tradable, is_limit_hit]
    """
    df = panel.copy()

    # 1. Remove ST stocks
    if 'is_st' in stock_info.columns:
        st_codes = stock_info[stock_info['is_st'] == 1]['code'].tolist()
        n_before = df['code'].nunique()
        df = df[~df['code'].isin(st_codes)]
        n_removed = n_before - df['code'].nunique()
        if n_removed > 0:
            print(f'    Removed {n_removed} ST stocks')

    # 2. Filter IPO age (120 trading days)
    if 'list_date' in stock_info.columns:
        df['date_str'] = df['date'].dt.strftime('%Y%m%d')
        df['date_dt'] = pd.to_datetime(df['date_str'])
        merged = df.merge(stock_info[['code', 'list_date']], on='code', how='left')
        # Calendar days threshold ≈ 180 calendar days = 120 trading days
        merged['days_since_ipo'] = (merged['date_dt'] - merged['list_date']).dt.days
        n_before = merged['code'].nunique()
        merged = merged[merged['days_since_ipo'] >= 180]
        n_removed = n_before - merged['code'].nunique()
        if n_removed > 0:
            print(f'    Removed {n_removed} stocks with IPO < 180 calendar days')
        df = merged.drop(columns=['date_str', 'date_dt', 'list_date', 'days_since_ipo'],
                         errors='ignore')

    # 3. Mark tradable status
    df['tradable'] = 1

    # 4. Remove suspended stocks
    if suspended_set and len(suspended_set) > 0:
        df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
        df['is_suspended'] = df.apply(
            lambda r: (r['code'], r['date_str']) in suspended_set, axis=1
        ).astype(int)
        n_suspended = df['is_suspended'].sum()
        df.loc[df['is_suspended'] == 1, 'tradable'] = 0
        df = df.drop(columns=['date_str', 'is_suspended'], errors='ignore')
        if n_suspended > 0:
            print(f'    {n_suspended} suspended obs marked untradable')

    # 5. Mark limit-up/down
    if limit_data is not None and len(limit_data) > 0:
        df = df.merge(limit_data[['code', 'up_limit', 'down_limit', 'pre_close']],
                      on='code', how='left')
        df['is_limit_up'] = (df['close'] >= df['up_limit'] * 0.995).astype(int)
        df['is_limit_down'] = (df['close'] <= df['down_limit'] * 1.005).astype(int)
        df['is_limit_hit'] = (df['is_limit_up'] | df['is_limit_down']).astype(int)
        df.loc[df['is_limit_hit'] == 1, 'tradable'] = 0
        n_limit = df['is_limit_hit'].sum()
        df = df.drop(columns=['up_limit', 'down_limit', 'pre_close'], errors='ignore')
        if n_limit > 0:
            print(f'    {n_limit} limit-hit obs marked untradable')

    df['tradable'] = df['tradable'].fillna(1).astype(int)

    n_tradable = df['tradable'].sum()
    n_total = len(df)
    print(f'    Tradable: {n_tradable}/{n_total} ({n_tradable/n_total:.1%})')

    # --- Generate data_audit.csv ---
    _generate_audit_report(df, ROOT / 'results' / 'tables')

    return df


def _generate_audit_report(panel, output_dir):
    """Generate per-date data quality audit."""
    import os
    output_dir = Path(output_dir) if not isinstance(output_dir, Path) else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    daily = panel.groupby('date').agg(
        n_stocks=('code', 'nunique'),
        n_tradable=('tradable', 'sum'),
        n_limit_hit=('is_limit_hit', 'sum') if 'is_limit_hit' in panel.columns else ('tradable', lambda x: (x == 0).sum()),
    ).reset_index()

    daily['pct_tradable'] = (daily['n_tradable'] / daily['n_stocks'] * 100).round(1)
    daily.to_csv(output_dir / 'data_audit.csv', index=False)
    print(f'    data_audit: {output_dir / "data_audit.csv"} ({len(daily)} dates)')


def add_industry_to_panel(panel, stock_info):
    """Add industry column to panel for sector-neutral calculations."""
    if 'industry' not in stock_info.columns:
        return panel
    return panel.merge(stock_info[['code', 'industry']], on='code', how='left')
