"""
data_process.py
===============
HAMR Unified Data Layer — Tushare primary, akshare fallback.

Normalisation:
  - All variables → RankPct ∈ [0, 1] cross-sectionally per date
  - "Higher = stronger signal" for every variable
  - Financial data strictly PIT-aligned by disclosure date (ann_date)

Data directory (per HAMR Data Source Guide):
  data/raw/         — original API responses
  data/processed/    — cleaned standard tables (date × code panels)
  data/features/     — factor intermediate variables
  data/external/     — GitHub, GDELT, Baidu Index cache
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import warnings

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / 'data' / 'raw'
DATA_PROCESSED = ROOT / 'data' / 'processed'
DATA_FEATURES = ROOT / 'data' / 'features'
DATA_EXTERNAL = ROOT / 'data' / 'external'

for d in [DATA_RAW, DATA_PROCESSED, DATA_FEATURES, DATA_EXTERNAL]:
    d.mkdir(parents=True, exist_ok=True)


# ======================================================================
# Utilities
# ======================================================================

def rankpct(series):
    """截面 RankPct: [0, 1], NA → 0.5"""
    return series.rank(pct=True).fillna(0.5)


def cross_rankpct(df, col, date_col='date'):
    """每日截面 RankPct"""
    return df.groupby(date_col)[col].transform(rankpct)


def winsorize(series, lo=0.01, hi=0.99):
    qlo, qhi = series.quantile(lo), series.quantile(hi)
    return series.clip(qlo, qhi)


# ======================================================================
# Tushare Data Layer (primary)
# ======================================================================

class TushareClient:
    """Tushare API wrapper. Requires TUSHARE_TOKEN env var or config."""

    def __init__(self, token=None):
        import os
        self.token = token or os.environ.get('TUSHARE_TOKEN', '')
        self.pro = None

    def _connect(self):
        if self.pro is None and self.token:
            import tushare as ts
            ts.set_token(self.token)
            self.pro = ts.pro_api()

    def fetch_daily(self, codes, start_date, end_date):
        """Fetch daily price data via Tushare pro.daily()"""
        self._connect()
        if self.pro is None:
            return None

        all_data = []
        trade_dates = self.pro.trade_cal(
            exchange='SSE', start_date=start_date, end_date=end_date,
            is_open='1'
        )
        for _, row in trade_dates.iterrows():
            d = row['cal_date']
            try:
                df = self.pro.daily(trade_date=d)
                if not df.empty:
                    all_data.append(df)
                time.sleep(0.1)
            except Exception:
                pass
        return pd.concat(all_data, ignore_index=True) if all_data else None

    def fetch_daily_basic(self, start_date, end_date):
        """Fetch daily_basic (market cap, turnover, valuation)"""
        self._connect()
        if self.pro is None:
            return None

        all_data = []
        trade_dates = self.pro.trade_cal(
            exchange='SSE', start_date=start_date, end_date=end_date,
            is_open='1'
        )
        for _, row in trade_dates.iterrows():
            try:
                df = self.pro.daily_basic(
                    trade_date=row['cal_date'],
                    fields='ts_code,trade_date,turnover_rate,volume_ratio,'
                           'pe_ttm,pb,ps_ttm,total_mv,circ_mv,float_share'
                )
                if not df.empty:
                    all_data.append(df)
                time.sleep(0.15)
            except Exception:
                pass
        return pd.concat(all_data, ignore_index=True) if all_data else None

    def fetch_financials(self, codes, start_date):
        """Fetch financial indicators with ann_date for PIT alignment."""
        self._connect()
        if self.pro is None:
            return None

        try:
            df = self.pro.fina_indicator(
                ts_code=','.join(codes[:500]),
                start_date=start_date,
                fields='ts_code,ann_date,end_date,roe,roa,'
                       'grossprofit_margin,debt_to_assets,current_ratio,'
                       'ocf_to_operating_ni,accruals,bps,profit_dedt'
            )
            return df
        except Exception:
            return None

    def fetch_moneyflow(self, codes, start_date, end_date):
        """Fetch individual stock moneyflow."""
        self._connect()
        if self.pro is None:
            return None

        all_data = []
        for code in codes:
            try:
                df = self.pro.moneyflow(
                    ts_code=code, start_date=start_date, end_date=end_date
                )
                if not df.empty:
                    all_data.append(df)
                time.sleep(0.2)
            except Exception:
                pass
        return pd.concat(all_data, ignore_index=True) if all_data else None


# ======================================================================
# AKShare Fallback (demo mode, no token needed)
# ======================================================================

def _fetch_akshare_prices(codes, days=250):
    """Fallback: fetch price data via akshare."""
    end = datetime.now()
    start = end - timedelta(days=days)
    sd, ed = start.strftime('%Y%m%d'), end.strftime('%Y%m%d')

    results = []
    for i, code in enumerate(codes):
        for attempt in range(2):  # reduced retries
            try:
                import akshare as ak
                df = ak.stock_zh_a_hist(
                    symbol=code, period='daily',
                    start_date=sd, end_date=ed, adjust='qfq',
                    timeout=8  # short timeout
                )
                if df is not None and len(df) >= 60:
                    df['code'] = code
                    df['date'] = pd.to_datetime(df['日期'])
                    df = df.rename(columns={
                        '开盘': 'open', '收盘': 'close', '最高': 'high',
                        '最低': 'low', '成交量': 'volume', '成交额': 'amount',
                        '换手率': 'turnover'
                    })
                    keep = ['date', 'code', 'open', 'high', 'low', 'close',
                            'volume', 'amount', 'turnover']
                    results.append(df[[c for c in keep if c in df.columns]])
                break
            except Exception:
                if attempt < 1:
                    time.sleep(0.5)
        if (i + 1) % 5 == 0:
            print(f'    [{i+1}/{len(codes)}] ({len(results)} valid)...')
        time.sleep(0.15)  # rate limit avoidance
    return pd.concat(results, ignore_index=True) if results else None


CSI1000_FALLBACK = [
    '002415', '300750', '300124', '002475', '300059', '002230',
    '002142', '300274', '002594', '300014', '002714', '300408',
    '300433', '002304', '300498', '300760', '002460', '300122',
    '002456', '300394', '002916', '300782', '002049', '688981',
    '603501', '688012', '300661', '002241', '688536', '603290',
    '688037', '002812', '300763', '688390', '300496', '688256',
    '300223', '002920', '603986', '688111', '300677', '688036',
    '002841', '300529', '688521', '002985', '300595', '600536',
    '603160', '688188', '002371', '688396', '603267', '002463',
    '688561', '688208', '300136', '603890', '002384', '688798',
    '300303', '603129', '688556', '002791', '300285', '688589',
    '002407', '300454', '688005', '002850', '300627', '002957',
    '300638', '688321', '002993', '300751', '603538', '002979',
    '688608', '603899', '300087', '688169', '002906', '300346',
    '688262', '002737', '300470', '688277', '002831', '300298',
    '688333', '001308',
]


def _code_to_ts(code):
    """Convert 6-digit code to Tushare format: '600519.SH' or '000001.SZ'"""
    code = str(code).zfill(6)
    suffix = '.SH' if code.startswith(('6', '9')) else '.SZ'
    return code + suffix


def _fetch_tushare_prices(codes, token, days=200):
    """Fetch daily price + daily_basic via Tushare."""
    import tushare as ts

    ts.set_token(token)
    pro = ts.pro_api()

    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        cal = pro.trade_cal(exchange='SSE', start_date=start.strftime('%Y%m%d'),
                            end_date=end.strftime('%Y%m%d'), is_open='1')
        trade_dates = sorted(cal['cal_date'].tolist())
    except Exception:
        trade_dates = []
        d = start
        while d <= end:
            if d.weekday() < 5:
                trade_dates.append(d.strftime('%Y%m%d'))
            d += timedelta(days=1)

    ts_codes = set(_code_to_ts(c) for c in codes)
    print(f'  Tushare: fetching {len(ts_codes)} stocks x {len(trade_dates)} dates...')

    # --- Fetch daily + daily_basic per date ---
    daily_parts = []
    basic_parts = []

    for i, d in enumerate(trade_dates):
        try:
            df_daily = pro.daily(trade_date=d)
            if df_daily is not None and not df_daily.empty:
                df_daily = df_daily[df_daily['ts_code'].isin(ts_codes)].copy()
                if len(df_daily) > 0:
                    df_daily['date'] = pd.to_datetime(d)
                    daily_parts.append(df_daily)
        except Exception:
            pass

        try:
            df_basic = pro.daily_basic(trade_date=d,
                fields='ts_code,trade_date,turnover_rate,turnover_rate_f,'
                       'volume_ratio,pe_ttm,pb,ps_ttm,total_mv,circ_mv,float_share')
            if df_basic is not None and not df_basic.empty:
                df_basic = df_basic[df_basic['ts_code'].isin(ts_codes)].copy()
                if len(df_basic) > 0:
                    df_basic['date'] = pd.to_datetime(d)
                    basic_parts.append(df_basic)
        except Exception:
            pass

        if (i + 1) % 30 == 0:
            print(f'    [{i+1}/{len(trade_dates)}] dates...')
        time.sleep(0.35)

    if not daily_parts:
        return None

    # --- Merge daily + basic ---
    daily_df = pd.concat(daily_parts, ignore_index=True)
    daily_df = daily_df.rename(columns={
        'open': 'open', 'high': 'high', 'low': 'low',
        'close': 'close', 'vol': 'volume', 'amount': 'amount'
    })
    daily_df['code'] = daily_df['ts_code'].str[:6]
    daily_df['trade_date'] = daily_df['date'].dt.strftime('%Y%m%d')

    keep_daily = ['date', 'code', 'open', 'high', 'low', 'close',
                  'volume', 'amount', 'trade_date', 'ts_code']
    panel = daily_df[[c for c in keep_daily if c in daily_df.columns]].copy()

    if basic_parts:
        basic_df = pd.concat(basic_parts, ignore_index=True)
        basic_df['code'] = basic_df['ts_code'].str[:6]
        basic_df['trade_date'] = basic_df['date'].dt.strftime('%Y%m%d')
        basic_df = basic_df.rename(columns={
            'turnover_rate': 'turnover', 'turnover_rate_f': 'turnover_f',
            'volume_ratio': 'vol_ratio', 'pe_ttm': 'pe', 'pb': 'pb',
            'ps_ttm': 'ps', 'total_mv': 'total_mv', 'circ_mv': 'circ_mv',
            'float_share': 'float_pct'
        })
        merge_cols = ['code', 'trade_date']
        basic_keep = merge_cols + ['turnover', 'turnover_f', 'vol_ratio',
                                    'pe', 'pb', 'ps', 'total_mv', 'circ_mv']
        panel = panel.merge(
            basic_df[[c for c in basic_keep if c in basic_df.columns]],
            on=merge_cols, how='left'
        )

    n_st = panel['code'].nunique()
    n_dt = panel['date'].nunique()
    has_turnover = 'turnover' in panel.columns and panel['turnover'].notna().sum() > 100
    has_mcap = 'total_mv' in panel.columns and panel['total_mv'].notna().sum() > 100
    print(f'  Tushare OK: {len(panel):,} obs | {n_st} stocks | {n_dt} dates')
    print(f'    turnover: {"YES" if has_turnover else "NO"} | '
          f'market_cap: {"YES" if has_mcap else "NO"} | '
          f'PE/PB: {"YES" if "pe" in panel.columns else "NO"}')
    return panel


def _fetch_akshare_csi1000(max_stocks=None):
    """Get CSI 1000 constituents via akshare, with fallback list."""
    try:
        import akshare as ak
        df = ak.index_stock_cons(symbol='000852')
        codes = [str(c).strip().zfill(6) for c in df['品种代码'].tolist()]
        codes = sorted(set(c for c in codes if c.isdigit() and len(c) == 6))
        return codes[:max_stocks] if max_stocks else codes
    except Exception:
        return CSI1000_FALLBACK[:max_stocks] if max_stocks else CSI1000_FALLBACK


# ======================================================================
# Unified Fetch API
# ======================================================================

def fetch_data_pipeline(n_stocks=100, token=None):
    """
    Main data pipeline.

    1. Stock universe → CSI 1000 (small/mid-cap focus)
    2. Daily prices via Tushare (primary) or akshare (fallback)
    3. Build processed panel: [date, code, close, volume, amount, turnover...]
    4. Compute derived metrics (returns, volatility, illiquidity)
    """
    import os
    token = token or os.environ.get('TUSHARE_TOKEN', '')

    print(f'\n[Data Pipeline] Fetching {n_stocks} stocks...')

    # Step 1: Universe
    codes = _fetch_akshare_csi1000(n_stocks)
    print(f'  Universe: {len(codes)} CSI 1000 stocks')

    # Step 2: Price data — try Tushare first
    panel = None
    if token:
        panel = _fetch_tushare_prices(codes, token)

    if panel is None or panel['code'].nunique() < max(5, n_stocks * 0.1):
        # Fallback to akshare
        print('  Tushare unavailable — trying akshare...')
        panel = _fetch_akshare_prices(codes)

    if panel is None or panel['code'].nunique() < max(5, n_stocks * 0.1):
        print('  All APIs unavailable — using synthetic demo data.')
        from .demo_data import generate_demo_panel
        panel = generate_demo_panel(n_stocks=n_stocks)

    # Step 3: Compute derived metrics
    panel = _compute_panel_metrics(panel)

    n_stocks_valid = panel['code'].nunique()
    n_dates = panel['date'].nunique()
    date_min, date_max = panel['date'].min().date(), panel['date'].max().date()
    print(f'  Panel: {len(panel):,} obs | {n_stocks_valid} stocks | '
          f'{n_dates} days | {date_min} → {date_max}')

    return panel


def _build_panel_tushare(daily_df, basic_df):
    """Build unified panel from Tushare daily + daily_basic."""
    daily = daily_df.rename(columns={
        'ts_code': 'tushare_code', 'trade_date': 'date',
        'open': 'open', 'high': 'high', 'low': 'low',
        'close': 'close', 'vol': 'volume', 'amount': 'amount'
    })
    daily['date'] = pd.to_datetime(daily['date'])
    daily['code'] = daily['tushare_code'].str[:6]

    if basic_df is not None and not basic_df.empty:
        basic = basic_df.rename(columns={
            'ts_code': 'tushare_code', 'trade_date': 'date',
            'turnover_rate': 'turnover', 'circ_mv': 'circ_mv',
            'total_mv': 'total_mv', 'pe_ttm': 'pe', 'pb': 'pb'
        })
        basic['date'] = pd.to_datetime(basic['date'])
        basic['code'] = basic['tushare_code'].str[:6]
        panel = daily.merge(
            basic[['date', 'code', 'turnover', 'circ_mv', 'total_mv', 'pe', 'pb']],
            on=['date', 'code'], how='left'
        )
    else:
        panel = daily

    return panel.sort_values(['code', 'date']).reset_index(drop=True)


def _compute_panel_metrics(panel):
    """Compute derived metrics for the processed panel."""
    df = panel.copy()

    for col in ['close', 'volume', 'amount', 'turnover']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for code, grp in df.groupby('code'):
        idx = df['code'] == code
        close = grp['close'].values
        vol = grp['volume'].values if 'volume' in grp.columns else np.ones(len(grp))
        amt = grp['amount'].values if 'amount' in grp.columns else close * vol

        # Returns
        df.loc[idx, 'ret_1d'] = np.concatenate([[np.nan], np.diff(close) / close[:-1]])
        df.loc[idx, 'ret_5d'] = np.concatenate([[np.nan]*5, close[5:]/close[:-5]-1])
        df.loc[idx, 'ret_10d'] = np.concatenate([[np.nan]*10, close[10:]/close[:-10]-1])
        df.loc[idx, 'ret_20d'] = np.concatenate([[np.nan]*20, close[20:]/close[:-20]-1])

        # Amihud illiquidity
        df.loc[idx, 'amihud'] = np.abs(df.loc[idx, 'ret_1d'].values) / (amt * 1e-8 + 0.01)

        # Volatility (20d rolling)
        df.loc[idx, 'volatility_20d'] = (
            pd.Series(df.loc[idx, 'ret_1d'].values).rolling(20).std().values
        )

        # Dollar volume
        df.loc[idx, 'dollar_volume'] = amt

        # Turnover metrics
        if 'turnover' in grp.columns:
            df.loc[idx, 'turnover_avg_20d'] = (
                pd.Series(grp['turnover'].values).rolling(20).mean().values
            )
            df.loc[idx, 'turnover_spike'] = (
                grp['turnover'].values /
                (df.loc[idx, 'turnover_avg_20d'].values + 1e-10)
            )

        # Abnormal turnover (per data guide formula)
        if 'turnover' in grp.columns:
            turnover_series = pd.Series(grp['turnover'].values)
            turnover_ma60 = turnover_series.rolling(60, min_periods=10)
            df.loc[idx, 'ab_turnover'] = (
                turnover_series.values / (turnover_ma60.mean().shift(1).values + 1e-10)
            )

        # Abnormal amount
        amount_series = pd.Series(amt)
        amount_ma60 = amount_series.rolling(60, min_periods=10)
        df.loc[idx, 'ab_amount'] = (
            amount_series.values / (amount_ma60.mean().shift(1).values + 1e-10)
        )

        # Intraday volatility
        if 'high' in grp.columns and 'low' in grp.columns:
            pre_close = np.concatenate([[close[0]], close[:-1]])
            df.loc[idx, 'intraday_vol'] = (
                (grp['high'].values - grp['low'].values) / (pre_close + 1e-10)
            )

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df
