"""
data_process_ext.py
====================
Extended data fetching: financial data + moneyflow.
Added after Tushare permission upgrade.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time


def fetch_financials(pro, codes, start_date='20240101'):
    """
    Fetch financial indicators for QualityScore.

    Returns DataFrame with PIT-aligned financial metrics:
      [ts_code, ann_date, end_date, roe, roa, current_ratio,
       debt_to_assets, grossprofit_margin, ocf_to_operating_ni, bps]
    """
    ts_codes = [f'{c}.SH' if c.startswith('6') else f'{c}.SZ' for c in codes]

    all_data = []
    for i, tc in enumerate(ts_codes):
        try:
            df = pro.fina_indicator(
                ts_code=tc, start_date=start_date,
                end_date=datetime.now().strftime('%Y%m%d'),
                fields='ts_code,ann_date,end_date,roe,roa,'
                       'current_ratio,debt_to_assets,grossprofit_margin,'
                       'ocf_to_operating_ni,bps,or_yoy,profit_dedt'
            )
            if df is not None and len(df) > 0:
                df['code'] = tc[:6]
                all_data.append(df)
        except Exception:
            pass
        if (i + 1) % 20 == 0:
            print(f'    fina: {i+1}/{len(ts_codes)}...')
        time.sleep(0.3)

    if not all_data:
        return None

    return pd.concat(all_data, ignore_index=True)


def fetch_moneyflows(pro, codes, start_date='20251001'):
    """
    Fetch individual stock moneyflow for FundingVacuum.

    Returns DataFrame with:
      [ts_code, trade_date, buy_sm_vol, sell_sm_vol,
       buy_md_vol, sell_md_vol, buy_lg_vol, sell_lg_vol,
       buy_elg_vol, sell_elg_vol]
    """
    ts_codes = [f'{c}.SH' if c.startswith('6') else f'{c}.SZ' for c in codes]
    end_date = datetime.now().strftime('%Y%m%d')

    all_data = []
    for i, tc in enumerate(ts_codes):
        try:
            df = pro.moneyflow(ts_code=tc, start_date=start_date, end_date=end_date)
            if df is not None and len(df) > 0:
                df['code'] = tc[:6]
                all_data.append(df)
        except Exception:
            pass
        if (i + 1) % 20 == 0:
            print(f'    moneyflow: {i+1}/{len(ts_codes)}...')
        time.sleep(0.35)

    if not all_data:
        return None

    mf = pd.concat(all_data, ignore_index=True)
    mf['date'] = pd.to_datetime(mf['trade_date'])
    return mf


def merge_financials_to_panel(panel, financials):
    """
    Merge financial data into panel with PIT alignment.

    For each stock on each date, take the most recent financial report
    whose ann_date <= trade_date (point-in-time, no look-ahead).
    """
    if financials is None or len(financials) == 0:
        return panel

    fin = financials.copy()
    fin['ann_date_dt'] = pd.to_datetime(fin['ann_date'], format='%Y%m%d', errors='coerce')
    fin = fin.dropna(subset=['ann_date_dt'])  # remove null dates

    # For each (code, date) in panel, find latest financial report
    panel_dates = panel[['code', 'date']].drop_duplicates()
    panel_dates = panel_dates.sort_values(['code', 'date'])

    merged_rows = []
    for code, grp in panel_dates.groupby('code'):
        code_fin = fin[fin['code'] == code].sort_values('ann_date_dt')
        if len(code_fin) == 0:
            continue

        # Merge asof: for each trade date, get latest fin report
        grp_sorted = grp.sort_values('date')
        code_fin_sorted = code_fin.sort_values('ann_date_dt')

        # Drop code from financials to avoid merge conflict with grp's code
        code_fin_sorted = code_fin.drop(columns=['code', 'ts_code'], errors='ignore')
        code_fin_sorted = code_fin_sorted.sort_values('ann_date_dt')

        merged = pd.merge_asof(
            grp_sorted, code_fin_sorted,
            left_on='date', right_on='ann_date_dt',
            direction='backward'
        )
        merged_rows.append(merged)

    if not merged_rows:
        return panel

    fin_panel = pd.concat(merged_rows, ignore_index=True)
    if isinstance(fin_panel.columns, pd.MultiIndex):
        fin_panel.columns = ['_'.join(str(c) for c in col).strip('_') for col in fin_panel.columns]

    # Select key columns
    fin_cols = ['code', 'date', 'roe', 'roa', 'current_ratio',
                'debt_to_assets', 'grossprofit_margin',
                'ocf_to_operating_ni', 'bps', 'or_yoy', 'profit_dedt']
    available = [c for c in fin_cols if c in fin_panel.columns]

    panel = panel.merge(
        fin_panel[available], on=['code', 'date'], how='left'
    )
    return panel


def merge_moneyflow_to_panel(panel, moneyflow_df):
    """
    Merge moneyflow data into panel.

    Computes net institutional flow:
      main_net = buy_lg_vol - sell_lg_vol + buy_elg_vol - sell_elg_vol
    """
    if moneyflow_df is None or len(moneyflow_df) == 0:
        return panel

    mf = moneyflow_df.copy()

    # Net large + extra-large order flow (institutional proxy)
    mf['main_net'] = 0
    for side, sign in [('buy_lg_vol', 1), ('sell_lg_vol', -1),
                        ('buy_elg_vol', 1), ('sell_elg_vol', -1)]:
        if side in mf.columns:
            mf['main_net'] += sign * pd.to_numeric(mf[side], errors='coerce').fillna(0)

    mf_cols = ['code', 'date', 'main_net']
    available = [c for c in mf_cols if c in mf.columns]

    mf_sub = mf[available].drop_duplicates(subset=['code', 'date'])
    panel = panel.merge(mf_sub, on=['code', 'date'], how='left')
    return panel


# ============================================================================
# Extended data: express (业绩快报), income/cashflow, hk_hold
# ============================================================================

def fetch_express(pro, codes, start_date='20250101'):
    """
    Fetch earnings express (业绩快报) for NonFundamentalOK.
    Returns: [code, ann_date, end_date, revenue, n_income, yoy_net_profit]
    """
    ts_codes = [f'{c}.SH' if c.startswith('6') else f'{c}.SZ' for c in codes]
    all_data = []
    for tc in ts_codes:
        try:
            df = pro.express(ts_code=tc, start_date=start_date,
                             end_date=datetime.now().strftime('%Y%m%d'))
            if df is not None and len(df) > 0:
                df['code'] = tc[:6]
                all_data.append(df)
        except Exception:
            pass
        time.sleep(0.3)
    if not all_data:
        return None
    return pd.concat(all_data, ignore_index=True)


def fetch_income_cashflow(pro, codes, start_date='20250101'):
    """
    Fetch income statement + cashflow for deeper QualityScore.
    """
    ts_codes = [f'{c}.SH' if c.startswith('6') else f'{c}.SZ' for c in codes]
    all_data = []
    for tc in ts_codes:
        try:
            inc = pro.income(ts_code=tc, start_date=start_date,
                             end_date=datetime.now().strftime('%Y%m%d'),
                             fields='ts_code,ann_date,end_date,revenue,operate_profit,n_income,total_cogs')
            cf = pro.cashflow(ts_code=tc, start_date=start_date,
                              end_date=datetime.now().strftime('%Y%m%d'),
                              fields='ts_code,ann_date,n_cashflow_act')
            if inc is not None and len(inc) > 0:
                inc['code'] = tc[:6]
                if cf is not None and len(cf) > 0:
                    cf = cf.rename(columns={'n_cashflow_act': 'oper_cf'})
                    inc = inc.merge(cf[['ts_code', 'ann_date', 'oper_cf']],
                                   on=['ts_code', 'ann_date'], how='left')
                else:
                    inc['oper_cf'] = np.nan
                all_data.append(inc)
        except Exception:
            pass
        time.sleep(0.35)
    if not all_data:
        return None
    return pd.concat(all_data, ignore_index=True)


def merge_express_to_panel(panel, express_df):
    """
    Merge express data: if recent earnings flash shows significant
    profit decline vs prior period → flag as potential fundamental issue.
    Adds column: express_neg (1 = negative flash report relative to prior).
    """
    if express_df is None or len(express_df) == 0:
        return panel

    exp = express_df.copy()
    exp['ann_date_dt'] = pd.to_datetime(exp['ann_date'], format='%Y%m%d', errors='coerce')
    exp = exp.dropna(subset=['ann_date_dt'])

    # Merge PIT
    panel_dates = panel[['code', 'date']].drop_duplicates().sort_values(['code', 'date'])
    merged_rows = []
    for code, grp in panel_dates.groupby('code'):
        code_exp = exp[exp['code'] == code].sort_values('ann_date_dt')
        if len(code_exp) == 0:
            continue
        grp_s = grp.sort_values('date')
        code_exp_s = code_exp.drop(columns=['code', 'ts_code'], errors='ignore')
        merged = pd.merge_asof(grp_s, code_exp_s.sort_values('ann_date_dt'),
                               left_on='date', right_on='ann_date_dt', direction='backward')
        merged_rows.append(merged)
    if not merged_rows:
        return panel

    exp_panel = pd.concat(merged_rows, ignore_index=True)
    if isinstance(exp_panel.columns, pd.MultiIndex):
        exp_panel.columns = ['_'.join(str(c) for c in col).strip('_') for col in exp_panel.columns]
    keep = ['code', 'date', 'n_income']
    available = [c for c in keep if c in exp_panel.columns]
    return panel.merge(exp_panel[available], on=['code', 'date'], how='left')


def merge_income_to_panel(panel, income_df):
    """
    Merge income/cashflow data (PIT aligned).
    Adds: oper_cf, n_income_inc (net profit from income statement),
          oper_profit_inc
    """
    if income_df is None or len(income_df) == 0:
        return panel

    inc = income_df.copy()
    inc['ann_date_dt'] = pd.to_datetime(inc['ann_date'], format='%Y%m%d', errors='coerce')
    inc = inc.dropna(subset=['ann_date_dt'])

    panel_dates = panel[['code', 'date']].drop_duplicates().sort_values(['code', 'date'])
    merged_rows = []
    for code, grp in panel_dates.groupby('code'):
        code_inc = inc[inc['code'] == code].sort_values('ann_date_dt')
        if len(code_inc) == 0:
            continue
        grp_s = grp.sort_values('date')
        code_inc_s = code_inc.drop(columns=['code', 'ts_code'], errors='ignore')
        merged = pd.merge_asof(grp_s, code_inc_s.sort_values('ann_date_dt'),
                               left_on='date', right_on='ann_date_dt', direction='backward')
        merged_rows.append(merged)
    if not merged_rows:
        return panel

    inc_panel = pd.concat(merged_rows, ignore_index=True)
    if isinstance(inc_panel.columns, pd.MultiIndex):
        inc_panel.columns = ['_'.join(str(c) for c in col).strip('_') for col in inc_panel.columns]
    keep = ['code', 'date', 'n_income', 'oper_cf', 'revenue', 'operate_profit']
    available = [c for c in keep if c in inc_panel.columns]
    return panel.merge(inc_panel[available], on=['code', 'date'], how='left')


# ============================================================================
# Forecast data (业绩预告) — for NonFundamentalOK real-time earnings warning
# ============================================================================

def fetch_forecast(pro, codes, ann_date='20260331'):
    """
    Fetch earnings forecasts for NonFundamentalOK.

    Negative forecast types indicate fundamental deterioration:
      预减, 首亏, 续亏, 增亏, 略减

    Returns DataFrame with [code, ann_date, type, p_change_min, net_profit_min].
    """
    all_data = []
    for tc in [f'{c}.SH' if c.startswith('6') else f'{c}.SZ' for c in codes]:
        try:
            df = pro.forecast(ts_code=tc, start_date='20251001',
                             end_date=datetime.now().strftime('%Y%m%d'),
                             fields='ts_code,ann_date,end_date,type,p_change_min,p_change_max,net_profit_min,net_profit_max')
            if df is not None and len(df) > 0:
                df['code'] = tc[:6]
                all_data.append(df)
        except Exception:
            pass
        time.sleep(0.3)
    if not all_data:
        return None
    return pd.concat(all_data, ignore_index=True)


def merge_forecast_to_panel(panel, forecast_df):
    """
    Merge forecast data into panel (PIT aligned).

    Adds column: forecast_neg (1 = negative forecast type).
    Negative types: 预减, 首亏, 续亏, 增亏, 略减
    """
    if forecast_df is None or len(forecast_df) == 0:
        return panel

    fc = forecast_df.copy()
    fc['ann_date_dt'] = pd.to_datetime(fc['ann_date'], format='%Y%m%d', errors='coerce')
    fc = fc.dropna(subset=['ann_date_dt'])

    # Mark negative forecast types
    neg_types = ['预减', '首亏', '续亏', '增亏', '略减']
    fc['forecast_neg'] = fc['type'].isin(neg_types).astype(int)

    # PIT merge
    panel_dates = panel[['code', 'date']].drop_duplicates().sort_values(['code', 'date'])
    merged_rows = []
    for code, grp in panel_dates.groupby('code'):
        code_fc = fc[fc['code'] == code].sort_values('ann_date_dt')
        if len(code_fc) == 0:
            continue
        grp_s = grp.sort_values('date')
        code_fc_s = code_fc.drop(columns=['code', 'ts_code'], errors='ignore')
        merged = pd.merge_asof(grp_s, code_fc_s.sort_values('ann_date_dt'),
                               left_on='date', right_on='ann_date_dt', direction='backward')
        merged_rows.append(merged)
    if not merged_rows:
        return panel

    fc_panel = pd.concat(merged_rows, ignore_index=True)
    if isinstance(fc_panel.columns, pd.MultiIndex):
        fc_panel.columns = ['_'.join(str(c) for c in col).strip('_') for col in fc_panel.columns]
    keep = ['code', 'date', 'forecast_neg', 'p_change_min', 'type']
    available = [c for c in keep if c in fc_panel.columns]
    return panel.merge(fc_panel[available], on=['code', 'date'], how='left')
