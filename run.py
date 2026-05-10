#!/usr/bin/env python3
"""
run.py — HAMR Mispricing Reversion Factor: Full Academic Pipeline
==================================================================

Usage:
    python run.py              # Standard (100 stocks CSI 1000)
    python run.py --quick      # Quick (50 stocks)
    python run.py --full       # Full (500 stocks)

Pipeline:
    Step 1: Data (Tushare/akshare → processed panel)
    Step 2: AIHeat state
    Step 3: Template Mismatch
    Step 4: QualityScore
    Step 5: ResidualWeakness + NonFundamentalOK → MispricingPressure
    Step 6: FundingVacuum + TrapGuard
    Step 7: HAMR factor assembly
    Step 8: IC analysis (Rank IC + ICIR + Newey-West)
    Step 9: Quintile backtest + Control comparison
    Step 10: Mechanism tests
    Step 11: ML extension (optional)
    Step 12: Visualization
"""

import sys
import warnings
import numpy as np
import pandas as pd
import tushare as ts
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(ROOT))

from config import TUSHARE_TOKEN, GITHUB_TOKEN

ts.set_token(TUSHARE_TOKEN)
PRO = ts.pro_api()

from src import (data_process, ai_heat, template_cluster, quality,
                  residual, funding_vacuum, hamr_factor,
                  backtest, regression, mechanism, ml_model, visualization)


CONFIG = {
    'quick':   {'n_stocks': 200, 'label': 'Quick Demo'},
    'default': {'n_stocks': 400, 'label': 'Standard'},
    'full':    {'n_stocks': 500, 'label': 'Full CSI 1000'},
}


def sep(title):
    print(f'\n{"="*65}\n  {title}\n{"="*65}')


def main():
    mode = 'default'
    if '--full' in sys.argv:
        mode = 'full'
    elif '--quick' in sys.argv:
        mode = 'quick'

    cfg = CONFIG[mode]

    print(f'\n  HAMR Factor — Academic Verification Pipeline')
    print(f'  Mode: {cfg["label"]} ({cfg["n_stocks"]} stocks)')
    print(f'  Time: {datetime.now().strftime("%Y-%m-%d %H:%M")}')

    # ================================================================
    # Step 1: Data
    # ================================================================
    sep('Step 1: Data Acquisition (CSI 1000)')
    panel = data_process.fetch_data_pipeline(n_stocks=cfg['n_stocks'], token=TUSHARE_TOKEN)

    # --- Extended data: financials + moneyflow ---
    sep('Step 1b: Financial & Fund Flow Data')
    from src.data_process_ext import (fetch_financials, fetch_moneyflows,
                                        merge_financials_to_panel,
                                        merge_moneyflow_to_panel)

    codes = panel['code'].unique().tolist()
    print(f'  Fetching financials for {len(codes)} stocks...')
    financials = fetch_financials(PRO, codes)
    if financials is not None:
        print(f'    fina_indicator: {len(financials)} rows')
        panel = merge_financials_to_panel(panel, financials)

    print(f'  Fetching moneyflow for {len(codes)} stocks...')
    moneyflow = fetch_moneyflows(PRO, codes)
    if moneyflow is not None:
        print(f'    moneyflow: {len(moneyflow)} rows')
        panel = merge_moneyflow_to_panel(panel, moneyflow)

    # Additional data (already permitted)
    from src.data_process_ext import (
        fetch_express, fetch_income_cashflow,
        fetch_forecast, merge_forecast_to_panel,
        merge_express_to_panel, merge_income_to_panel
    )

    print(f'  Fetching express (earnings flash)...')
    express_df = fetch_express(PRO, codes)
    if express_df is not None and len(express_df) > 0:
        print(f'    express: {len(express_df)} rows')
        panel = merge_express_to_panel(panel, express_df)

    print(f'  Fetching income + cashflow...')
    inc_df = fetch_income_cashflow(PRO, codes)
    if inc_df is not None and len(inc_df) > 0:
        print(f'    income/cashflow: {len(inc_df)} rows')
        panel = merge_income_to_panel(panel, inc_df)

    print(f'  Fetching forecast (earnings warnings)...')
    fc_df = fetch_forecast(PRO, codes)
    if fc_df is not None and len(fc_df) > 0:
        neg_count = (fc_df['type'].isin(['预减','首亏','续亏','增亏','略减'])).sum() if 'type' in fc_df.columns else 0
        print(f'    forecast: {len(fc_df)} rows ({neg_count} negative)')
        panel = merge_forecast_to_panel(panel, fc_df)

    # --- Stock universe filters ---
    sep('Step 1c: Trading Constraints & Filters')
    from src import data_filters

    stock_info = data_filters.build_stock_info(PRO)

    # Get current date range for suspend/limit data
    dates_sorted = sorted(panel['date'].unique())
    sd_str = pd.to_datetime(dates_sorted[0]).strftime('%Y%m%d')
    ed_str = pd.to_datetime(dates_sorted[-1]).strftime('%Y%m%d')

    suspended = data_filters.fetch_suspend_dates(PRO, sd_str, ed_str)
    limit_df = data_filters.fetch_limit_prices(PRO, ed_str)

    panel = data_filters.filter_panel(panel, stock_info, suspended, limit_df)
    panel = data_filters.add_industry_to_panel(panel, stock_info)

    # Cache panel for faster re-runs
    cache_path = Path(ROOT) / 'data' / f'_panel_{cfg["n_stocks"]}stocks.pkl'
    panel.to_pickle(cache_path)
    print(f'  Cached: {cache_path}')

    # --- Guba community data ---
    from src.guba_data import fetch_universe_posts, compute_community_aiheat, compute_stock_attention
    print(f'\n  Fetching Guba (stock forum) data...')
    guba_posts = fetch_universe_posts(codes, max_workers=3)
    stock_attention_df = None
    if guba_posts:
        community = compute_community_aiheat(guba_posts)
        stock_attention_df = compute_stock_attention(None, guba_posts)
        if community:
            print(f'    Community: {community["total_posts"]} posts / '
                  f'{community["n_stocks"]} stocks = {community["avg_posts"]:.0f} avg'
                  f' (activity: {community["activity_score"]:.2f})')
    else:
        community = None

    # ================================================================
    # Steps 2-7: HAMR Factor
    # ================================================================
    from src.github_timeseries import build_github_timeseries
    print('\n  Building GitHub activity time series...')
    github_ts = build_github_timeseries(token=GITHUB_TOKEN)

    from src.news_aiheat import build_news_timeseries
    from src.ai_heat import load_search_aiheat
    print('  Fetching financial news AI intensity...')
    news_ts = build_news_timeseries()
    search_ts = load_search_aiheat()
    if search_ts is not None:
        print(f'    Search AIHeat: {len(search_ts)} data points loaded')
    if github_ts is not None and len(github_ts) > 10:
        print(f'    {len(github_ts)} daily points, '
              f'{int(github_ts["stars_cum"].max())} cumulative stars')
        ai_state = ai_heat.compute_ai_heat_from_panel(panel, github_ts=github_ts,
                news_ts=news_ts, community=community, search_ts=search_ts)
    else:
        from src.ai_heat import fetch_github_aiheat
        github_data = fetch_github_aiheat()
        print(f'    GitHub snapshot: {github_data["stars"]} stars'
              f' (time series unavailable, using snapshot)')
        ai_state = ai_heat.compute_ai_heat_from_panel(panel, github_data=github_data)
    print('  AIStateScore range: [{:.3f}, {:.3f}]'.format(
        ai_state['AIStateScore'].min(), ai_state['AIStateScore'].max()))

    try:
        mismatch = template_cluster.compute_template_affinity(panel, recent_only=True)
    except Exception:
        mismatch = template_cluster.compute_template_affinity_proxy(panel)
    print('  MismatchScore range: [{:.3f}, {:.3f}]'.format(
        mismatch['MismatchScore'].min(), mismatch['MismatchScore'].max()))

    qual = quality.compute_quality_score(panel)
    print('  QualityScore range: [{:.3f}, {:.3f}]'.format(
        qual['QualityScore'].min(), qual['QualityScore'].max()))

    res_weak = residual.compute_residual_weakness(panel)
    nonfund = residual.compute_non_fundamental_ok(panel)
    mispricing = residual.compute_mispricing_pressure(mismatch, res_weak, nonfund)
    print('  MispricingPressure range: [{:.3f}, {:.3f}]'.format(
        mispricing['MispricingPressure'].min(),
        mispricing['MispricingPressure'].max()))

    vacuum = funding_vacuum.compute_funding_vacuum(panel, stock_attention_df=stock_attention_df)
    liq = funding_vacuum.compute_liquidity_capacity(panel)
    trap = funding_vacuum.compute_trap_guard(qual, liq, panel=panel)

    # --- QuantDinger AI Flow (optional) ---
    from src.quantdinger_flow import load_quantdinger_ai_flow
    ai_flow_df = load_quantdinger_ai_flow()
    if ai_flow_df is not None:
        print(f'    QuantDinger AIFlow: {len(ai_flow_df)} rows')
    print('  FundingVacuum range: [{:.3f}, {:.3f}]'.format(
        vacuum['FundingVacuum'].min(), vacuum['FundingVacuum'].max()))

    panel = hamr_factor.compute_hamr(
        panel, ai_state, mismatch, qual, mispricing, vacuum, liq, trap
    )
    panel = hamr_factor.compute_forward_returns(panel)

    # ================================================================
    # Step 8: IC Analysis
    # ================================================================
    sep('Step 8: IC Analysis')
    from src.ic_test import full_ic_analysis
    ic_results = full_ic_analysis(panel, factor_col='hamr_zscore',
                                   horizons=(1, 5, 10, 20))

    print(f'\n  {"Horizon":>6s} | {"IC Mean":>8s} | {"ICIR":>7s} | '
          f'{"NW t":>7s} | {"IC>0":>6s} | Verdict')
    print(f'  {"-"*58}')
    for h, d in sorted(ic_results.items()):
        s = d['stats']
        print(f'  {h:>4s}d  | {s["ic_mean"]:+.4f}  | {s["icir"]:+.3f}  | '
              f'{s["nw_tstat"]:+.3f}  | {s["ic_pos_ratio"]:.1%}  | '
              f'{s["verdict"]}')

    # ================================================================
    # Step 9: Backtest
    # ================================================================
    sep('Step 9: Quintile Backtest + Control Comparison')
    grp = backtest.quintile_test(panel, factor_col='hamr_zscore',
                                  return_col='fwd_20d')

    print(f'\n  {"Portfolio":>8s} | Return | Count')
    for g in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        print(f'  {g:>8s} | {grp["group_means"][g]:+.4%} | '
              f'{grp["group_counts"].get(g, 0)}')
    print(f'  {"-"*30}')
    print(f'  Spread Q5-Q1: {grp["spread"]:+.4%}  '
          f'(ann. {grp["spread_annualized"]:+.1%})')
    print(f'  Monotonicity: {grp["monotonicity"]:+.3f}')

    comparisons = backtest.compare_controls(panel)
    for h, comps in comparisons.items():
        print(f'\n  {h}:')
        for name, r in comps.items():
            star = '>>' if name == 'HAMR_Final' and r['spread'] > 0 else '  '
            print(f'    {star} {name:>22s}: spread={r["spread"]:+.4%}')

    # ================================================================
    # Step 10: Mechanism
    # ================================================================
    sep('Step 10: Mechanism Tests')

    # Factor orthogonality
    corrs = mechanism.test_factor_orthogonality(panel)
    print('\n  Factor correlations with HAMR:')
    for name, cor in corrs.items():
        val_str = f'{cor:+.3f}' if not (isinstance(cor, float) and np.isnan(cor)) else 'N/A'
        print(f'    {name:>12s}: {val_str}')

    # AIHeat conditional
    ai_ic = backtest.conditional_ic_by_aistate(panel)
    if ai_ic:
        print('\n  AIHeat State-Conditional IC (10d):')
        for state, s in ai_ic.items():
            print(f'    {state:>15s}: IC={s["ic_mean"]:+.4f}, '
                  f'ICIR={s.get("icir", 0):+.3f}, {s["n_dates"]}d')

    # Reversal double-sort
    try:
        rev_test = mechanism.test_reversal_vs_hamr(panel)
        if rev_test:
            print('\n  HAMR IC within Reversal quintiles (10d):')
            for q, s in rev_test.items():
                print(f'    Rev {q:>4s}: IC={s["ic_mean"]:+.4f}, '
                      f't={s["t_stat"]:+.3f}, {s["n_dates"]}d')
    except Exception as e:
        print(f'\n  Reversal double-sort: skipped ({type(e).__name__})')

    # ================================================================
    # Step 11: ML (optional, skip on quick)
    # ================================================================
    if mode != 'quick':
        sep('Step 11: ML Extension')
        ml_results = ml_model.run_ml_pipeline(panel)
    else:
        ml_results = None

    # ================================================================
    # Step 12: Charts
    # ================================================================
    sep('Step 12: Visualization')

    # Component attribution
    from src.backtest import quintile_test
    attribution = {}
    for comp in ['MismatchScore', 'QualityScore', 'ResidualWeakness',
                 'NonFundamentalOK', 'FundingVacuum', 'LiquidityCapacity']:
        if comp in panel.columns:
            try:
                ret_col = 'fwd_10d'
                valid = panel[[comp, ret_col]].dropna()
                ics = []
                for _, g in valid.groupby(panel.loc[valid.index, 'date']):
                    if len(g) >= 10:
                        from scipy import stats
                        ic, _ = stats.spearmanr(g[comp], g[ret_col])
                        ics.append(ic)
                ics = np.array(ics)
                attribution[comp] = {
                    'ic_mean': float(ics.mean()) if len(ics) > 0 else 0,
                    'icir': float(ics.mean()/ics.std()) if len(ics)>1 and ics.std()>0 else 0
                }
            except Exception:
                attribution[comp] = {'ic_mean': 0}

    visualization.generate_all(panel, ic_results, grp,
                                comparisons, attribution)

    # ================================================================
    # Done
    # ================================================================
    sep('Summary')
    print(f'  Pipeline complete.')
    print(f'  Factor: HAMR (Homogeneous AI Mispricing Reversion)')
    print(f'  Stocks: {panel["code"].nunique()}  |  '
          f'Days: {panel["date"].nunique()}')
    print(f'  Charts: {visualization.CHART_DIR}')
    print(f'\n  Not financial advice. Academic research only.')


if __name__ == '__main__':
    main()
