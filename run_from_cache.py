#!/usr/bin/env python3
"""
run_from_cache.py — Skip data fetch, load cached panel, compute HAMR factor
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
from src import (ai_heat, template_cluster, quality,
                  residual, funding_vacuum, hamr_factor,
                  backtest, regression, mechanism, ml_model, visualization)

ts.set_token(TUSHARE_TOKEN)
PRO = ts.pro_api()

def sep(title):
    print(f'\n{"="*65}\n  {title}\n{"="*65}')

def main():
    print(f'\n  HAMR Factor — From Cache')
    print(f'  Time: {datetime.now().strftime("%Y-%m-%d %H:%M")}')

    # Load cached panel
    cache_path = Path(ROOT) / 'data' / '_panel_200stocks.pkl'
    print(f'  Loading: {cache_path}')
    panel = pd.read_pickle(cache_path)
    print(f'  Panel: {len(panel)} obs | {panel["code"].nunique()} stocks | {panel["date"].nunique()} days')
    print(f'  Columns: {list(panel.columns)[:20]}...')

    codes = panel['code'].unique().tolist()

    # Steps 2-7: HAMR Factor
    sep('Steps 2-7: HAMR Factor Assembly')

    from src.github_timeseries import build_github_timeseries
    print('  Building GitHub activity time series...')
    github_ts = build_github_timeseries(token=GITHUB_TOKEN)
    if github_ts is not None and len(github_ts) > 10:
        print(f'    {len(github_ts)} daily points, '
              f'{int(github_ts["stars_cum"].max())} cumulative stars')
    else:
        print('    GitHub TS: insufficient data')

    from src.news_aiheat import build_news_timeseries
    from src.ai_heat import load_search_aiheat
    print('  Fetching financial news AI intensity...')
    news_ts = build_news_timeseries()
    search_ts = load_search_aiheat()
    if search_ts is not None:
        print(f'    Search AIHeat: {len(search_ts)} data points loaded')

    # Guba data (may take time, skip if needed)
    from src.guba_data import fetch_universe_posts, compute_community_aiheat, compute_stock_attention
    print('  Fetching Guba (stock forum) data...')
    try:
        guba_posts = fetch_universe_posts(codes, max_workers=3)
        stock_attention_df = None
        if guba_posts:
            community = compute_community_aiheat(guba_posts)
            stock_attention_df = compute_stock_attention(None, guba_posts)
            if community:
                print(f'    Community: {community["total_posts"]} posts / '
                      f'{community["n_stocks"]} stocks = {community["avg_posts"]:.0f} avg')
        else:
            community = None
    except Exception as e:
        print(f'    Guba skipped: {e}')
        community = None
        stock_attention_df = None

    ai_state = ai_heat.compute_ai_heat_from_panel(panel, github_ts=github_ts,
            news_ts=news_ts, community=community, search_ts=search_ts)
    print('  AIStateScore range: [{:.3f}, {:.3f}]'.format(
        ai_state['AIStateScore'].min(), ai_state['AIStateScore'].max()))

    # Template Mismatch
    try:
        mismatch = template_cluster.compute_template_affinity(
            panel, recent_only=False, recent_days=20)
        print('  Template clustering: KMeans full-history')
    except Exception as e:
        print(f'  Template clustering failed; proxy: {type(e).__name__}: {e}')
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

    # Step 8: IC Analysis
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

    # Step 9: Backtest
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

    # Step 10: Mechanism
    sep('Step 10: Mechanism Tests')
    corrs = mechanism.test_factor_orthogonality(panel)
    print('\n  Factor correlations with HAMR:')
    for name, cor in corrs.items():
        val_str = f'{cor:+.3f}' if not (isinstance(cor, float) and np.isnan(cor)) else 'N/A'
        print(f'    {name:>12s}: {val_str}')

    # AIHeat conditional IC
    from src.backtest import conditional_ic_by_aistate
    ai_ic = conditional_ic_by_aistate(panel)
    if ai_ic:
        print('\n  AIHeat State-Conditional IC (10d):')
        for state, s in ai_ic.items():
            print(f'    {state:>15s}: IC={s["ic_mean"]:+.4f}, '
                  f'ICIR={s.get("icir", 0):+.3f}, {s["n_dates"]}d')

    try:
        rev_test = mechanism.test_reversal_vs_hamr(panel)
        if rev_test:
            print('\n  HAMR IC within Reversal quintiles (10d):')
            for q, s in rev_test.items():
                print(f'    Rev {q:>4s}: IC={s["ic_mean"]:+.4f}, '
                      f't={s["t_stat"]:+.3f}, {s["n_dates"]}d')
    except Exception as e:
        print(f'\n  Reversal double-sort: skipped ({type(e).__name__})')

    # Step 12: Charts
    sep('Step 12: Visualization')
    from src.backtest import quintile_test
    attribution = {}
    for comp in ['MismatchScore', 'QualityScore', 'ResidualWeakness',
                 'NonFundamentalOK', 'FundingVacuum', 'LiquidityCapacity']:
        if comp in panel.columns:
            try:
                from scipy import stats
                ret_col = 'fwd_10d'
                valid = panel[[comp, ret_col]].dropna()
                ics = []
                for _, g in valid.groupby(panel.loc[valid.index, 'date']):
                    if len(g) >= 10:
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

    # Summary
    sep('Summary')
    print(f'  Pipeline complete.')
    print(f'  Factor: HAMR (Homogeneous AI Mispricing Reversion)')
    print(f'  Stocks: {panel["code"].nunique()}  |  '
          f'Days: {panel["date"].nunique()}')
    print(f'  Charts: {visualization.CHART_DIR}')
    print(f'\n  Not financial advice. Academic research only.')

if __name__ == '__main__':
    main()
