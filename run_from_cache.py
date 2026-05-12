#!/usr/bin/env python3
"""
run_from_cache.py — Run HAMR pipeline from cached panel data
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(ROOT))

from config import TUSHARE_TOKEN, GITHUB_TOKEN

print('\n' + '='*65)
print('  HAMR Factor — Run from Cache (Updated GitHub Data)')
print('  Time:', datetime.now().strftime('%Y-%m-%d %H:%M'))
print('='*65)

# Load cached panel
print('\n[1/10] Loading cached panel data...')
panel = pd.read_pickle(ROOT / 'data' / '_panel_200stocks.pkl')
print(f'  Shape: {panel.shape}')
print(f'  Date range: {panel["date"].min()} to {panel["date"].max()}')
print(f'  Stocks: {panel["code"].nunique()}')

# Load updated GitHub data
print('\n[2/10] Loading updated GitHub AIHeat data...')
github_df = pd.read_csv('data/external/github_activity.csv')
github_df['date'] = pd.to_datetime(github_df['date'])
panel['date'] = pd.to_datetime(panel['date'])
panel = panel.merge(github_df[['date', 'github_activity_score']], on='date', how='left')
panel['AIHeat'] = panel['github_activity_score'].fillna(0.5)
print(f'  AIHeat range: {panel["AIHeat"].min():.3f} to {panel["AIHeat"].max():.3f}')
print(f'  AIHeat mean: {panel["AIHeat"].mean():.3f}')

# Import HAMR modules
from src import (template_cluster, quality, residual, funding_vacuum, 
                 hamr_factor, ic_test, backtest, visualization)
from src.data_process import cross_rankpct

# Step 3: Template Mismatch
print('\n[3/10] Computing Template Mismatch Score...')
mismatch_df = template_cluster.compute_template_affinity(panel)
print(f'  Mismatch range: {mismatch_df["MismatchScore"].min():.3f} to {mismatch_df["MismatchScore"].max():.3f}')

# Step 4: Quality Score
print('\n[4/10] Computing Quality Score...')
quality_df = quality.compute_quality_score(panel)
quality_col = 'QualityScore' if 'QualityScore' in quality_df.columns else quality_df.columns[-1]
print(f'  Quality range: {quality_df[quality_col].min():.3f} to {quality_df[quality_col].max():.3f}')

# Step 5: Residual Weakness
print('\n[5/10] Computing Residual Weakness...')
residual_df = residual.compute_residual_weakness(panel)
residual_col = 'ResidualWeakness' if 'ResidualWeakness' in residual_df.columns else residual_df.columns[-1]
non_fund_df = residual.compute_non_fundamental_ok(panel)
non_fund_col = 'NonFundamentalOK' if 'NonFundamentalOK' in non_fund_df.columns else non_fund_df.columns[-1]
print(f'  Residual range: {residual_df[residual_col].min():.3f} to {residual_df[residual_col].max():.3f}')

# Step 6: Funding Vacuum
print('\n[6/10] Computing Funding Vacuum...')
funding_df = funding_vacuum.compute_funding_vacuum(panel)
funding_col = 'FundingVacuum' if 'FundingVacuum' in funding_df.columns else funding_df.columns[-1]
print(f'  Funding range: {funding_df[funding_col].min():.3f} to {funding_df[funding_col].max():.3f}')

# Step 7: Compute Mispricing Pressure
print('\n[7/10] Computing Mispricing Pressure...')
mispricing_df = mismatch_df[['date', 'code', 'MismatchScore']].copy()
mispricing_df = mispricing_df.merge(residual_df[['date', 'code', residual_col]], on=['date', 'code'], how='left')
mispricing_df = mispricing_df.merge(non_fund_df[['date', 'code', non_fund_col]], on=['date', 'code'], how='left')
mispricing_df['MispricingPressure'] = (
    mispricing_df['MismatchScore'] * 
    mispricing_df[residual_col] * 
    mispricing_df[non_fund_col]
).clip(0, 1)
print(f'  Mispricing range: {mispricing_df["MispricingPressure"].min():.3f} to {mispricing_df["MispricingPressure"].max():.3f}')

# Step 8: Liquidity and Trap Guard
print('\n[8/10] Computing Liquidity & Trap Guard...')
# Simple liquidity: dollar volume rank
panel['LiquidityCapacity'] = cross_rankpct(panel, 'dollar_volume') if 'dollar_volume' in panel.columns else 0.5
# Simple trap guard: avoid limit-up/limit-down
panel['TrapGuard'] = 1.0
if 'is_limit_hit' in panel.columns:
    panel.loc[panel['is_limit_hit'] == 1, 'TrapGuard'] = 0.0
liquidity_df = panel[['date', 'code', 'LiquidityCapacity']].copy()
trap_df = panel[['date', 'code', 'TrapGuard']].copy()
print(f'  Liquidity range: {liquidity_df["LiquidityCapacity"].min():.3f} to {liquidity_df["LiquidityCapacity"].max():.3f}')

# Step 9: Assemble HAMR
print('\n[9/10] Assembling HAMR Factor...')
ai_state_df = panel[['date', 'AIHeat']].drop_duplicates().rename(columns={'AIHeat': 'AIStateScore'})

hamr = hamr_factor.compute_hamr(
    panel=panel,
    ai_state=ai_state_df,
    mismatch=mismatch_df,
    quality=quality_df,
    mispricing=mispricing_df,
    vacuum=funding_df,
    liquidity=liquidity_df,
    trap_guard=trap_df,
    verbose=True
)
print(f'  HAMR_Final range: {hamr["HAMR_Final"].min():.3f} to {hamr["HAMR_Final"].max():.3f}')

# Step 10: IC Test
print('\n[10/10] IC Test (All Horizons)...')
print('\n' + '-'*65)
print('  Horizon | IC Mean | ICIR  | NW t  | Signif')
print('-'*65)

# Add forward returns for IC testing
for h in [1, 5, 10, 20, 60]:
    col = f'ret_{h}d'
    if col in hamr.columns:
        hamr[f'fwd_{h}d'] = hamr.groupby('code')[col].shift(-h)

results = []
for h in [1, 5, 10, 20, 60]:
    ret_col = f'fwd_{h}d'
    if ret_col in hamr.columns:
        try:
            ic_df = ic_test.compute_daily_rank_ic(hamr, 'hamr_zscore', ret_col)
            stats = ic_test.compute_ic_statistics(ic_df['ic'].values)
            sig = '***' if abs(stats['nw_tstat']) > 2.58 else '**' if abs(stats['nw_tstat']) > 1.96 else '*' if abs(stats['nw_tstat']) > 1.65 else ''
            print(f'  {h:5d}d  | {stats["ic_mean"]:+7.4f} | {stats["icir"]:+5.2f} | {stats["nw_tstat"]:+5.2f} | {sig}')
            results.append({'horizon': h, **stats})
        except Exception as e:
            print(f'  {h:5d}d  | Error: {str(e)[:30]}')
    else:
        print(f'  {h:5d}d  | No forward return column')

print('-'*65)

# Summary
print('\n' + '='*65)
print('  SUMMARY')
print('='*65)

if results:
    best = max(results, key=lambda x: abs(x['nw_tstat']))
    print(f'  Best horizon: {best["horizon"]}d (NW-t={best["nw_tstat"]:+.2f})')
    
    # Check if any are significant
    sig_results = [r for r in results if abs(r['nw_tstat']) > 1.96]
    if sig_results:
        print(f'  Significant results: {len(sig_results)}/{len(results)}')
        for r in sig_results:
            direction = 'POSITIVE' if r['ic_mean'] > 0 else 'NEGATIVE'
            print(f'    {r["horizon"]}d: {direction} IC={r["ic_mean"]:+.4f} (NW-t={r["nw_tstat"]:+.2f})')
    else:
        print('  No significant results found.')

# Save results
hamr.to_pickle(ROOT / 'data' / 'hamr_results.pkl')
print(f'\n  Results saved to data/hamr_results.pkl')

print('\nDone!')
