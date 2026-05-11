"""
test_long_horizon.py — Recompute HAMR from cache + test 60d/120d IC
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.ic_test import full_ic_analysis
from src.data_process import cross_rankpct

# Load cache
cache_path = ROOT / 'data' / '_panel_200stocks.pkl'
print(f'Loading: {cache_path}')
cache = pd.read_pickle(cache_path)
cache['date'] = pd.to_datetime(cache['date'])
print(f'  {cache["code"].nunique()} stocks x {cache["date"].nunique()} dates')

# ================================================================
# Step 1: Compute all sub-components (on cache panel)
# ================================================================
print('\n--- Computing all sub-components ---')

# Template Mismatch
print('  MismatchScore...')
from src.template_cluster import compute_template_affinity
mismatch = compute_template_affinity(cache, n_clusters=3, recent_only=False, recent_days=20, lookback_days=20)

# QualityScore
print('  QualityScore...')
from src.quality import compute_quality_score
qual = compute_quality_score(cache)

# ResidualWeakness + NonFundamentalOK
print('  ResidualWeakness + NonFundamentalOK...')
from src.residual import compute_residual_weakness, compute_non_fundamental_ok
res_weak = compute_residual_weakness(cache)
nonfund = compute_non_fundamental_ok(cache)

# MispricingPressure (merge mismatch, residual, nonfund)
mispricing = mismatch[['date','code','MismatchScore']].merge(
    res_weak[['date','code','ResidualWeakness']], on=['date','code'], how='inner'
).merge(nonfund[['date','code','NonFundamentalOK']], on=['date','code'], how='inner')
mispricing['MispricingPressure'] = (
    mispricing['MismatchScore'].fillna(0.5) *
    mispricing['ResidualWeakness'].fillna(0.5) *
    mispricing['NonFundamentalOK'].fillna(0.5)
).clip(0, 1)

# FundingVacuum + Liquidity + TrapGuard
print('  FundingVacuum...')
from src.funding_vacuum import compute_funding_vacuum, compute_liquidity_capacity, compute_trap_guard
vacuum = compute_funding_vacuum(cache)
liq = compute_liquidity_capacity(cache)
trap = compute_trap_guard(qual, liq)

# AIStateScore
print('  AIStateScore...')
from src.ai_heat import compute_ai_heat_from_panel
ai_state = compute_ai_heat_from_panel(cache)

# ================================================================
# Step 2: Assemble HAMR factor
# ================================================================
print('\n--- Assembling HAMR ---')
from src.hamr_factor import compute_hamr, add_hamr_diagnostic_variants
panel = compute_hamr(cache, ai_state, mismatch, qual, mispricing, vacuum, liq, trap)
panel = add_hamr_diagnostic_variants(panel)

print(f'  HAMR_Final range: [{panel["HAMR_Final"].min():.4f}, {panel["HAMR_Final"].max():.4f}]')
print(f'  Columns: {len(panel.columns)}')

# ================================================================
# Step 3: Compute forward returns ONCE (now panel has all factor cols)
# ================================================================
print('\n--- Computing forward returns (1d..120d) ---')
from src.hamr_factor import compute_forward_returns
panel = compute_forward_returns(panel, horizons=(1, 5, 10, 20, 60, 120))
for h in [1,5,10,20,60,120]:
    n = panel[f'fwd_{h}d'].notna().sum()
    print(f'  fwd_{h}d: {n} valid')

# ================================================================
# IC Analysis
# ================================================================
print(f'\n{"="*70}')
print('  HAMR Long-Horizon IC Analysis')
print(f'{"="*70}')

# 1. Baseline HAMR
print(f'\n--- 1. Baseline HAMR (hamr_zscore) ---')
ic_baseline = full_ic_analysis(panel, factor_col='hamr_zscore',
                                horizons=(1, 5, 10, 20, 60, 120))

print(f'\n  {"Horizon":>6s} | {"IC Mean":>8s} | {"ICIR":>7s} | {"NW t":>7s} | {"IC>0":>6s} | {"N":>4s} | {"Verdict":>12s}')
print(f'  {"-"*72}')
for h in ['1','5','10','20','60','120']:
    if h in ic_baseline:
        s = ic_baseline[h]['stats']
        n = len(ic_baseline[h]['ic_series'])
        print(f'  {h:>4s}d  | {s["ic_mean"]:+.4f}  | {s["icir"]:+.3f}  | '
              f'{s["nw_tstat"]:+.3f}  | {s["ic_pos_ratio"]:.1%}  | {n:>3d} | {s["verdict"]}')
    else:
        print(f'  {h:>4s}d  | (no valid data)')

# 2. Diagnostic variants
print(f'\n--- 2. Diagnostic Variant Comparison ---')
print(f'\n  {"Factor":>25s} | {"20d IC":>8s} | {"60d IC":>8s} | {"120d IC":>8s} | {"60d NW t":>9s} | {"120d NW t":>9s}')
print(f'  {"-"*85}')

diagnostic_factors = [
    'hamr_zscore','HAMR_Core_Raw_rank','HAMR_NoAI_rank',
    'HAMR_Diag_Final_rank','HAMR_Entry_rank','HAMR_ReverseCheck_rank'
]
for fac in diagnostic_factors:
    if fac not in panel.columns:
        continue
    try:
        ic20 = full_ic_analysis(panel, factor_col=fac, horizons=(20,))
        ic60 = full_ic_analysis(panel, factor_col=fac, horizons=(60,))
        ic120 = full_ic_analysis(panel, factor_col=fac, horizons=(120,))
        s20 = ic20['20']['stats'] if '20' in ic20 else {'ic_mean':0}
        s60 = ic60['60']['stats'] if '60' in ic60 else {'ic_mean':0,'nw_tstat':0}
        s120 = ic120['120']['stats'] if '120' in ic120 else {'ic_mean':0,'nw_tstat':0}
        if '20' in ic20:
            print(f'  {fac:>25s} | {s20["ic_mean"]:+.4f}  | {s60["ic_mean"]:+.4f}  | '
                  f'{s120["ic_mean"]:+.4f}  | {s60["nw_tstat"]:+.3f}     | {s120["nw_tstat"]:+.3f}')
        else:
            print(f'  {fac:>25s} | ----')
    except Exception as e:
        print(f'  {fac:>25s} | skipped: {type(e).__name__}: {str(e)[:50]}')

# 3. Sub-component IC
print(f'\n--- 3. Sub-Component IC ---')
print(f'\n  {"Component":>25s} | {"20d IC":>8s} | {"60d IC":>8s} | {"120d IC":>8s} | {"60d ICIR":>8s} | {"120d ICIR":>8s}')
print(f'  {"-"*85}')
for comp in ['MismatchScore','QualityScore','ResidualWeakness',
             'NonFundamentalOK','FundingVacuum','LiquidityCapacity',
             'MispricingPressure','HAMR_Core','HAMR_Final']:
    if comp not in panel.columns:
        continue
    try:
        vals = {}
        for horizon in [20, 60, 120]:
            ret_col = f'fwd_{horizon}d'
            valid = panel[[comp, ret_col, 'date']].dropna()
            ics = []
            for _, grp in valid.groupby('date'):
                if len(grp) < 10: continue
                ic, _ = stats.spearmanr(grp[comp], grp[ret_col])
                if np.isfinite(ic): ics.append(ic)
            ics = np.array(ics)
            vals[horizon] = {
                'ic_mean': float(ics.mean()) if len(ics)>0 else 0,
                'icir': float(ics.mean()/ics.std()) if len(ics)>1 and ics.std()>0 else 0
            }
        print(f'  {comp:>25s} | {vals[20]["ic_mean"]:+.4f}  | {vals[60]["ic_mean"]:+.4f}  | '
              f'{vals[120]["ic_mean"]:+.4f}  | {vals[60]["icir"]:+.3f}  | {vals[120]["icir"]:+.3f}')
    except Exception as e:
        print(f'  {comp:>25s} | skipped: {type(e).__name__}')

print(f'\n{"="*70}')
print('  Done.')
print(f'{"="*70}')
