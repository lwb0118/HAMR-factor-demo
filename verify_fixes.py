"""Quick verification: all fixes produce positive IC"""
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.template_cluster import compute_template_affinity
from src.quality import compute_quality_score
from src.residual import compute_residual_weakness, compute_non_fundamental_ok
from src.funding_vacuum import compute_funding_vacuum, compute_liquidity_capacity, compute_trap_guard
from src.ai_heat import compute_ai_heat_from_panel
from src.hamr_factor import compute_hamr, compute_forward_returns
from src.ic_test import full_ic_analysis

cache = pd.read_pickle(ROOT / 'data' / '_panel_200stocks.pkl')
cache['date'] = pd.to_datetime(cache['date'])

mismatch = compute_template_affinity(cache, n_clusters=3, recent_only=False, recent_days=20, lookback_days=20)
qual = compute_quality_score(cache)
res_weak = compute_residual_weakness(cache)
nonfund = compute_non_fundamental_ok(cache)
mispricing = mismatch[['date','code','MismatchScore']].merge(
    res_weak[['date','code','ResidualWeakness']], on=['date','code'], how='inner'
).merge(nonfund[['date','code','NonFundamentalOK']], on=['date','code'], how='inner')
mispricing['MispricingPressure'] = (
    mispricing['MismatchScore'].fillna(0.5)
    * mispricing['ResidualWeakness'].fillna(0.5)
    * mispricing['NonFundamentalOK'].fillna(0.5)
).clip(0, 1)

vacuum = compute_funding_vacuum(cache)
liq = compute_liquidity_capacity(cache)
trap = compute_trap_guard(qual, liq)
ai_state = compute_ai_heat_from_panel(cache)

panel = compute_hamr(cache, ai_state, mismatch, qual, mispricing, vacuum, liq, trap)
panel = compute_forward_returns(panel, horizons=(1, 5, 10, 20, 60, 120))

ic = full_ic_analysis(panel, factor_col='hamr_zscore', horizons=(20, 60, 120))

print("=== Verification: Post-Fix IC ===")
for h in ['20', '60', '120']:
    s = ic[h]['stats']
    ok = "PASS" if s['ic_mean'] > 0 else "FAIL"
    print(f"  {h}d: IC={s['ic_mean']:+.4f}  ICIR={s['icir']:+.3f}  NWt={s['nw_tstat']:+.2f}  [{ok}]")
print("\nAll checks passed!" if all(ic[h]['stats']['ic_mean'] > 0 for h in ['20','60','120']) else "\nWARNING: negative IC detected!")
