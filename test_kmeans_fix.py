"""Quick test: fixed KMeans IC comparison"""
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

cache = pd.read_pickle(ROOT / 'data' / '_panel_200stocks.pkl')
cache['date'] = pd.to_datetime(cache['date'])

from src.template_cluster import compute_template_affinity
from src.quality import compute_quality_score
from src.residual import compute_residual_weakness, compute_non_fundamental_ok
from src.funding_vacuum import compute_funding_vacuum, compute_liquidity_capacity, compute_trap_guard
from src.ai_heat import compute_ai_heat_from_panel
from src.hamr_factor import compute_hamr, compute_forward_returns, add_hamr_diagnostic_variants
from src.ic_test import full_ic_analysis

print("Testing FIXED KMeans template clustering...")
try:
    mismatch = compute_template_affinity(cache, n_clusters=3,
                                          recent_only=False, recent_days=20,
                                          lookback_days=20)
    ms_min = mismatch["MismatchScore"].min()
    ms_max = mismatch["MismatchScore"].max()
    def_ratio = (mismatch["TemplateAffinity"].round(6) == 0.5).mean()
    print(f"  KMeans OK - MismatchScore: [{ms_min:.3f}, {ms_max:.3f}]")
    print(f"  TemplateAffinity default ratio: {def_ratio:.1%}")
except Exception as e:
    print(f"  KMeans failed: {e}")
    from src.template_cluster import compute_template_affinity
    mismatch = compute_template_affinity(cache, n_clusters=3, recent_only=False, recent_days=20, lookback_days=20)
    print("  Fallback to proxy")

qual = compute_quality_score(cache)
res_weak = compute_residual_weakness(cache)
nonfund = compute_non_fundamental_ok(cache)

mispricing = mismatch[["date","code","MismatchScore"]].merge(
    res_weak[["date","code","ResidualWeakness"]], on=["date","code"], how="inner"
).merge(nonfund[["date","code","NonFundamentalOK"]], on=["date","code"], how="inner")
mispricing["MispricingPressure"] = (
    mispricing["MismatchScore"].fillna(0.5)
    * mispricing["ResidualWeakness"].fillna(0.5)
    * mispricing["NonFundamentalOK"].fillna(0.5)
).clip(0, 1)

vacuum = compute_funding_vacuum(cache)
liq = compute_liquidity_capacity(cache)
trap = compute_trap_guard(qual, liq)
ai_state = compute_ai_heat_from_panel(cache)

panel = compute_hamr(cache, ai_state, mismatch, qual, mispricing, vacuum, liq, trap)
panel = add_hamr_diagnostic_variants(panel)
panel = compute_forward_returns(panel, horizons=(1, 5, 10, 20, 60, 120))

ic = full_ic_analysis(panel, factor_col="hamr_zscore",
                       horizons=(1, 5, 10, 20, 60, 120))

print(f"\n=== FIXED KMeans IC ===")
print(f"{'h':>5s}  {'IC Mean':>8s}  {'ICIR':>7s}  {'NW t':>7s}  {'IC>0':>6s}  Verdict")
for h in ["1","5","10","20","60","120"]:
    if h in ic:
        s = ic[h]["stats"]
        print(f"{h:>4s}d  {s['ic_mean']:+.4f}   {s['icir']:+.3f}   "
              f"{s['nw_tstat']:+.3f}   {s['ic_pos_ratio']:.1%}   {s['verdict']}")
