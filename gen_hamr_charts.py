"""
gen_hamr_charts.py — HAMR 5-120d IC + Quintile charts
======================================================
Generates:
  1. IC timeseries per horizon (5d, 10d, 20d, 60d, 120d)
  2. IC distribution per horizon
  3. IC decay summary
  4. Quintile return bar chart per horizon
  5. Combined dashboard
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch

# 中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.size': 11, 'axes.titlesize': 14, 'axes.labelsize': 12,
})

OUT_DIR = ROOT / 'results' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── colour palette ─────────────────────────────────────────────────
C = {
    'blue':   '#2E86AB',
    'orange': '#A23B72',
    'green':  '#2A9D4F',
    'red':    '#E76F51',
    'purple': '#6A4C93',
    'teal':   '#198C8C',
    'dark':   '#264653',
    'grey':   '#8D99AE',
    'light':  '#E9ECEF',
}
HORIZON_COLOURS = {'5': C['blue'], '10': C['teal'], '20': C['orange'],
                   '60': C['red'], '120': C['purple']}

# ═══════════════════════════════════════════════════════════════════
#  Data
# ═══════════════════════════════════════════════════════════════════
print('Loading & computing HAMR factor...')

cache = pd.read_pickle(ROOT / 'data' / '_panel_200stocks.pkl')
cache['date'] = pd.to_datetime(cache['date'])

from src.template_cluster import compute_template_affinity
from src.quality import compute_quality_score
from src.residual import compute_residual_weakness, compute_non_fundamental_ok
from src.funding_vacuum import compute_funding_vacuum, compute_liquidity_capacity, compute_trap_guard
from src.ai_heat import compute_ai_heat_from_panel
from src.hamr_factor import compute_hamr, add_hamr_diagnostic_variants, compute_forward_returns
from src.ic_test import full_ic_analysis, compute_daily_rank_ic, compute_ic_statistics

mismatch = compute_template_affinity(cache, n_clusters=3, recent_only=False, recent_days=20, lookback_days=20)
qual = compute_quality_score(cache)
res_weak = compute_residual_weakness(cache)
nonfund = compute_non_fundamental_ok(cache)

mispricing = mismatch[['date','code','MismatchScore']].merge(
    res_weak[['date','code','ResidualWeakness']], on=['date','code'], how='inner'
).merge(nonfund[['date','code','NonFundamentalOK']], on=['date','code'], how='inner')
mispricing['MispricingPressure'] = (
    mispricing['MismatchScore'].fillna(0.5) *
    mispricing['ResidualWeakness'].fillna(0.5) *
    mispricing['NonFundamentalOK'].fillna(0.5)
).clip(0, 1)

vacuum = compute_funding_vacuum(cache)
liq = compute_liquidity_capacity(cache)
trap = compute_trap_guard(qual, liq)
ai_state = compute_ai_heat_from_panel(cache)

panel = compute_hamr(cache, ai_state, mismatch, qual, mispricing, vacuum, liq, trap)
panel = add_hamr_diagnostic_variants(panel)
panel = compute_forward_returns(panel, horizons=(1, 5, 10, 20, 60, 120))

print(f'  Panel: {panel["code"].nunique()} stocks × {panel["date"].nunique()} dates')
print(f'  HAMR_Final range: [{panel["HAMR_Final"].min():.4f}, {panel["HAMR_Final"].max():.4f}]')

HORIZONS = ['5', '10', '20', '60', '120']

# ── compute all IC data ────────────────────────────────────────────
ic_results = full_ic_analysis(panel, factor_col='hamr_zscore',
                               horizons=(5, 10, 20, 60, 120))

print('\nIC results:')
for h in HORIZONS:
    s = ic_results[h]['stats']
    print(f'  {h}d: IC={s["ic_mean"]:+.4f}  ICIR={s["icir"]:+.3f}  NW-t={s["nw_tstat"]:+.3f}  |IC|>0={s["ic_pos_ratio"]:.0%}  n={len(ic_results[h]["ic_series"])}')


def _save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f'  Saved: {name}')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  1 — IC Timeseries (one chart per horizon)
# ═══════════════════════════════════════════════════════════════════
print('\n[1/5] IC Timeseries...')

for h in HORIZONS:
    ic_df = ic_results[h]['ic_series'].copy()
    s = ic_results[h]['stats']
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 7),
                              gridspec_kw={'height_ratios': [3, 1]})
    
    # —— upper: IC bars + rolling mean ——
    ax = axes[0]
    dates = pd.to_datetime(ic_df['date'])
    ic_vals = ic_df['ic'].values
    bar_colours = [C['blue'] if v > 0 else C['red'] for v in ic_vals]
    ax.bar(dates, ic_vals, color=bar_colours, width=1.0, alpha=0.7, linewidth=0)
    
    # rolling mean (20d)
    roll = pd.Series(ic_vals).rolling(20, min_periods=5).mean()
    ax.plot(dates, roll, color=C['dark'], linewidth=1.8, label='20d Rolling Mean')
    
    ax.axhline(y=0, color='grey', linewidth=0.5, linestyle='--')
    ax.axhline(y=s['ic_mean'], color=C['green'], linewidth=1.2,
               linestyle='-', alpha=0.6,
               label=f"Mean IC = {s['ic_mean']:+.4f}")
    
    ax.set_ylabel('Rank IC')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax.set_title(f'HAMR — Daily Rank IC ({h}d Forward)', fontweight='bold',
                 fontsize=15)
    
    # —— lower: cumulative IC ——
    ax2 = axes[1]
    cum = ic_vals.cumsum()
    ax2.fill_between(dates, cum, 0, where=(cum >= 0), color=C['blue'], alpha=0.25)
    ax2.fill_between(dates, cum, 0, where=(cum < 0), color=C['red'], alpha=0.25)
    ax2.plot(dates, cum, color=C['dark'], linewidth=1.2)
    ax2.set_ylabel('Cumulative IC')
    ax2.set_xlabel('Date')
    
    # stats box
    text = (f"Mean IC = {s['ic_mean']:+.4f}\n"
            f"ICIR = {s['icir']:+.3f}\n"
            f"NW t = {s['nw_tstat']:+.2f}\n"
            f"|IC|>0 = {s['ic_pos_ratio']:.1%}")
    ax2.text(0.98, 0.05, text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))
    
    for a in axes:
        a.grid(True, alpha=0.3)
    
    plt.tight_layout()
    _save(fig, f'ic_timeseries_{h}d.png')


# ═══════════════════════════════════════════════════════════════════
#  2 — IC Distribution (one chart per horizon)
# ═══════════════════════════════════════════════════════════════════
print('[2/5] IC Distribution...')

for h in HORIZONS:
    ic_vals = ic_results[h]['ic_series']['ic'].values
    s = ic_results[h]['stats']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n, bins, patches = ax.hist(ic_vals, bins=25, color=C['blue'],
                                alpha=0.65, edgecolor='white', density=True)
    
    # normal fit
    mu, sigma = np.mean(ic_vals), np.std(ic_vals)
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), color=C['red'],
            linewidth=2, linestyle='--', label=f'N({mu:.3f}, {sigma:.3f})')
    
    ax.axvline(x=0, color='grey', linewidth=0.8, linestyle='-')
    ax.axvline(x=mu, color=C['green'], linewidth=1.5,
               label=f'Mean = {mu:+.4f}')
    
    ax.set_xlabel('Rank IC')
    ax.set_ylabel('Density')
    ax.set_title(f'HAMR — IC Distribution ({h}d Forward)', fontweight='bold',
                 fontsize=15)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    
    plt.tight_layout()
    _save(fig, f'ic_distribution_{h}d.png')


# ═══════════════════════════════════════════════════════════════════
#  3 — IC Decay (IC Mean + ICIR + NW-t across horizons)
# ═══════════════════════════════════════════════════════════════════
print('[3/5] IC Decay...')

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

labels = [f'{h}d' for h in HORIZONS]
ic_means = [ic_results[h]['stats']['ic_mean'] for h in HORIZONS]
icirs    = [ic_results[h]['stats']['icir'] for h in HORIZONS]
nw_ts    = [ic_results[h]['stats']['nw_tstat'] for h in HORIZONS]
x = np.arange(len(HORIZONS))

# IC Mean
ax = axes[0]
bars = ax.bar(x, ic_means, color=[HORIZON_COLOURS[h] for h in HORIZONS],
              alpha=0.85, width=0.6)
for i, v in enumerate(ic_means):
    ax.text(i, v + 0.002 * (1 if v > 0 else -1),
            f'{v:+.3f}', ha='center', fontsize=9, fontweight='bold')
ax.axhline(y=0, color='grey', linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('Mean IC'); ax.set_title('IC Mean', fontweight='bold')
ax.grid(True, alpha=0.25)

# ICIR
ax = axes[1]
bars = ax.bar(x, icirs, color=[HORIZON_COLOURS[h] for h in HORIZONS],
              alpha=0.85, width=0.6)
for i, v in enumerate(icirs):
    ax.text(i, v + 0.02, f'{v:+.3f}', ha='center', fontsize=9, fontweight='bold')
ax.axhline(y=0, color='grey', linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('ICIR'); ax.set_title('IC Information Ratio', fontweight='bold')
ax.grid(True, alpha=0.25)

# NW t-stat
ax = axes[2]
bar_cols = [C['green'] if v > 1.96 else (C['orange'] if abs(v) > 1.28 else C['red'])
            for v in nw_ts]
ax.bar(x, nw_ts, color=bar_cols, alpha=0.85, width=0.6)
for i, v in enumerate(nw_ts):
    ax.text(i, v + 0.15 * (1 if v > 0 else -1),
            f'{v:+.1f}', ha='center', fontsize=9, fontweight='bold')
ax.axhline(y=1.96, color=C['green'], linewidth=0.8, linestyle='--', alpha=0.6, label='p=0.05')
ax.axhline(y=-1.96, color=C['green'], linewidth=0.8, linestyle='--', alpha=0.6)
ax.axhline(y=0, color='grey', linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('NW t-stat'); ax.set_title('Newey-West t-stat', fontweight='bold')
ax.legend(fontsize=7); ax.grid(True, alpha=0.25)

fig.suptitle('HAMR — IC Decay Across Horizons', fontweight='bold', fontsize=16, y=1.01)
plt.tight_layout()
_save(fig, 'ic_decay_analysis.png')


# ═══════════════════════════════════════════════════════════════════
#  4 — Quintile Returns (one chart per horizon)
# ═══════════════════════════════════════════════════════════════════
print('[4/5] Quintile Returns...')

from src.backtest import quintile_test

for h in HORIZONS:
    ret_col = f'fwd_{h}d'
    if ret_col not in panel.columns:
        print(f'  {h}d: no forward return col, skip')
        continue
    
    try:
        grp = quintile_test(panel[['date', ret_col, 'hamr_zscore']].dropna(),
                            factor_col='hamr_zscore', return_col=ret_col, n_groups=5)
    except Exception as e:
        print(f'  {h}d quintile: {e}')
        continue
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    q_labels = [f'Q{i}' for i in range(1, 6)]
    returns = [grp['group_means'].get(q, 0) for q in q_labels]
    stds    = [grp['group_stds'].get(q, 0) for q in q_labels]
    
    colours = ['#d73027', '#fc8d59', '#ffffbf', '#91bfdb', '#4575b4']
    bars = ax.bar(q_labels, returns, color=colours, alpha=0.9, width=0.6,
                  edgecolor='white', linewidth=0.8)
    
    # value labels
    for bar, val in zip(bars, returns):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.001,
                f'{val:+.3%}', ha='center', fontsize=11, fontweight='bold')
    
    # spread annotation
    spread = grp['spread']
    ax.annotate(f'Q5−Q1 = {spread:+.3%}',
                xy=(4, returns[4]), xytext=(4.5, returns[4] * 1.5),
                fontsize=12, fontweight='bold', color=C['green'] if spread > 0 else C['red'],
                arrowprops=dict(arrowstyle='->', color=C['dark'], lw=1.5))
    
    # monotonicity
    mono = grp.get('monotonicity', 0)
    ax.set_title(f'HAMR Quintile Portfolio Returns ({h}d Forward)\n'
                 f'Monotonicity ρ = {mono:+.3f}  ·  Q5−Q1 = {spread:+.3%}',
                 fontweight='bold', fontsize=14)
    
    ax.set_ylabel('Portfolio Return')
    ax.axhline(y=0, color='grey', linewidth=0.5)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(True, alpha=0.25, axis='y')
    
    plt.tight_layout()
    _save(fig, f'quintile_returns_{h}d.png')


# ═══════════════════════════════════════════════════════════════════
#  5 — Combined Dashboard
# ═══════════════════════════════════════════════════════════════════
print('[5/5] Dashboard...')

fig = plt.figure(figsize=(20, 14))

# ── (A) IC Decay strip ────────────────────────────────────────────
ax_a = fig.add_subplot(3, 4, (1, 2))
ax_a.bar(labels, ic_means,
         color=[HORIZON_COLOURS[h] for h in HORIZONS], alpha=0.85, width=0.6)
for i, v in enumerate(ic_means):
    ax_a.text(i, v + 0.002, f'{v:+.4f}', ha='center', fontsize=9, fontweight='bold')
ax_a.axhline(y=0, color='grey', linewidth=0.5)
ax_a.set_title('IC Mean Across Horizons', fontweight='bold')
ax_a.set_ylabel('Mean IC'); ax_a.grid(True, alpha=0.25)

# ── (B) ICIR Decay ────────────────────────────────────────────────
ax_b = fig.add_subplot(3, 4, (3, 4))
ax_b.bar(labels, icirs,
         color=[HORIZON_COLOURS[h] for h in HORIZONS], alpha=0.85, width=0.6)
for i, v in enumerate(icirs):
    ax_b.text(i, v + 0.02, f'{v:+.3f}', ha='center', fontsize=9, fontweight='bold')
ax_b.axhline(y=0, color='grey', linewidth=0.5)
ax_b.set_title('ICIR Across Horizons', fontweight='bold')
ax_b.set_ylabel('ICIR'); ax_b.grid(True, alpha=0.25)

# ── (C–G) Quintile charts 5d→120d ─────────────────────────────────
for idx, h in enumerate(HORIZONS):
    ret_col = f'fwd_{h}d'
    if ret_col not in panel.columns:
        continue
    
    try:
        grp = quintile_test(panel[['date', ret_col, 'hamr_zscore']].dropna(),
                            factor_col='hamr_zscore', return_col=ret_col, n_groups=5)
    except Exception:
        continue
    
    ax = fig.add_subplot(3, 5, 6 + idx)
    q_labels = [f'Q{i}' for i in range(1, 6)]
    returns = [grp['group_means'].get(q, 0) for q in q_labels]
    colours_q = ['#d73027', '#fc8d59', '#ffffbf', '#91bfdb', '#4575b4']
    
    ax.bar(q_labels, returns, color=colours_q, alpha=0.9, width=0.6)
    ax.axhline(y=0, color='grey', linewidth=0.5)
    spread = grp['spread']
    ax.set_title(f'{h}d (Q5−Q1={spread:+.2%})', fontsize=10, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(True, alpha=0.25, axis='y')
    
    # value label on highest bar
    max_ret = max(returns)
    max_idx = returns.index(max_ret)
    ax.text(max_idx, max_ret + 0.002, f'{max_ret:+.2%}', ha='center',
            fontsize=8, fontweight='bold')

# ── (H) Cumulative IC ─────────────────────────────────────────────
ax_h = fig.add_subplot(3, 4, (9, 12))
ic20 = ic_results['20']['ic_series'].copy()
dates = pd.to_datetime(ic20['date'])
cum = ic20['ic'].values.cumsum()
ax_h.fill_between(dates, cum, 0, where=(cum >= 0), color=C['green'], alpha=0.2)
ax_h.fill_between(dates, cum, 0, where=(cum < 0), color=C['red'], alpha=0.2)
ax_h.plot(dates, cum, color=C['dark'], linewidth=1.5)
ax_h.set_title('Cumulative IC (20d Forward)', fontweight='bold')
ax_h.set_ylabel('Cumulative IC'); ax_h.set_xlabel('Date')
ax_h.grid(True, alpha=0.25)

fig.suptitle('HAMR Factor — Full Horizon Analysis (5d–120d)',
             fontweight='bold', fontsize=18, y=1.01)
plt.tight_layout()
_save(fig, 'hamr_dashboard.png')

print(f'\nAll charts saved to: {OUT_DIR}')
