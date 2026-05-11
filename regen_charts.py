"""Regenerate key charts with FIXED KMeans for README"""
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.template_cluster import compute_template_affinity
from src.quality import compute_quality_score
from src.residual import compute_residual_weakness, compute_non_fundamental_ok
from src.funding_vacuum import compute_funding_vacuum, compute_liquidity_capacity, compute_trap_guard
from src.ai_heat import compute_ai_heat_from_panel
from src.hamr_factor import compute_hamr, compute_forward_returns
from src.ic_test import full_ic_analysis
from src.backtest import quintile_test

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

ic_results = full_ic_analysis(panel, factor_col='hamr_zscore', horizons=(1, 5, 10, 20, 60, 120))

OUT = ROOT / 'results' / 'figures'
C = {'blue': '#2E86AB', 'orange': '#A23B72', 'green': '#2A9D4F', 'red': '#E76F51', 'purple': '#6A4C93', 'teal': '#198C8C', 'dark': '#264653'}
HORIZON_COLOURS = {'1': C['blue'], '5': C['teal'], '10': C['orange'], '20': C['red'], '60': C['purple'], '120': C['green']}

# ── 1. quintile_returns.png (20d) ─────────────────────────────────
print('Generating quintile_returns.png...')
grp = quintile_test(panel[['date', 'fwd_20d', 'hamr_zscore']].dropna(),
                    factor_col='hamr_zscore', return_col='fwd_20d', n_groups=5)

fig, ax = plt.subplots(figsize=(10, 6))
q_labels = [f'Q{i}' for i in range(1, 6)]
returns = [grp['group_means'].get(q, 0) for q in q_labels]
colours = ['#d73027', '#fc8d59', '#ffffbf', '#91bfdb', '#4575b4']
bars = ax.bar(q_labels, returns, color=colours, alpha=0.9, width=0.6, edgecolor='white', linewidth=0.8)
for bar, val in zip(bars, returns):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.001, f'{val:+.3%}', ha='center', fontsize=11, fontweight='bold')
spread = grp['spread']
ax.annotate(f'Q5-Q1 = {spread:+.3%}', xy=(4, returns[4]), xytext=(4.5, returns[4] * 1.5),
            fontsize=12, fontweight='bold', color=C['green'] if spread > 0 else C['red'],
            arrowprops=dict(arrowstyle='->', color=C['dark'], lw=1.5))
mono = grp.get('monotonicity', 0)
ax.set_title(f'HAMR Quintile Portfolio Returns (20d Forward)\nMonotonicity p = {mono:+.3f}  -  Q5-Q1 = {spread:+.3%}',
             fontweight='bold', fontsize=14)
ax.set_ylabel('Portfolio Return')
ax.axhline(y=0, color='grey', linewidth=0.5)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
ax.grid(True, alpha=0.25, axis='y')
plt.tight_layout()
fig.savefig(OUT / 'quintile_returns.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f'  Q5-Q1 = {spread:+.3%}, Monotonicity = {mono:+.3f}')

# ── 2. ic_timeseries_20d.png ──────────────────────────────────────
print('Generating ic_timeseries_20d.png...')
ic20 = ic_results['20']['ic_series'].copy()
s = ic_results['20']['stats']
fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={'height_ratios': [3, 1]})
ax = axes[0]
dates = pd.to_datetime(ic20['date'])
ic_vals = ic20['ic'].values
bar_colours = [C['blue'] if v > 0 else C['red'] for v in ic_vals]
ax.bar(dates, ic_vals, color=bar_colours, width=1.0, alpha=0.7, linewidth=0)
roll = pd.Series(ic_vals).rolling(20, min_periods=5).mean()
ax.plot(dates, roll, color=C['dark'], linewidth=1.8, label='20d Rolling Mean')
ax.axhline(y=0, color='grey', linewidth=0.5, linestyle='--')
ax.axhline(y=s['ic_mean'], color=C['green'], linewidth=1.2, linestyle='-', alpha=0.6,
           label=f"Mean IC = {s['ic_mean']:+.4f}")
ax.set_ylabel('Rank IC')
ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
ax.set_title('HAMR - Daily Rank IC (20d Forward)', fontweight='bold', fontsize=15)
ax2 = axes[1]
cum = ic_vals.cumsum()
ax2.fill_between(dates, cum, 0, where=(cum >= 0), color=C['blue'], alpha=0.25)
ax2.fill_between(dates, cum, 0, where=(cum < 0), color=C['red'], alpha=0.25)
ax2.plot(dates, cum, color=C['dark'], linewidth=1.2)
ax2.set_ylabel('Cumulative IC')
ax2.set_xlabel('Date')
text = (f"Mean IC = {s['ic_mean']:+.4f}\n"
        f"ICIR = {s['icir']:+.3f}\n"
        f"NW t = {s['nw_tstat']:+.2f}\n"
        f"IC>0 = {s['ic_pos_ratio']:.1%}")
ax2.text(0.98, 0.05, text, transform=ax2.transAxes, fontsize=9,
         verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))
for a in axes: a.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(OUT / 'ic_timeseries_20d.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"  IC={s['ic_mean']:+.4f}, ICIR={s['icir']:+.3f}")

# ── 3. ic_distribution_20d.png ────────────────────────────────────
print('Generating ic_distribution_20d.png...')
fig, ax = plt.subplots(figsize=(10, 6))
n, bins, patches = ax.hist(ic_vals, bins=25, color=C['blue'], alpha=0.65, edgecolor='white', density=True)
mu, sigma = np.mean(ic_vals), np.std(ic_vals)
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
ax.plot(x, stats.norm.pdf(x, mu, sigma), color=C['red'], linewidth=2, linestyle='--',
        label=f'N({mu:.3f}, {sigma:.3f})')
ax.axvline(x=0, color='grey', linewidth=0.8, linestyle='-')
ax.axvline(x=mu, color=C['green'], linewidth=1.5, label=f'Mean = {mu:+.4f}')
ax.set_xlabel('Rank IC')
ax.set_ylabel('Density')
ax.set_title('HAMR - IC Distribution (20d Forward)', fontweight='bold', fontsize=15)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.25)
plt.tight_layout()
fig.savefig(OUT / 'ic_distribution_20d.png', dpi=300, bbox_inches='tight', facecolor='white')

# ── 4. ic_decay_analysis.png ──────────────────────────────────────
print('Generating ic_decay_analysis.png...')
HORIZONS = ['1', '5', '10', '20', '60', '120']
labels = [f'{h}d' for h in HORIZONS]
ic_means = [ic_results[h]['stats']['ic_mean'] for h in HORIZONS]
icirs = [ic_results[h]['stats']['icir'] for h in HORIZONS]
nw_ts = [ic_results[h]['stats']['nw_tstat'] for h in HORIZONS]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
x = np.arange(len(HORIZONS))

ax = axes[0]
ax.bar(x, ic_means, color=[HORIZON_COLOURS[h] for h in HORIZONS], alpha=0.85, width=0.6)
for i, v in enumerate(ic_means):
    ax.text(i, v + 0.002 * (1 if v > 0 else -1), f'{v:+.3f}', ha='center', fontsize=9, fontweight='bold')
ax.axhline(y=0, color='grey', linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('Mean IC'); ax.set_title('IC Mean', fontweight='bold')
ax.grid(True, alpha=0.25)

ax = axes[1]
ax.bar(x, icirs, color=[HORIZON_COLOURS[h] for h in HORIZONS], alpha=0.85, width=0.6)
for i, v in enumerate(icirs):
    ax.text(i, v + 0.02, f'{v:+.3f}', ha='center', fontsize=9, fontweight='bold')
ax.axhline(y=0, color='grey', linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('ICIR'); ax.set_title('IC Information Ratio', fontweight='bold')
ax.grid(True, alpha=0.25)

ax = axes[2]
bar_cols = [C['green'] if v > 1.96 else ('#E9C46A' if abs(v) > 1.28 else C['red']) for v in nw_ts]
ax.bar(x, nw_ts, color=bar_cols, alpha=0.85, width=0.6)
for i, v in enumerate(nw_ts):
    ax.text(i, v + 0.15 * (1 if v > 0 else -1), f'{v:+.1f}', ha='center', fontsize=9, fontweight='bold')
ax.axhline(y=1.96, color=C['green'], linewidth=0.8, linestyle='--', alpha=0.6, label='p=0.05')
ax.axhline(y=-1.96, color=C['green'], linewidth=0.8, linestyle='--', alpha=0.6)
ax.axhline(y=0, color='grey', linewidth=0.5)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('NW t-stat'); ax.set_title('Newey-West t-stat', fontweight='bold')
ax.legend(fontsize=7); ax.grid(True, alpha=0.25)

fig.suptitle('HAMR - IC Decay Across Horizons', fontweight='bold', fontsize=16, y=1.01)
plt.tight_layout()
fig.savefig(OUT / 'ic_decay_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')

print('\nAll README charts regenerated with FIXED KMeans!')
print('Now push to GitHub to update the website.')
