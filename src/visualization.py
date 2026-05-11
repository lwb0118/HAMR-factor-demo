"""
visualization.py
=================
Publication-quality charts for HAMR factor analysis.

Per HAMR Framework section 18:
  - AIHeat time series
  - Sub-component distributions
  - IC time series + cumulative IC
  - IC decay across horizons
  - IC distribution (with normal fit)
  - Quintile returns (with monotonicity)
  - HAMR vs control factor comparison
  - Factor correlation heatmap
  - Component attribution bar chart
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent
CHART_DIR = ROOT / 'results' / 'figures'
CHART_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
    'legend.fontsize': 9,
    'figure.facecolor': '#FAFAFA', 'axes.facecolor': '#FAFAFA',
    'axes.grid': True, 'grid.alpha': 0.25,
})

C = {
    'primary': '#1a5276', 'red': '#e74c3c', 'green': '#27ae60',
    'grey': '#7f8c8d', 'orange': '#f39c12',
    'q5': ['#2c3e50', '#3498db', '#2ecc71', '#f39c12', '#e74c3c'],
}


def _save(fig, name):
    path = CHART_DIR / name
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  {name}')


# -----------------------------------------------------------------------
# IC Charts
# -----------------------------------------------------------------------

def plot_ic_timeseries(ic_series, horizon, stats=None):
    """Daily IC bars + cumulative IC overlay."""
    dates = ic_series['date'].values
    ics = ic_series['ic'].values
    n = len(ics)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                     gridspec_kw={'height_ratios': [2, 1]})

    colors = [C['green'] if v > 0 else C['red'] for v in ics]
    ax1.bar(range(n), ics, color=colors, width=0.8, alpha=0.7)
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.axhline(y=np.mean(ics), color=C['primary'], linestyle='--',
                linewidth=1.5, label=f'Mean IC = {np.mean(ics):.4f}')
    ax1.set_ylabel('Rank IC')
    ax1.set_title(f'HAMR — Daily Rank IC ({horizon}d Forward)')
    ax1.legend(loc='upper right')

    ticks = np.linspace(0, n-1, min(10, n), dtype=int)
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([str(d)[:10] for d in dates[ticks]], rotation=45)

    cum = np.cumsum(ics)
    ax2.fill_between(range(n), 0, cum, color=C['primary'], alpha=0.3)
    ax2.plot(range(n), cum, color=C['primary'], linewidth=1.5,
             label=f'Cumul. IC = {cum[-1]:.2f}')
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_ylabel('Cumulative IC')
    ax2.set_xlabel('Trading Day')
    ax2.legend(loc='upper right')
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([str(d)[:10] for d in dates[ticks]], rotation=45)

    if stats:
        fig.suptitle(
            f'ICIR={stats.get("icir", 0):.3f} | '
            f'NW t={stats.get("nw_tstat", 0):.3f} | '
            f'IC>0={stats.get("ic_pos_ratio", 0):.1%}',
            y=1.02, fontsize=10, fontstyle='italic', color=C['grey']
        )

    plt.tight_layout()
    _save(fig, f'ic_timeseries_{horizon}d.png')


def plot_ic_decay(ic_results):
    """ICIR and NW t-stat across horizons."""
    horizons = sorted(ic_results.keys())
    icirs = [ic_results[h]['stats']['icir'] for h in horizons]
    nw_ts = [ic_results[h]['stats']['nw_tstat'] for h in horizons]
    labels = [f'{h}d' for h in horizons]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    c1 = [C['green'] if v > 0.1 else C['grey'] if v > 0 else C['red']
          for v in icirs]
    ax1.bar(labels, icirs, color=c1, width=0.5, edgecolor='white')
    ax1.axhline(y=0.2, color=C['green'], linestyle='--', alpha=0.7, label='Moderate')
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_ylabel('ICIR'); ax1.set_title('ICIR Decay'); ax1.legend()

    c2 = [C['green'] if v > 1.96 else C['orange'] if v > 1.28 else C['red']
          for v in nw_ts]
    ax2.bar(labels, nw_ts, color=c2, width=0.5, edgecolor='white')
    ax2.axhline(y=1.96, color=C['primary'], linestyle='--', alpha=0.7, label='95% sig.')
    ax2.axhline(y=-1.96, color=C['primary'], linestyle='--', alpha=0.7)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_ylabel('NW t-stat'); ax2.set_title('Significance Decay'); ax2.legend()

    plt.tight_layout()
    _save(fig, 'ic_decay_analysis.png')


def plot_ic_distribution(ic_series, horizon):
    """IC histogram with normal fit."""
    ics = ic_series['ic'].values if hasattr(ic_series, 'values') else np.array(ic_series)
    mu, sigma = np.mean(ics), np.std(ics, ddof=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(ics, bins=min(30, len(ics)//3), density=True, alpha=0.6,
            color=C['primary'], edgecolor='white')

    from scipy.stats import norm
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
    ax.plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=2,
            label=f'N({mu:.3f}, {sigma:.3f})')
    ax.axvline(x=0, color='black', linestyle='--')
    ax.axvline(x=mu, color='red', label=f'Mean={mu:.4f}')
    ax.set_xlabel('Daily Rank IC'); ax.set_ylabel('Density')
    ax.set_title(f'HAMR — IC Distribution ({horizon}d)')
    ax.legend()

    pos_ratio = np.mean(ics > 0)
    ax.text(0.97, 0.97, f'IC>0: {pos_ratio:.1%}', transform=ax.transAxes,
            fontsize=11, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    _save(fig, f'ic_distribution_{horizon}d.png')


# -----------------------------------------------------------------------
# Factor Charts
# -----------------------------------------------------------------------

def plot_quintile_returns(group_results, horizon='5d'):
    """Quintile bar chart with monotonicity annotation."""
    means = group_results['group_means']
    groups = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    values = [means.get(g, 0) for g in groups]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(groups, [v*100 for v in values], color=C['q5'],
                  width=0.55, edgecolor='white', linewidth=1.5)

    for bar, val in zip(bars, values):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, y + (0.05 if val >= 0 else -0.05),
                f'{val:.2%}', ha='center', fontsize=10, fontweight='bold',
                va='bottom' if val >= 0 else 'top')

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_ylabel('Forward Return (%)')
    ax.set_title(f'HAMR Quintile Portfolio Returns ({horizon})')

    spread = group_results['spread']
    mono = group_results.get('monotonicity', 0)
    ax.text(0.97, 0.95,
            f'Q5-Q1: {spread:.4%}\nMonotonicity: {mono:.3f}',
            transform=ax.transAxes, fontsize=10, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    _save(fig, 'quintile_returns.png')


def plot_ai_heat_state(panel):
    """AIStateScore time series."""
    if 'AIStateScore' not in panel.columns:
        return

    daily = panel.groupby('date')['AIStateScore'].first().reset_index().sort_values('date')
    fig, ax = plt.subplots(figsize=(12, 3.5))

    ax.fill_between(range(len(daily)), daily['AIStateScore'], alpha=0.3,
                     color=C['primary'])
    ax.plot(range(len(daily)), daily['AIStateScore'], color=C['primary'],
            linewidth=1.5)
    ax.axhline(y=0.5, color='black', linewidth=0.5, linestyle='--', label='Median')

    n = len(daily)
    ticks = np.linspace(0, n-1, min(10, n), dtype=int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(d)[:10] for d in daily['date'].values[ticks]], rotation=45)
    ax.set_ylabel('AIStateScore')
    ax.set_title('HAMR — AIHeat Market State Over Time')
    ax.legend()

    plt.tight_layout()
    _save(fig, 'ai_heat_state.png')


def plot_factor_correlation(panel):
    """Sub-component correlation heatmap."""
    cols = ['MismatchScore', 'QualityScore', 'ResidualWeakness',
            'NonFundamentalOK', 'FundingVacuum', 'LiquidityCapacity',
            'HAMR_Core', 'HAMR_Final']
    available = [c for c in cols if c in panel.columns]
    if len(available) < 2:
        return

    corr = panel[available].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    for i in range(len(available)):
        for j in range(len(available)):
            ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center',
                    fontsize=8, color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black')

    ax.set_xticks(range(len(available)))
    ax.set_yticks(range(len(available)))
    ax.set_xticklabels(available, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(available, fontsize=8)
    ax.set_title('HAMR — Sub-Component Correlation Matrix')
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    _save(fig, 'factor_correlation.png')


def plot_component_attribution(attribution):
    """Bar chart of IC per sub-component."""
    if not attribution:
        return
    names = [k for k in attribution if k != 'NonFundamentalOK'
             or attribution[k].get('ic_mean', 0) != 0]
    ic_means = [attribution[n]['ic_mean'] for n in names]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = [C['green'] if v > 0 else C['red'] for v in ic_means]
    ax.bar(range(len(names)), ic_means, color=colors, width=0.55, edgecolor='white')

    for i, v in enumerate(ic_means):
        ax.text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=10,
                fontweight='bold')

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('Rank IC (10d)')
    ax.set_title('HAMR — Component Attribution')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    _save(fig, 'component_attribution.png')


def plot_control_comparison(comparisons, horizon='5d'):
    """HAMR vs control factor spread comparison."""
    if horizon not in comparisons:
        return
    comps = comparisons[horizon]
    names = list(comps.keys())
    spreads = [comps[n].get('spread', 0) for n in names]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [C['primary'] if n == 'HAMR_Final' else C['grey'] for n in names]
    ax.barh(names, [s*100 for s in spreads], color=colors, height=0.5,
            edgecolor='white')

    for i, (n, s) in enumerate(zip(names, spreads)):
        ax.text(s*100 + 0.05, i, f'{s:.2%}', va='center', fontsize=10,
                fontweight='bold' if n == 'HAMR_Final' else 'normal')

    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('Q5-Q1 Spread (%)')
    ax.set_title(f'HAMR vs Control Factors ({horizon})')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    _save(fig, f'control_comparison_{horizon}.png')


# -----------------------------------------------------------------------
# Generate All
# -----------------------------------------------------------------------

def generate_all(panel, ic_results, group_results, comparisons=None,
                 attribution=None, ai_conditional=None):
    """
    Generate all publication-quality charts.
    """
    print('\n[Charts] Generating...')

    # IC charts
    for h in ['5', '10', '20', '60', '120']:
        if h in ic_results:
            plot_ic_timeseries(ic_results[h]['ic_series'], h,
                               ic_results[h]['stats'])
            plot_ic_distribution(ic_results[h]['ic_series'], h)

    if ic_results:
        plot_ic_decay(ic_results)

    # Factor charts
    if group_results:
        plot_quintile_returns(group_results)

    if panel is not None:
        plot_ai_heat_state(panel)
        plot_factor_correlation(panel)

    if attribution:
        plot_component_attribution(attribution)

    if comparisons:
        for h in comparisons:
            plot_control_comparison(comparisons, h)

    print(f'  Saved to: {CHART_DIR}')
