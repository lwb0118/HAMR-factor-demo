<p align="center">
  <img src="results/figures/quintile_returns.png" width="550" alt="HAMR Quintile Returns"/>
</p>

# HAMR Factor — HAMR Mispricing Reversion Research

> **《AI 同质化交易错杀修复》因子 — 学术验证与复现**

<p align="center">
  <b>HAMR (Homogeneous AI Mispricing Reversion)</b><br>
  <em>Identifying Quality Stocks Mispriced by AI-Driven Template Trading</em>
</p>

<p align="center">
  <a href="#-intuition"><strong>Intuition</strong></a> ·
  <a href="#-abstract"><strong>Abstract</strong></a> ·
  <a href="#-theory"><strong>Theory</strong></a> ·
  <a href="#-factor-construction"><strong>Factor</strong></a> ·
  <a href="#-empirical-results"><strong>Results</strong></a> ·
  <a href="#-usage"><strong>Usage</strong></a> ·
  <a href="#-architecture"><strong>Architecture</strong></a>
</p>

---

## 🧠 Intuition

When AI quant tools and templated strategies become widespread, capital
concentrates in stocks that fit the "hot template" — high momentum, high
turnover, attention-grabbing names.

Meanwhile, fundamentally sound stocks that don't match current templates
may be neglected or passively sold, creating **short-term mispricing**.

The HAMR factor identifies these stocks *before* the market corrects:

> **High-quality + non-template + residual weakness + no crowding
> → potential mispricing reversal**

---

## 📜 Abstract

We introduce the **HAMR (Homogeneous AI Mispricing Reversion)** factor —
a conditional, mean-reverting, quality-constrained Alpha for Chinese
A-shares.

Unlike simple reversal or low-crowding factors, HAMR requires
**simultaneous satisfaction** of five conditions:

| Condition | What it means |
|-----------|---------------|
| 🔍 **Template Mismatch** | Stock doesn't fit current hot trading templates |
| 🛡️ **Quality Barrier** | Fundamentals are sound — not a value trap |
| 📉 **Residual Weakness** | Recent underperformance NOT explained by market/industry |
| ✅ **Non-Fundamental OK** | Decline is NOT due to bad news or earnings deterioration |
| 🌌 **Funding Vacuum** | Stock is NOT currently being chased by crowd money |

The factor is **state-conditional**: it activates primarily when
AI-driven template trading intensity (AIHeat) is elevated.

---

## 🧠 Theoretical Foundation

### The Mispricing-Reversion Chain

```
AIHeat ↑ → Hot templates form → Capital flows to template stocks
    → Quality non-template stocks lose attention
    → Residual weakness (not fundamental)
    → Mispricing correction when attention returns
```

### Why Not Just Reversal?

| Factor | What it captures | HAMR difference |
|--------|-----------------|-----------------|
| **Short-term reversal** | Price bouncing back | HAMR requires quality + template mismatch |
| **Low volatility** | Stable, boring stocks | HAMR requires *residual* weakness |
| **Value** | Cheap stocks | HAMR filters out value traps |
| **Low crowding** | Unpopular stocks | HAMR adds timing + quality conditions |

### Behavioral Microfoundations

1. **Attention Cascades** (Peng & Xiong, 2006): Retail attention follows
   hot templates, creating neglect elsewhere
2. **Informational Cascades** (Bikhchandani et al., 1992): Template
   adoption is self-reinforcing
3. **Limits to Arbitrage** (Shleifer & Vishny, 1997): Even sophisticated
   investors may avoid "unfashionable" stocks short-term

---

## ⚙️ Factor Construction

### Five-Layer Architecture

```
                  ┌──────────────────┐
Layer 1           │   AIHeat_State   │  Market condition (gating)
                  └────────┬─────────┘
                           ↓
        ┌──────────────────┼──────────────────┐
Layer 2 │  MismatchScore   │                  │  Template deviation
        └────────┬─────────┘                  │
                 ↓                             │
Layer 3 │     QualityScore  │                  │  Fundamental barrier
        └────────┬─────────┘                  │
                 ↓                             │
        ┌────────┴─────────┐                  │
Layer 4 │ ResidualWeakness │ NonFundamentalOK │  Mispricing detection
        └────────┬─────────┘                  │
                 ↓                             │
Layer 5 │   FundingVacuum   │  TrapGuard      │  Execution filter
        └───────────────────┘                  │
```

### Core Formulas

```
MispricingPressure = MismatchScore × ResidualWeakness × NonFundamentalOK
HAMR_Core = QualityScore × MispricingPressure × FundingVacuum
HAMR_Final = AIStateScore × HAMR_Core × LiquidityCapacity × TrapGuard
```

All sub-variables are RankPct-transformed to [0, 1], unified as
"higher = stronger signal".

### Variable Definitions

| Variable | Range | Direction | Construction |
|----------|-------|-----------|-------------|
| **MismatchScore** | [0,1] | Higher = less template-like | 1 − RankPct(HotTemplateMatch) |
| **QualityScore** | [0,1] | Higher = better quality | Composite: vol stability, liquidity, return stability |
| **ResidualWeakness** | [0,1] | Higher = weaker residual | RankPct(−(ret_5d − market_ret_5d)) |
| **NonFundamentalOK** | [0,1] | Higher = not fundamental | 1 − RankPct(extreme negative deviations) |
| **FundingVacuum** | [0,1] | Higher = less crowded | 1 − RankPct(CrowdingScore) |
| **AIStateScore** | [0,1] | Market-level | Market turnover + dispersion z-score |
| **LiquidityCapacity** | [0,1] | Higher = more liquid | RankPct(dollar_volume) |
| **TrapGuard** | [0,1] | Soft filter | Sigmoid(QualityScore, LiquidityCapacity) |

> **Note**: This demo uses price-based proxy variables. Production
> implementation uses financial statement data (PIT-aligned), NLP-based
> template clustering, and multi-source AIHeat indicators.

---

## 🔬 Empirical Methodology

Standard academic factor testing protocol:

### 1. Cross-Sectional Rank IC (Spearman)

```
IC_t = Corr(Rank(HAMR_Final_i,t), Rank(ForwardReturn_i,t→t+h))
```

### 2. ICIR & Newey-West HAC Inference

```
ICIR = mean(IC) / std(IC)
NW_SE = HAC standard error with Bartlett kernel
```

### 3. Quintile Portfolio Test

Stocks sorted into Q1–Q5 by HAMR_Final. Monotonic Q1<Q2<...<Q5
return pattern validates the factor.

### 4. Control Factor Comparison

HAMR compared against:
- QualityScore only
- ResidualWeakness only (reversal proxy)
- QualityScore × ResidualWeakness (naive combination)

### 5. AIHeat State-Conditional IC

IC computed separately for high vs. low AIHeat states to test
whether HAMR works better when template trading is active.

---

## 📂 Project Structure

```
hamr-factor-demo/
├── run.py                          # Main entry point
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── src/
│   ├── __init__.py                 # Package init
│   ├── data_fetcher.py             # Data acquisition (akshare API)
│   ├── factor_calculator.py        # 5-layer HAMR factor computation
│   ├── ic_test.py                  # IC/ICIR/Newey-West testing suite
│   ├── backtest.py                 # Quintile + control comparison
│   └── visualization.py            # Publication-quality charts
├── data/                           # Cached data (auto-populated)
├── results/
│   └── charts/                     # Generated figures
├── docs/
│   ├── methodology.md              # Detailed methodology
│   └── factor_definition.md        # Full variable specifications
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Internet connection (for akshare data)

### Installation

```bash
git clone https://github.com/yourusername/hamr-factor-demo.git
cd hamr-factor-demo
pip install -r requirements.txt
```

### Running the Demo

```bash
# Quick demo (15 stocks, ~15 seconds)
python run.py --quick

# Standard demo (30 stocks, ~40 seconds)
python run.py

# Full market analysis (200 stocks, ~90 seconds)
python run.py --full
```

### Expected Output

```
╔══════════════════════════════════════════════════════════╗
║   HAMR (Homogeneous AI Mispricing Reversion) Factor     ║
╠══════════════════════════════════════════════════════════╣
║  Mode: Standard Demo                                    ║
╚══════════════════════════════════════════════════════════╝

Step 1: Data Acquisition
Step 2: HAMR Factor Computation (5-layer)
Step 3: Cross-Sectional Rank IC Analysis
Step 4: Quintile Portfolio Backtest
Step 5: Control Factor Comparison
Step 6: Component Attribution
Step 7: Publication-Quality Visualizations
```

---

## 📊 Key Charts

### 20-Day Quintile Returns
<p align="center">
  <img src="results/figures/quintile_returns.png" width="500" alt="Quintile Returns"/>
</p>

### IC Time Series & Distribution
<p align="center">
  <img src="results/figures/ic_timeseries_20d.png" width="400" alt="IC 20d"/>
  <img src="results/figures/ic_distribution_20d.png" width="400" alt="IC Dist 20d"/>
</p>

### IC Decay Across Horizons
<p align="center">
  <img src="results/figures/ic_decay_analysis.png" width="500" alt="IC Decay"/>
</p>

### AIHeat State & Factor Correlation
<p align="center">
  <img src="results/figures/ai_heat_state.png" width="500" alt="AIHeat State"/>
</p>

All figures: `results/figures/`

---

## 🔄 Relation to SDL Factor

| Dimension | SDL | HAMR |
|-----------|-----|------|
| **Signal type** | Institutional flow timing | Template mispricing reversion |
| **Key condition** | (none — always active) | AIHeat state (conditional) |
| **Quality filter** | Implicit (flow-driven) | Explicit (QualityScore barrier) |
| **Value trap guard** | None | TrapGuard + NonFundamentalOK |
| **Variable type** | z-score | RankPct [0,1] |
| **Formula structure** | Linear additive | Multiplicative interaction |
| **Best use** | Stock selection | Conditional Alpha overlay |

---

## ⚠️ Limitations (Current Demo Version)

1. **Proxy variables**: QualityScore, AIHeat, NonFundamentalOK use
   price-based proxies; production needs financial statements and external data
2. **Template clustering**: Simplified as momentum+turnover score;
   production should use actual clustering on multi-dimensional features
3. **Single market regime**: ~6 months of data insufficient for
   robust regime analysis
4. **No transaction cost model**: Demo; production needs impact
   cost estimation

### Production Roadmap

| Phase | Description |
|-------|-------------|
| **v1.0** | Price-based proxies + akshare (this repo) |
| **v2.0** | PIT financial data + Tushare + template clustering |
| **v3.0** | GitHub/GDELT AIHeat + NLP news filtering |
| **v4.0** | QuantDinger logs + production backtest |

---

## 📚 Selected References

1. Peng, L., & Xiong, W. (2006). Investor attention, overconfidence
   and category learning. *Journal of Financial Economics*.
2. Bikhchandani, S., Hirshleifer, D., & Welch, I. (1992). A theory
   of fads, fashion, custom, and cultural change as informational cascades.
   *Journal of Political Economy*.
3. Grinold, R. C., & Kahn, R. N. (2000). *Active Portfolio Management*.
4. Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite,
   heteroskedasticity and autocorrelation consistent covariance matrix.
   *Econometrica*.
5. Shleifer, A., & Vishny, R. W. (1997). The limits of arbitrage.
   *Journal of Finance*.

---

## ⚠️ Disclaimer

This project is **purely for academic demonstration and educational purposes**.
It is not intended for live trading or investment decision-making. The factor
implementation uses simplified proxies; production-grade HAMR incorporates
proprietary methodology and multi-source data not disclosed here.

**Not financial advice. Use at your own risk.**

---

<p align="center">
  <sub>© 2026 HAMR Research. MIT License.</sub>
</p>
