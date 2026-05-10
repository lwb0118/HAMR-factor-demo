# HAMR Factor — Methodology Notes

## Research Design

### Core Hypothesis

> In AI-driven template-trading environments, quality stocks that
> deviate from hot templates experience non-fundamental selling
> pressure, creating exploitable mispricing reversals.

### Five Research Questions

1. **Does HAMR predict cross-sectional returns?**
   → Rank IC + ICIR + Newey-West inference

2. **Is HAMR different from quality/reversal/value factors?**
   → Control factor comparison (Quality, Reversal, Quality×Reversal)

3. **Does HAMR work better when AIHeat is high?**
   → State-conditional IC analysis

4. **Which sub-components drive the signal?**
   → Component attribution (IC per sub-variable)

5. **Is the strategy executable?**
   → Transaction cost analysis, capacity estimation

## Statistical Methodology

### Rank IC (Information Coefficient)

Spearman rank correlation between factor value and forward return,
computed cross-sectionally each day. Produces IC time series.

### ICIR (IC Information Ratio)

ICIR = mean(IC) / std(IC)

Grinold & Kahn (2000) benchmarks:
- ICIR > 0.5: Strong factor
- ICIR > 0.2: Moderate factor
- ICIR > 0.0: Weak but positive

### Newey-West HAC

Accounts for autocorrelation in IC time series using Bartlett kernel
with automatic lag selection (n^{1/4}).

### Quintile Portfolio Test

Stocks sorted into 5 equal-sized portfolios by factor value.
Monotonic return pattern (Q1 < Q2 < ... < Q5) validates the factor.

### AIHeat State-Conditional Analysis

Split dates by median AIStateScore. Compare IC between high/low states.
HAMR should show stronger signal in high AIHeat periods.

## Limitations

1. **Proxy quality**: Price-based QualityScore proxies are noisy
2. **No transaction costs**: Demo only; production needs impact model
3. **Single data source**: akshare may have coverage gaps
4. **Short history**: Limited by akshare fund flow data (~120 days)
5. **Simplified template detection**: No actual clustering

## Production Path

See `docs/factor_definition.md` for production variable upgrades.
