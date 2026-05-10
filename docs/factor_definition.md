# HAMR Factor — Full Variable Specification

## Factor Definition

```
HAMR_Final(i,t) = AIStateScore(t)
                × QualityScore(i,t)
                × MispricingPressure(i,t)
                × FundingVacuum(i,t)
                × LiquidityCapacity(i,t)
                × TrapGuard(i,t)

MispricingPressure(i,t) = MismatchScore(i,t)
                        × ResidualWeakness(i,t)
                        × NonFundamentalOK(i,t)
```

## Variable Specifications

All sub-variables use **RankPct** transformation → output ∈ [0, 1].
All variables are **direction-unified**: higher = stronger signal.

### Layer 1: AIHeat_State (Market-Level Condition)

| Component | Formula | Data Source |
|-----------|---------|-------------|
| AIHeat_raw | 0.35×TurnoverZ + 0.35×DispersionZ + 0.30×ConcentrationZ | Price/volume |
| AIHeat_MA20 | MA(AIHeat_raw, 20) | — |
| AIStateScore | RankPct_rolling(AIHeat_MA20, 252) | — |

**Production upgrade**: GitHub activity, search trends, GDELT news,
QuantDinger strategy generation logs.

### Layer 2: MismatchScore (Template Deviation)

| Component | Formula | Data Source |
|-----------|---------|-------------|
| HotTemplateMatch | 0.4×RankPct(|ret_20d|) + 0.3×RankPct(turnover) + 0.3×RankPct(spike) | Price/volume |
| MismatchScore | 1 − RankPct(HotTemplateMatch) | — |

**Production upgrade**: Cluster hot stocks by multi-dimensional features
(market cap, volatility, momentum, turnover, sector), compute cosine
distance to nearest centroid.

### Layer 3: QualityScore (Fundamental Barrier)

| Component | Weight | Proxy |
|-----------|--------|-------|
| Volatility inverse | 25% | 1 − RankPct(volatility_20d) |
| Liquidity | 25% | RankPct(dollar_volume) |
| Return stability | 25% | RankPct(1/std(ret_20d)) |
| Amihud inverse | 25% | 1 − RankPct(Amihud illiquidity) |

**Production upgrade**: ROE, ROA, gross margin stability, OCF/NI,
D/E ratio, interest coverage, accruals, earnings revisions — all
PIT-aligned by disclosure date.

### Layer 4: Mispricing Pressure

| Component | Formula |
|-----------|---------|
| ResidualWeakness | RankPct(−(ret_5d − market_ret_5d)) |
| NonFundamentalOK | 1 − RankPct(extreme negative deviation avg) |
| MispricingPressure | MismatchScore × ResidualWeakness × NonFundamentalOK |

**Production upgrade**: Residualize against market + industry + style
factors. NonFundamentalOK should incorporate earnings revisions,
news sentiment, regulatory events.

### Layer 5: Execution Filter

| Component | Formula |
|-----------|---------|
| CrowdingScore | 0.35×RankPct(turnover) + 0.35×RankPct(spike) + 0.30×RankPct(ret_5d) |
| FundingVacuum | 1 − RankPct(CrowdingScore) |
| LiquidityCapacity | RankPct(dollar_volume_20d) |
| TrapGuard | Sigmoid(QualityScore, 0.35) × Sigmoid(LiquidityCapacity, 0.25) |

## Normalization Conventions

1. **Winsorize** extremes at 1%/99% before RankPct
2. **RankPct** applied cross-sectionally per date
3. All variables clipped to [0, 1]
4. Missing values filled at 0.5 (neutral)

## Data Alignment Rules

- Factor(i, t) uses only data available at or before time t
- Financial data aligned by disclosure date (`ann_date <= t`), not report period
- Forward returns computed from t+1 close to t+h close
- No look-ahead in any variable construction
