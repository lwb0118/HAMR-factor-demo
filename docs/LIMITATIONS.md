# HAMR Factor — Limitations

This is **HAMR v1.0 Demo / Research Prototype**. The following limitations should be considered when interpreting results.

## Data Coverage

| Area | Status | Detail |
|------|--------|--------|
| Sample period | ~6 months | Oct 2025 – May 2026. Not yet multi-year. |
| Stock universe | CSI 1000, 200 stocks | Not full A-share market. |
| Financial data | PIT-aligned, quarterly | Coverage varies by reporting season. |
| AIHeat external | GitHub (2 repos), financial news, guba | GDELT blocked; Baidu/WeChat Index no public API. |

## Proxy Variables

| Variable | Current implementation | Production upgrade |
|----------|----------------------|-------------------|
| MismatchScore | ret_20d + turnover proxy | Multi-feature KMeans clustering on HotSet |
| NonFundamentalOK | Price deviation + forecast penalty | Real announcements, NLP, regulatory events |
| StockAttention | Latest guba page snapshot | Daily historical attention time series |
| AIHeat | Agent run activity state | QuantDinger logs, strategy generation frequency |

## No Transaction Cost Model

Results are **gross returns**. Strategy tests do not include:
- Commission fees
- Market impact / slippage
- Short-selling constraints (A-share short selling is restricted)
- Capacity limits

## Survivorship & Look-Ahead

- Current CSI 1000 constituents used; historical index changes not fully modeled
- Financial data uses `ann_date` PIT alignment but check is not yet automated
- Forward returns computed from t+1; same-day trading not assumed

## Statistical Caveats

- ~130 trading days → Newey-West t-stat may be unstable with small samples
- Quintile results are in-sample; no walk-forward out-of-sample test yet
- Placedo / randomization tests not yet implemented
- Factor performance may vary across market regimes

## Production Readiness

**This project is NOT suitable for live trading.** Key gaps:
- No real-time data pipeline
- No monitoring or alerting
- No risk management or position sizing
- No execution system integration

## Disclaimer

**For academic research and educational purposes only. Not financial advice.** All results should be interpreted as statistical associations, not as sources of deterministic returns. Past performance does not guarantee future results.
