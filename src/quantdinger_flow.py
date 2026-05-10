"""
quantdinger_flow.py
===================
QuantDinger AI strategy flow data loader.

Loads local CSV of paper trading signals for PositiveAIFlow.
When no data exists, returns None (neutral value used in FundingVacuum).
"""

import pandas as pd
from pathlib import Path

DATA_EXTERNAL = Path(__file__).parent.parent / 'data' / 'external'


def load_quantdinger_ai_flow(csv_path=None):
    """
    Load QuantDinger paper trading flow data.

    Expected CSV: date,code,buy_signals,sell_signals,backtest_inclusion
    Returns DataFrame with [date, code, ai_flow] where:
      ai_flow = buy_signals - sell_signals (net positive AI attention)
    """
    if csv_path is None:
        csv_path = DATA_EXTERNAL / 'quantdinger_ai_flow.csv'

    csv_path = Path(csv_path) if not isinstance(csv_path, Path) else csv_path
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path, parse_dates=['date'])
    df['ai_flow'] = df.get('buy_signals', 0) - df.get('sell_signals', 0)
    if 'backtest_inclusion' in df.columns:
        df['ai_flow'] += df['backtest_inclusion'] * 0.5
    return df[['date', 'code', 'ai_flow']]
