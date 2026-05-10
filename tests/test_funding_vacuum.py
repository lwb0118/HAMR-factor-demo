import pandas as pd, numpy as np
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from src.funding_vacuum import compute_funding_vacuum

def test_with_stock_attention():
    rng = np.random.default_rng(42)
    dates = pd.date_range('2026-01-01', periods=10, freq='B')
    codes = [f'{600000+i}' for i in range(30)]
    rows = []
    for d in dates:
        for c in codes:
            rows.append({'date': d, 'code': c, 'ret_5d': rng.normal(0, 0.03),
                         'turnover': rng.uniform(1, 10), 'amount': rng.lognormal(14, 0.8)})
    panel = pd.DataFrame(rows)
    sa = pd.DataFrame({'code': codes, 'stock_attention_raw': rng.uniform(10, 100, len(codes))})
    result = compute_funding_vacuum(panel, stock_attention_df=sa)
    assert 'FundingVacuum' in result.columns
    assert 'CrowdingScore' in result.columns
    print('PASS: funding_vacuum with stock attention')

test_with_stock_attention()
