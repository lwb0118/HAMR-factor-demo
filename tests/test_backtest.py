"""
test_backtest.py — verify per-date quintile grouping
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from src.backtest import quintile_test


def test_per_date_grouping():
    """Each date should have its own quintile groups."""
    rng = np.random.default_rng(42)
    dates = pd.date_range('2026-01-01', periods=20, freq='B')
    codes = [f'{600000+i}' for i in range(50)]
    rows = []
    for d in dates:
        for c in codes:
            rows.append({
                'date': d, 'code': c,
                'factor': rng.normal(0, 1),
                'fwd_ret': rng.normal(0.01, 0.03)
            })
    panel = pd.DataFrame(rows)
    result = quintile_test(panel, 'factor', 'fwd_ret', n_groups=5)
    assert 'group_means' in result
    assert 'Q1' in result['group_means']
    assert 'Q5' in result['group_means']
    assert abs(result['monotonicity']) <= 1
    print('PASS: per-date quintile test')


if __name__ == '__main__':
    test_per_date_grouping()
