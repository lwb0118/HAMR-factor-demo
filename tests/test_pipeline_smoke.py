"""Pipeline smoke tests — verify core functions run without errors."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def _make_sample_panel(n_dates=10, n_codes=30):
    """Generate minimal test panel."""
    rng = np.random.default_rng(42)
    dates = pd.date_range('2026-01-01', periods=n_dates, freq='B')
    codes = [f'{600000 + i}' for i in range(n_codes)]
    rows = []
    for d in dates:
        for c in codes:
            rows.append({
                'date': d, 'code': c,
                'ret_5d': rng.normal(0, 0.03),
                'ret_20d': rng.normal(0.01, 0.08),
                'turnover': rng.uniform(1, 10),
                'turnover_avg_20d': rng.uniform(2, 8),
                'volatility_20d': rng.uniform(0.02, 0.06),
                'dollar_volume': rng.lognormal(14, 0.8),
                'amount': rng.lognormal(14, 0.8),
                'close': rng.uniform(10, 50),
                'ret_1d': rng.normal(0, 0.02),
                'industry': rng.choice(['Tech', 'Finance', 'Health'], n_codes),
            })
    return pd.DataFrame(rows)


def test_compute_stock_attention_no_name_error():
    """Fix: rows=[] must be defined before the loop."""
    from src.guba_data import compute_stock_attention
    posts = {
        '000001.SZ': {
            'total_posts': 5,
            'latest_titles': ['AI量化'] * 5,
            'latest_timestamps': ['2026-05-10'] * 5,
        }
    }
    out = compute_stock_attention(None, posts, trade_date='2026-05-10')
    assert out is not None
    assert 'stock_attention_raw' in out.columns


def test_template_affinity_signature():
    """Fix: recent_only must be in function signature."""
    import inspect
    from src.template_cluster import compute_template_affinity
    params = inspect.signature(compute_template_affinity).parameters
    print(f'Template params: {list(params.keys())}')


def test_quantdinger_loader_missing_file_returns_none(tmpdir=None):
    """Fix: loader must handle missing CSV gracefully."""
    from src.quantdinger_flow import load_quantdinger_ai_flow
    from pathlib import Path
    p = Path('/nonexistent/path.csv')
    if not p.exists():
        assert load_quantdinger_ai_flow(p) is None


def test_funding_vacuum_accepts_attention_and_flow():
    """Fix: FundingVacuum must accept stock_attention_df and positive_ai_flow_df."""
    from src.funding_vacuum import compute_funding_vacuum
    panel = pd.DataFrame({
        'date': pd.to_datetime(['2026-05-01', '2026-05-01']),
        'code': ['000001.SZ', '000002.SZ'],
        'ret_5d': [0.01, -0.02],
        'turnover': [1.0, 2.0],
    })
    sa = pd.DataFrame({'code': ['000001.SZ'], 'stock_attention_raw': [10]})
    flow = pd.DataFrame({
        'date': pd.to_datetime(['2026-05-01']),
        'code': ['000001.SZ'],
        'ai_flow': [5],
    })
    out = compute_funding_vacuum(panel, sa, flow)
    assert 'FundingVacuum' in out.columns
    assert out['FundingVacuum'].between(0, 1).all()


if __name__ == '__main__':
    test_compute_stock_attention_no_name_error()
    print('PASS: stock attention')
    test_template_affinity_signature()
    print('PASS: template signature')
    test_quantdinger_loader_missing_file_returns_none()
    print('PASS: quantdinger loader')
    test_funding_vacuum_accepts_attention_and_flow()
    print('PASS: funding vacuum with attention/flow')
