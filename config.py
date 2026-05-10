"""
config.py — API tokens and settings.
Load from environment variables or local file (not tracked by git).
"""

import os
from pathlib import Path

# --- Tushare ---
TUSHARE_TOKEN = os.environ.get('TUSHARE_TOKEN', 'your_tushare_token_here')

# --- GitHub ---
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')  # optional, higher rate limit

# --- Local override ---
_LOCAL_CONFIG = Path(__file__).parent / 'config.local.py'
if _LOCAL_CONFIG.exists():
    with open(_LOCAL_CONFIG, encoding='utf-8') as f:
        exec(f.read())
