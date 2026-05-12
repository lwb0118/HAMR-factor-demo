"""
config.py — API tokens and settings.
Load from environment variables or local file (not tracked by git).
"""

import os
from pathlib import Path

# --- Tushare ---
with open('C:/Users/14776/.openclaw/workspace/memory/tushare_token.txt', 'r') as f:
    TUSHARE_TOKEN = f.read().strip()

# --- GitHub ---
with open('C:/Users/14776/.openclaw/workspace/memory/github_token.txt', 'r') as f:
    GITHUB_TOKEN = f.read().strip()

# --- Local override ---
_LOCAL_CONFIG = Path(__file__).parent / 'config.local.py'
if _LOCAL_CONFIG.exists():
    with open(_LOCAL_CONFIG, encoding='utf-8') as f:
        exec(f.read())
