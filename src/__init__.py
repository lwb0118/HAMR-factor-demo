# HAMR Mispricing Reversion Alpha — Factor Research Package
# src/__init__.py

__version__ = '1.0.0'
__author__ = 'HAMR Research'

from . import (
    data_process, data_process_ext, data_filters,
    ai_heat, github_timeseries, news_aiheat, guba_data,
    template_cluster, quality, residual, funding_vacuum,
    hamr_factor, ic_test, backtest,
    regression, mechanism, ml_model, visualization
)
