"""
Statistical Arbitrage Trading Project
-------------------------------------
Modules for data ingestion, mathematical transformations, factor modeling, 
and Ornstein-Uhlenbeck strategy simulation.
"""

# Expose main data pipeline functions
from .data_pipeline import (
    download_raw,
    load_raw,
    get_prices_returns_volume,
    get_prices_long,
    get_returns_long,
    get_volume_long,
    get_metadata,
    load_metadata,
    process_universe_etfs,
    download_prices_etf,
    load_prices_etf,
    get_returns_etf,
    get_returns_long_etf
)

# Expose math and statistical utilities
from .math_utils import (
    compute_sector_correlation_matrix,
    compute_correlation_distribution,
    apply_volume_dampener
)

# Expose regression and factor extraction models
from .factor_models import (
    get_rolling_universe,
    compute_rolling_split,
    compute_rolling_split_pca
)

# Expose the strategy engines
from .strategy_engine import (
    calibrate_tradable_universe,
    simulate_hedged_trading,
    evaluate_window,
    run_full_strategy,
    simulate_hedged_trading_pca,
    evaluate_window_pca, 
    run_full_strategy_pca
)

#Expose statistical analysis

from .statistics import (
    paired_block_bootstrap,
    plot_bootstrap_results
)

# Expose visualization tools
from .plotting_utils import (
    plot_correlation_heatmap,
    plot_correlation_histogram,
    plot_performance
)

__all__ = [
    "download_raw",
    "load_raw",
    "get_prices_returns_volume",
    "get_prices_long",
    "get_returns_long",
    "get_volume_long",
    "get_metadata",
    "load_metadata",
    "process_universe_etfs",
    "download_prices_etf",
    "load_prices_etf",
    "get_returns_etf",
    "get_returns_long_etf",
    "compute_sector_correlation_matrix",
    "compute_correlation_distribution",
    "apply_volume_dampener",
    "get_rolling_universe",
    "compute_rolling_split",
    "compute_rolling_split_pca",
    "calibrate_tradable_universe",
    "simulate_hedged_trading",
    "evaluate_window",
    "run_full_strategy",
    "simulate_hedged_trading_pca",
    "evaluate_window_pca",
    "run_full_strategy_pca",
    "paired_block_bootstrap",
    "plot_bootstrap_results",
    "plot_correlation_heatmap",
    "plot_correlation_histogram",
    "plot_performance",
]