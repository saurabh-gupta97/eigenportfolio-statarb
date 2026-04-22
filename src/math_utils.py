import pandas as pd
import numpy as np
from scipy.stats import skew, norm

def compute_sector_correlation_matrix(returns_wide, metadata):
    """
    Filters universe, sorts by sector ETF, computes the correlation matrix, 
    and finds the boundaries between sectors.
    """
    print("Filtering universe for full coverage and sorting by sector...")
    
    # 1. Filter for full coverage and sort
    valid_meta = metadata[metadata['return_coverage'] == 1.0].copy()
    valid_meta = valid_meta.dropna(subset=['etf'])
    valid_meta = valid_meta.sort_values(by=['etf', 'ticker'])
    
    # 2. Extract ordered lists
    ordered_tickers = valid_meta['ticker'].tolist()
    ordered_etfs = valid_meta['etf'].tolist()
    
    print(f"Kept {len(ordered_tickers)} highly liquid, full-coverage stocks.")
    
    # 3. Compute the empirical correlation matrix
    clean_returns = returns_wide[ordered_tickers]
    corr_matrix = clean_returns.corr()
    
    # 4. Find the index boundaries where the sector changes
    boundaries = []
    current_etf = ordered_etfs[0]
    
    for i, etf in enumerate(ordered_etfs):
        if etf != current_etf:
            boundaries.append(i)
            current_etf = etf
            
    return corr_matrix, ordered_tickers, boundaries

def compute_correlation_distribution(corr_matrix):
    """
    Extracts the unique pairwise correlations (upper triangle) and 
    computes statistical moments and Gaussian fit parameters.
    """
    # Dynamically get the dimension of the matrix instead of hardcoding
    n = corr_matrix.shape[0]
    
    # Extract strictly off-diagonal upper triangle elements
    corr_dist = corr_matrix.values[np.triu_indices(n, k=1)]
    
    # Calculate statistics
    corr_mean = np.mean(corr_dist)
    corr_var = np.var(corr_dist)
    corr_skew = skew(corr_dist)
    
    # Fit theoretical Gaussian
    mu_fit, sigma_fit = norm.fit(corr_dist)
    
    stats_dict = {
        'mean': corr_mean,
        'var': corr_var,
        'skew': corr_skew,
        'mu_fit': mu_fit,
        'sigma_fit': sigma_fit
    }
    
    return corr_dist, stats_dict

def apply_volume_dampener(clean_window_returns, volume_wide, window_start, window_end, avg_window=60):
    """
    Transforms calendar-time returns into volume-adjusted 'trading time' returns.
    Dampens returns that occur on unusually high-volume days to prevent false s-score triggers.
    """
    valid_tickers = clean_window_returns.columns
    
    # 1. Compute rolling average on the full dataset first to avoid brittle index .get_loc() calls
    # min_periods=1 ensures we get an average even at the beginning of the dataset
    avg_volume_full = volume_wide[valid_tickers].rolling(window=avg_window, min_periods=1).mean()
    
    # 2. Align the sliced volume and moving average strictly to the return window
    current_volume = volume_wide.loc[window_start:window_end, valid_tickers]
    avg_volume = avg_volume_full.loc[window_start:window_end, valid_tickers]
    
    # 3. Calculate the volume ratio: V_{i,t} / \bar{V}_{i,t}
    # Replace 0s with 1s in avg_volume to prevent division by zero
    avg_volume = avg_volume.replace(0, 1)
    volume_ratio = current_volume / avg_volume
    
    # 4. Calculate the dampening factor: max(1, sqrt(volume_ratio))
    dampening_factor = np.sqrt(volume_ratio)
    dampening_factor = dampening_factor.clip(lower=1.0)
    
    # 5. Apply the dampener to the raw returns
    scaled_returns = clean_window_returns / dampening_factor
    
    return scaled_returns