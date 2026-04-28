import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def paired_block_bootstrap(returns_model_a, returns_model_b, block_size=20, n_iterations=10000):
    """
    Blazing-fast vectorized Paired Block Bootstrap for Sharpe Ratio comparison.
    Calculates 10,000 synthetic histories simultaneously using NumPy broadcasting.
    
    Parameters:
    returns_model_a: pd.Series of daily returns for Model A (e.g., PCA)
    returns_model_b: pd.Series of daily returns for Model B (e.g., ETF)
    block_size: int, number of days per block (20 days ~ 1 trading month)
    n_iterations: int, number of synthetic realities to generate
    
    Returns:
    obs_diff: float, the actual observed difference in Sharpe ratios
    p_value: float, the empirical probability that Model A is NOT better than Model B
    diffs: np.array, the distribution of all 10,000 synthetic Sharpe differences
    """
    # 1. Perfectly align the data to preserve cross-correlation
    df = pd.concat([returns_model_a, returns_model_b], axis=1, join='inner').dropna()
    arr = df.values  # Shape: (T, 2)
    T = arr.shape[0]
    
    if T < block_size:
        raise ValueError("Time series is shorter than the block size.")

    # 2. Calculate the True Observed Sharpe Ratios
    obs_mean = arr.mean(axis=0)
    obs_std = arr.std(axis=0)
    # Prevent division by zero
    obs_std[obs_std == 0] = 1e-8 
    obs_sharpe = np.sqrt(252) * (obs_mean / obs_std)
    obs_diff = obs_sharpe[0] - obs_sharpe[1]
    
    # 3. Vectorized Index Generation (The Matrix Magic)
    n_blocks = int(np.ceil(T / block_size))
    
    # Generate random starting indices for every block in every iteration
    # Shape: (n_iterations, n_blocks)
    start_indices = np.random.randint(0, T - block_size + 1, size=(n_iterations, n_blocks))
    
    # Create the block offsets: [0, 1, 2, ..., block_size - 1]
    offsets = np.arange(block_size)
    
    # Broadcast offsets to the start indices to generate all matrix coordinates at once
    # Shape expands to: (n_iterations, n_blocks, block_size) -> flattened to (n_iterations, n_blocks * block_size)
    indices = (start_indices[:, :, None] + offsets).reshape(n_iterations, -1)
    
    # Truncate any overflow to perfectly match original time series length T
    indices = indices[:, :T]
    
    # 4. Extract 10,000 parallel histories in one memory-mapped step
    # Shape: (n_iterations, T, 2)
    samples = arr[indices]
    
    # 5. Calculate Sharpe Ratios across the time axis (axis=1) for all universes
    means = samples.mean(axis=1) # Shape: (n_iterations, 2)
    stds = samples.std(axis=1)   # Shape: (n_iterations, 2)
    
    stds[stds == 0] = 1e-8
    sharpes = np.sqrt(252) * (means / stds)
    
    # 6. Calculate differences (Model A - Model B)
    diffs = sharpes[:, 0] - sharpes[:, 1]
    
    # 7. Calculate empirical p-value (How often did Model A fail to beat Model B?)
    p_value = np.mean(diffs <= 0)
    
    return obs_diff, p_value, diffs

def plot_bootstrap_results(diffs, obs_diff, p_value, save_path="data/results/bootstrap_hist.png"):
    """Visualizes the empirical distribution of the Sharpe differences."""
    plt.figure(figsize=(10, 6))
    
    # Plot the histogram of synthetic differences
    counts, bins, patches = plt.hist(diffs, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    
    # Highlight the zero line (the threshold of failure)
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Difference Threshold')
    
    # Highlight the actual observed difference
    plt.axvline(obs_diff, color='green', linestyle='-', linewidth=2.5, label=f'Observed Diff ({obs_diff:.4f})')
    
    plt.title('Paired Block Bootstrap: PCA vs ETF Sharpe Difference\n(10,000 Iterations)', fontsize=14)
    plt.xlabel('$\Delta$ Sharpe Ratio (PCA - ETF)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    # Add summary box
    summary_text = f"Empirical p-value: {p_value:.4f}\nSignificant at 5%: {'Yes' if p_value < 0.05 else 'No'}"
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
             
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()