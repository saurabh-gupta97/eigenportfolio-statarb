import numpy as np
from scipy.stats import norm

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import os

def plot_correlation_heatmap(corr_matrix, ordered_tickers, boundaries, save_path=None):
    """
    Renders the empirical correlation matrix with visual boundaries for sectors.
    """
    print("Generating heatmap...")
    plt.figure(figsize=(14, 12))
    
    # Create the heatmap
    ax = sns.heatmap(corr_matrix, cmap='coolwarm', vmin=0.0, vmax=1.0, 
                     xticklabels=False, yticklabels=False, cbar_kws={'shrink': 0.8})
                     
    # Draw boundary lines to explicitly show the sector blocks
    for b in boundaries:
        ax.axhline(b, color='black', linewidth=1.5)
        ax.axvline(b, color='black', linewidth=1.5)
        
    # Add an outer border
    ax.axhline(0, color='black', linewidth=3)
    ax.axhline(len(ordered_tickers), color='black', linewidth=3)
    ax.axvline(0, color='black', linewidth=3)
    ax.axvline(len(ordered_tickers), color='black', linewidth=3)

    plt.title('S&P Empirical Correlation Matrix (Clustered by Sector)', fontsize=16, pad=20)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {save_path}")
        
    plt.show()

def plot_correlation_histogram(corr_dist, stats_dict, save_path=None):
    """
    Plots the histogram of pairwise correlations along with theoretical and fitted Gaussian curves.
    """
    print("Generating distribution histogram...")
    plt.figure(figsize=(10, 6))
    
    # Plot empirical histogram
    plt.hist(corr_dist, bins=50, density=True, alpha=0.7, color='steelblue', label='Empirical Data')
    
    # Setup x-axis for theoretical curves
    x = np.linspace(corr_dist.min() - 0.1, corr_dist.max() + 0.1, 500)
    
    # Plot analytical Gaussian (using sample mean and variance)
    pdf_gauss = norm.pdf(x, stats_dict['mean'], np.sqrt(stats_dict['var']))
    plt.plot(x, pdf_gauss, label="Gaussian (Sample Moments)", linewidth=3, color='darkorange')
    
    # Plot Fitted Gaussian (using MLE)
    pdf_gauss_fit = norm.pdf(x, stats_dict['mu_fit'], stats_dict['sigma_fit'])
    plt.plot(x, pdf_gauss_fit, label="Gaussian Fit (MLE)", linewidth=2, linestyle='--', color='red')

    plt.xlabel("Pairwise Correlation Coefficient")
    plt.ylabel("Density")
    plt.title("Distribution of Pairwise Correlations")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Histogram saved to {save_path}")
        
    plt.show()

# ------------------------------------------------------------------------------
# 6. VISUALIZATION AND EXECUTION
# ------------------------------------------------------------------------------
def plot_performance(strategy_returns, model_name="ETF-Baseline", save_path="data/results/performance.png"):
    """Plots the cumulative equity curve and daily returns distribution."""
    # Convert daily raw returns (summed across active pairs) to cumulative PnL
    cumulative_pnl = strategy_returns.cumsum()
    
    # Calculate Sharpe Ratio
    mean_daily_return = strategy_returns.mean()
    std_daily_return = strategy_returns.std()
    annualized_sharpe = np.sqrt(252) * (mean_daily_return / std_daily_return) if std_daily_return > 0 else 0
    
    # Set up matplotlib figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot 1: Cumulative Equity Curve
    axes[0].plot(cumulative_pnl.index, cumulative_pnl.values, color='blue', linewidth=1.5)
    axes[0].fill_between(cumulative_pnl.index, cumulative_pnl.values, 0, color='blue', alpha=0.1)
    axes[0].axhline(0, color='black', linewidth=1)
    axes[0].set_title(f'{model_name} Statistical Arbitrage | Annualized Sharpe: {annualized_sharpe:.2f}', fontsize=14)
    axes[0].set_ylabel('Cumulative PnL (Units)')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # Plot 2: Daily Returns Distribution
    sns.histplot(strategy_returns.values, bins=50, ax=axes[1], color='slategray', kde=True)
    axes[1].axvline(0, color='black', linestyle='--')
    axes[1].set_title('Distribution of Daily Portfolio Returns', fontsize=12)
    axes[1].set_xlabel('Daily PnL')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Performance plot saved to {save_path}")
        
    plt.show()