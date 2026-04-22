import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os

# Import the actual strategy function
from .strategy_engine import run_full_strategy_pca

def run_pca_parameter_sweep(returns_wide, etf_returns, volume_wide, t2e_dict, param_grid, save_path="data/results/hyperparameter_optimization.png"):
    """
    Executes a grid search across a dictionary of parameter lists to find the optimal
    hyperparameters for the PCA strategy.
    """
    print("Initializing Hyperparameter Grid Search...")
    
    # Extract keys and values from the param_grid
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    
    # Create all possible combinations of the parameters using itertools
    combinations = list(itertools.product(*values))
    print(f"Total parameter combinations to test: {len(combinations)}\n")
    
    results_list = []
    equity_curves = {}
    
    for i, combo in enumerate(combinations):
        # Map the current combination back to the parameter names
        current_params = dict(zip(keys, combo))
        print(f"--- Run {i+1}/{len(combinations)} | Testing: {current_params} ---")
        
        # Execute the actual modularized PCA strategy with the current parameters
        strategy_returns = run_full_strategy_pca(
            returns_wide=returns_wide, 
            etf_returns=etf_returns, 
            volume_wide=volume_wide, 
            t2e_dict=t2e_dict,
            **current_params
        )
        
        # Calculate Performance Metrics
        mean_ret = strategy_returns.mean()
        std_ret = strategy_returns.std()
        
        # Annualized Sharpe (assuming 252 trading days)
        sharpe = np.sqrt(252) * (mean_ret / std_ret) if std_ret > 0 else 0
        cumulative_return = strategy_returns.sum() # Simple sum of log/raw returns
        
        # Store results
        current_params['sharpe'] = sharpe
        current_params['total_return'] = cumulative_return
        results_list.append(current_params)
        
        # Save equity curve for plotting, keyed by a string representation of the parameters
        param_label = f"m:{current_params.get('num_factors', '-')} | en:{current_params.get('entry_threshold', '-')} | ex:{current_params.get('exit_threshold', '-')}"
        equity_curves[param_label] = strategy_returns.cumsum()
        
        print(f"Result -> Sharpe: {sharpe:.4f} | Total Ret: {cumulative_return:.4f}\n")
        
    # Convert results to a DataFrame and sort by best Sharpe ratio
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values(by='sharpe', ascending=False).reset_index(drop=True)
    
    print("Grid Search Complete. Top 5 Parameter Sets:")
    print(results_df.head(5))
    
    # --- PLOTTING ---
    # We will only plot the top 5 equity curves to keep the chart legible
    top_5_labels = []
    for idx, row in results_df.head(5).iterrows():
         top_5_labels.append(f"m:{row.get('num_factors', '-')} | en:{row.get('entry_threshold', '-')} | ex:{row.get('exit_threshold', '-')}")
            
    plt.figure(figsize=(12, 8))
    for label in top_5_labels:
        if label in equity_curves:
            curve = equity_curves[label]
            plt.plot(curve.index, curve.values, label=f"{label}", linewidth=2)
            
    plt.axhline(0, color='black', linewidth=1)
    plt.title('Top 5 Hyperparameter Combinations (PCA Strategy)', fontsize=14)
    plt.xlabel('Date')
    plt.ylabel('Cumulative PnL (Units)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Optimization plot saved to {save_path}")
        
    plt.show()
    
    return results_df, equity_curves