import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

# Local imports based on previous files
from .factor_models import get_rolling_universe, compute_rolling_split, compute_rolling_split_pca
from .math_utils import apply_volume_dampener

# ------------------------------------------------------------------------------
# 3. ORNSTEIN-UHLENBECK CALIBRATION
# ------------------------------------------------------------------------------
def calibrate_tradable_universe(train_residuals, adf_p_threshold=0.05):
    """Integrates residuals, runs ADF, and fits the OU process physics."""
    X_train = train_residuals.cumsum(axis=0)
    ou_parameters = {}
    
    for ticker in X_train.columns:
        path = X_train[ticker].values
        
        try:
            adf_stat, p_value, _, _, _, _ = adfuller(path, autolag='AIC')
        except Exception:
            continue 
            
        if p_value < adf_p_threshold:
            x_t = path[:-1]
            x_t_plus_1 = path[1:]
            
            A = np.vstack([np.ones(len(x_t)), x_t]).T
            coeffs, _, _, _ = np.linalg.lstsq(A, x_t_plus_1, rcond=None)
            a, b = coeffs
            
            if b <= 0 or b >= 1:
                continue
                
            epsilon = x_t_plus_1 - (a + b * x_t)
            var_epsilon = np.var(epsilon)
            
            m = a / (1 - b)
            sigma_eq = np.sqrt(var_epsilon / (1 - b**2))
            kappa = -np.log(b) * 252
            
            ou_parameters[ticker] = {
                'm': m,
                'sigma_eq': sigma_eq,
                'kappa': kappa,
                'X_train_end': path[-1] 
            }
    return ou_parameters

# ------------------------------------------------------------------------------
# 4. EXPLICITLY HEDGED TRADING SIMULATOR
# ------------------------------------------------------------------------------
def simulate_hedged_trading(test_stock_returns, test_etf_returns, test_residuals, 
                            ou_parameters, betas_record, t2e_dict, 
                            entry_threshold=1.25, exit_threshold=0.50):
    """Simulates market-neutral bang-bang trading on the test window."""
    pnl_df = pd.DataFrame(0.0, index=test_residuals.index, columns=list(ou_parameters.keys()), dtype=float)
    
    for ticker, params in ou_parameters.items():
        m = params['m']
        sigma_eq = params['sigma_eq']
        X_prev = params['X_train_end']
        
        etf_list = t2e_dict.get(ticker, [])
        assigned_etf = etf_list[0] if len(etf_list) > 0 else "SPY"
        beta = betas_record.get(ticker, {}).get(assigned_etf, 0.0)
        
        position = 0 
        
        for date, residual_return in test_residuals[ticker].items():
            stock_ret = test_stock_returns.at[date, ticker]
            etf_ret = test_etf_returns.at[date, assigned_etf]
            
            X_current = X_prev + residual_return
            s_score = (X_current - m) / sigma_eq
            
            if position == 1:
                pnl_df.at[date, ticker] = (1.0 * stock_ret) - (beta * etf_ret)
            elif position == -1:
                pnl_df.at[date, ticker] = (-1.0 * stock_ret) + (beta * etf_ret)
                
            if position == 0:
                if s_score < -entry_threshold:
                    position = 1  
                elif s_score > entry_threshold:
                    position = -1 
            elif position == 1:
                if s_score > -exit_threshold:
                    position = 0  
            elif position == -1:
                if s_score < exit_threshold:
                    position = 0  
                    
            X_prev = X_current
            
    portfolio_daily_pnl = pnl_df.sum(axis=1)
    return portfolio_daily_pnl
    
# ------------------------------------------------------------------------------
# 5. THE MASTER ORCHESTRATOR
# ------------------------------------------------------------------------------
def run_full_strategy(returns_wide, etf_returns, volume_wide, t2e_dict, 
                      train_days=252, test_days=21, 
                      entry_threshold=1.25, exit_threshold=0.50, 
                      adf_p_threshold=0.05, avg_window=60):
    """Steps the rolling window forward and aggregates out-of-sample portfolio returns."""
    returns_wide = returns_wide.sort_index()
    etf_returns = etf_returns.sort_index()
    total_trading_days = len(returns_wide)
    current_idx = 0
    
    all_portfolio_returns = []
    
    print(f"Starting rolling backtest simulation (Train: {train_days}, Test: {test_days})...")
    
    while current_idx + train_days + test_days <= total_trading_days:
        train_start_idx = current_idx
        train_end_idx = current_idx + train_days - 1
        test_start_idx = current_idx + train_days
        test_end_idx = current_idx + train_days + test_days - 1 
        
        train_start = returns_wide.index[train_start_idx]
        train_end = returns_wide.index[train_end_idx]
        test_start = returns_wide.index[test_start_idx]
        test_end = returns_wide.index[test_end_idx]

        valid_tickers, clean_window_returns = get_rolling_universe(returns_wide, train_start, test_end)

        if len(valid_tickers) > 0:
        
            # 1. Apply the Dampener FIRST
            adjusted_window_returns = apply_volume_dampener(
                clean_window_returns, volume_wide, 
                train_start, test_end,
                avg_window=avg_window
            )
            
            # 2. Pass the ADJUSTED returns to the ETF Regression
            residuals_train, residuals_test, betas = compute_rolling_split(
                adjusted_window_returns, etf_returns, t2e_dict, 
                train_start, train_end, test_start, test_end
            )
            
            ou_params = calibrate_tradable_universe(
                residuals_train, 
                adf_p_threshold=adf_p_threshold
            )
            
            if len(ou_params) > 0:
                # We must pass the raw returns to the explicitly hedged simulator
                test_stock_returns = clean_window_returns.loc[test_start:test_end]
                
                daily_pnl = simulate_hedged_trading(
                    test_stock_returns, etf_returns, residuals_test, 
                    ou_params, betas, t2e_dict,
                    entry_threshold=entry_threshold, exit_threshold=exit_threshold
                )
                all_portfolio_returns.append(daily_pnl)
            
        current_idx += test_days
        
    print("Simulation Complete.")
    
    strategy_returns = pd.concat(all_portfolio_returns, axis=0).sort_index()
    return strategy_returns

def simulate_hedged_trading_pca(test_stock_returns, test_residuals, 
                                ou_parameters, betas_record, F_test,
                                entry_threshold=1.25, exit_threshold=0.50):
    
    pnl_df = pd.DataFrame(0.0, index=test_residuals.index, columns=list(ou_parameters.keys()), dtype=float)
    
    for ticker, params in ou_parameters.items():
        m = params['m']
        sigma_eq = params['sigma_eq']
        X_prev = params['X_train_end']
        
        # Array of factor beta loadings for this specific stock
        betas = betas_record[ticker]
        
        position = 0 
        
        # F_test contains the returns of the factors for each day
        for day_idx, (date, residual_return) in enumerate(test_residuals[ticker].items()):
            stock_ret = test_stock_returns.at[date, ticker]
            
            # The return of our hedging basket is the sum of (Beta_j * F_j)
            hedge_ret = np.sum(betas * F_test[day_idx, :])
            
            X_current = X_prev + residual_return
            s_score = (X_current - m) / sigma_eq
            
            # Calculate Explicit Hedged PnL
            if position == 1:
                # Long Stock, Short the Factors
                pnl_df.at[date, ticker] = stock_ret - hedge_ret
            elif position == -1:
                # Short Stock, Long the Factors
                pnl_df.at[date, ticker] = -stock_ret + hedge_ret
                
            # Update Position State Machine
            if position == 0:
                if s_score < -entry_threshold:
                    position = 1  
                elif s_score > entry_threshold:
                    position = -1 
            elif position == 1:
                if s_score > -exit_threshold:
                    position = 0  
            elif position == -1:
                if s_score < exit_threshold:
                    position = 0  
                    
            X_prev = X_current
            
    portfolio_daily_pnl = pnl_df.sum(axis=1)
    return portfolio_daily_pnl

def run_full_strategy_pca(returns_wide, etf_returns, volume_wide, t2e_dict, 
                          train_days=252, test_days=21, num_factors=3,
                          entry_threshold=1.25, exit_threshold=0.50,
                          adf_p_threshold=0.05, avg_window=60):
    """Steps the rolling window forward and aggregates out-of-sample portfolio returns."""
    returns_wide = returns_wide.sort_index()
    etf_returns = etf_returns.sort_index()
    total_trading_days = len(returns_wide)
    current_idx = 0
    
    all_portfolio_returns = []
    
    print(f"Starting rolling PCA backtest simulation ({num_factors} Factors)...")
    
    while current_idx + train_days + test_days <= total_trading_days:
        train_start_idx = current_idx
        train_end_idx = current_idx + train_days - 1
        test_start_idx = current_idx + train_days
        test_end_idx = current_idx + train_days + test_days - 1 
        
        train_start = returns_wide.index[train_start_idx]
        train_end = returns_wide.index[train_end_idx]
        test_start = returns_wide.index[test_start_idx]
        test_end = returns_wide.index[test_end_idx]
        
        valid_tickers, clean_window_returns = get_rolling_universe(returns_wide, train_start, test_end)
        
        if len(valid_tickers) > 0:
            
            # --- VOLUME ADJUSTMENT STEP ---
            adjusted_window_returns = apply_volume_dampener(
                clean_window_returns, volume_wide, 
                train_start, test_end,
                avg_window=avg_window
            )
            
            # Now, pass the ADJUSTED returns into your PCA engine
            residuals_train, residuals_test, betas, F_test, _ = compute_rolling_split_pca(
                adjusted_window_returns,
                train_start, train_end, test_start, test_end, 
                num_factors=num_factors 
            )
            
            ou_params = calibrate_tradable_universe(
                residuals_train,
                adf_p_threshold=adf_p_threshold
            )
            
            if len(ou_params) > 0:
                # The simulator still uses the RAW stock returns to calculate actual dollar PnL
                test_stock_returns = clean_window_returns.loc[test_start:test_end]
                
                daily_pnl = simulate_hedged_trading_pca(
                    test_stock_returns, residuals_test, 
                    ou_params, betas, F_test,
                    entry_threshold=entry_threshold, exit_threshold=exit_threshold
                )
                all_portfolio_returns.append(daily_pnl)
        
        current_idx += test_days
        
    print("Simulation Complete.")
    
    strategy_returns = pd.concat(all_portfolio_returns, axis=0).sort_index()
    return strategy_returns