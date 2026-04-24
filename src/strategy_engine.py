import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

# Local imports based on previous files
from .factor_models import get_rolling_universe, compute_rolling_split, compute_rolling_split_pca
from .math_utils import apply_volume_dampener

# ------------------------------------------------------------------------------
# 1. ORNSTEIN-UHLENBECK CALIBRATION
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
# 2. EXPLICITLY HEDGED TRADING SIMULATORS
# ------------------------------------------------------------------------------
def simulate_hedged_trading(test_stock_returns, test_etf_returns, test_residuals, 
                            ou_parameters, betas_record, t2e_dict, 
                            entry_threshold=1.25, exit_threshold=0.50):
    """Simulates ETF market-neutral bang-bang trading on the test window."""
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
            
            # --- PRODUCTION GUARD CLAUSE ---
            if pd.isna(residual_return):
                pnl_df.at[date, ticker] = 0.0
                continue # Path and position freeze for today
                
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


def simulate_hedged_trading_pca(test_stock_returns, test_residuals, 
                                ou_parameters, betas_record, F_test,
                                entry_threshold=1.25, exit_threshold=0.50):
    """Simulates PCA market-neutral bang-bang trading on the test window."""
    pnl_df = pd.DataFrame(0.0, index=test_residuals.index, columns=list(ou_parameters.keys()), dtype=float)
    
    for ticker, params in ou_parameters.items():
        m = params['m']
        sigma_eq = params['sigma_eq']
        X_prev = params['X_train_end']
        
        betas = betas_record[ticker]
        position = 0 
        
        for day_idx, (date, residual_return) in enumerate(test_residuals[ticker].items()):
            
            # --- PRODUCTION GUARD CLAUSE ---
            if pd.isna(residual_return):
                pnl_df.at[date, ticker] = 0.0
                continue # Path and position freeze for today
                
            stock_ret = test_stock_returns.at[date, ticker]
            hedge_ret = np.sum(betas * F_test[day_idx, :])
            
            X_current = X_prev + residual_return
            s_score = (X_current - m) / sigma_eq
            
            if position == 1:
                pnl_df.at[date, ticker] = stock_ret - hedge_ret
            elif position == -1:
                pnl_df.at[date, ticker] = -stock_ret + hedge_ret
                
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
# 3. THE MASTER ORCHESTRATORS
# ------------------------------------------------------------------------------
def run_full_strategy(returns_wide, returns_wide_etf, volume_wide, t2e_dict, 
                      train_days=252, test_days=21, 
                      entry_threshold=1.25, exit_threshold=0.50, 
                      adf_p_threshold=0.05, avg_window=60):
    
    returns_wide = returns_wide.sort_index()
    returns_wide_etf = returns_wide_etf.sort_index()
    total_trading_days = len(returns_wide)
    current_idx = 0
    all_portfolio_returns = []
    
    print(f"Starting rolling ETF backtest simulation (Train: {train_days}, Test: {test_days})...")
    
    while current_idx + train_days + test_days <= total_trading_days:
        train_start_idx = current_idx
        train_end_idx = current_idx + train_days - 1
        test_start_idx = current_idx + train_days
        test_end_idx = current_idx + train_days + test_days - 1 
        
        train_start = returns_wide.index[train_start_idx]
        train_end = returns_wide.index[train_end_idx]
        test_start = returns_wide.index[test_start_idx]
        test_end = returns_wide.index[test_end_idx]

        train_tickers, train_returns, test_returns_raw = get_rolling_universe(
            returns_wide, train_start, train_end, test_start, test_end
        )

        if len(train_tickers) > 0:
            
            # Apply Dampener strictly to TRAINING data
            adjusted_train_returns = apply_volume_dampener(
                train_returns, volume_wide, 
                train_start, train_end,
                avg_window=avg_window
            )
            
            # Pass pre-split data to ETF Regression
            residuals_train, residuals_test, betas = compute_rolling_split(
                adjusted_train_returns, test_returns_raw, returns_wide_etf, t2e_dict 
            )
            
            ou_params = calibrate_tradable_universe(
                residuals_train, adf_p_threshold=adf_p_threshold
            )
            
            # All stocks that passed calibration are theoretically tradable
            tradable_tickers = list(ou_params.keys())
            
            if len(tradable_tickers) > 0:
                tradable_ou_params = {t: ou_params[t] for t in tradable_tickers}
                tradable_residuals_test = residuals_test[tradable_tickers]
                
                # Raw test returns (containing NaNs) flow into the simulator
                clean_test_returns = test_returns_raw[tradable_tickers]
                
                daily_pnl = simulate_hedged_trading(
                    clean_test_returns, returns_wide_etf, tradable_residuals_test, 
                    tradable_ou_params, betas, t2e_dict,
                    entry_threshold=entry_threshold, exit_threshold=exit_threshold
                )
                all_portfolio_returns.append(daily_pnl)
            
        current_idx += test_days
        
    print("Simulation Complete.")
    strategy_returns = pd.concat(all_portfolio_returns, axis=0).sort_index()
    return strategy_returns


def run_full_strategy_pca(returns_wide, returns_wide_etf, volume_wide, t2e_dict, 
                          train_days=252, test_days=21, num_factors=10,
                          entry_threshold=1.25, exit_threshold=0.50,
                          adf_p_threshold=0.05, avg_window=60):
    
    returns_wide = returns_wide.sort_index()
    returns_wide_etf = returns_wide_etf.sort_index() # Unused here but kept for signature consistency
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
        
        train_tickers, train_returns, test_returns_raw = get_rolling_universe(
            returns_wide, train_start, train_end, test_start, test_end
        )
        
        if len(train_tickers) > 0:
            
            # Apply Dampener strictly to TRAINING data
            adjusted_train_returns = apply_volume_dampener(
                train_returns, volume_wide, 
                train_start, train_end,
                avg_window=avg_window
            )
            
            # Pass pre-split data to PCA Regression
            residuals_train, residuals_test, betas, F_test, _ = compute_rolling_split_pca(
                adjusted_train_returns, test_returns_raw, num_factors=num_factors 
            )
            
            ou_params = calibrate_tradable_universe(
                residuals_train, adf_p_threshold=adf_p_threshold
            )
            
            tradable_tickers = list(ou_params.keys())
            
            if len(tradable_tickers) > 0:
                tradable_ou_params = {t: ou_params[t] for t in tradable_tickers}
                tradable_residuals_test = residuals_test[tradable_tickers]
                clean_test_returns = test_returns_raw[tradable_tickers]
                
                daily_pnl = simulate_hedged_trading_pca(
                    clean_test_returns, tradable_residuals_test, 
                    tradable_ou_params, betas, F_test,
                    entry_threshold=entry_threshold, exit_threshold=exit_threshold
                )
                all_portfolio_returns.append(daily_pnl)
        
        current_idx += test_days
        
    print("Simulation Complete.")
    strategy_returns = pd.concat(all_portfolio_returns, axis=0).sort_index()
    return strategy_returns