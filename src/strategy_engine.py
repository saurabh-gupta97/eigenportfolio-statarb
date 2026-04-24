import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

import itertools

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
# 3. SINGLE WINDOW EVALUATION
# ------------------------------------------------------------------------------
def evaluate_window(returns_wide, volume_wide, returns_wide_etf, t2e_dict,
                        train_start, train_end, test_start, test_end, 
                        entry_threshold=1.25, exit_threshold=0.50, 
                        adf_p_threshold=0.05, avg_window=60):
    """
    The 'Atom' of the ETF backtest. Executes the entire pipeline for a specific 
    train/test date slice and a specific set of hyperparameters.
    """
    train_tickers, train_returns, test_returns_raw = get_rolling_universe(
        returns_wide, train_start, train_end, test_start, test_end
    )
    
    if len(train_tickers) == 0:
        return pd.Series(dtype=float)
        
    # 1. Volume Dampener (Strictly Training Data)
    adjusted_train_returns = apply_volume_dampener(
        train_returns, volume_wide, train_start, train_end, avg_window=avg_window
    )
    
    # 2. ETF Regression Engine
    residuals_train, residuals_test, betas = compute_rolling_split(
        adjusted_train_returns, test_returns_raw, returns_wide_etf, t2e_dict 
    )
    
    # 3. Physics Calibration
    ou_params = calibrate_tradable_universe(
        residuals_train, adf_p_threshold=adf_p_threshold
    )
    
    tradable_tickers = list(ou_params.keys())
    
    if len(tradable_tickers) == 0:
        return pd.Series(dtype=float)
        
    # 4. Test Window Survivorship & Simulation
    tradable_ou_params = {t: ou_params[t] for t in tradable_tickers}
    tradable_residuals_test = residuals_test[tradable_tickers]
    clean_test_returns = test_returns_raw[tradable_tickers]
    
    daily_pnl = simulate_hedged_trading(
        clean_test_returns, returns_wide_etf, tradable_residuals_test, 
        tradable_ou_params, betas, t2e_dict,
        entry_threshold=entry_threshold, exit_threshold=exit_threshold
    )
    
    return daily_pnl


def evaluate_window_pca(returns_wide, volume_wide, 
                        train_start, train_end, test_start, test_end, 
                        num_factors, entry_threshold, exit_threshold, 
                        adf_p_threshold=0.05, avg_window=60):
    """
    The 'Atom' of the backtest. Executes the entire pipeline for a specific 
    train/test date slice and a specific set of hyperparameters.
    """
    train_tickers, train_returns, test_returns_raw = get_rolling_universe(
        returns_wide, train_start, train_end, test_start, test_end
    )
    
    if len(train_tickers) == 0:
        return pd.Series(dtype=float)
        
    # 1. Volume Dampener (Strictly Training Data)
    adjusted_train_returns = apply_volume_dampener(
        train_returns, volume_wide, train_start, train_end, avg_window=avg_window
    )
    
    # 2. PCA Engine
    residuals_train, residuals_test, betas, F_test, _ = compute_rolling_split_pca(
        adjusted_train_returns, test_returns_raw, num_factors=num_factors 
    )
    
    # 3. Physics Calibration
    ou_params = calibrate_tradable_universe(
        residuals_train, adf_p_threshold=adf_p_threshold
    )
    
    tradable_tickers = list(ou_params.keys())
    
    if len(tradable_tickers) == 0:
        return pd.Series(dtype=float)
        
    # 4. Test Window Survivorship & Simulation
    tradable_ou_params = {t: ou_params[t] for t in tradable_tickers}
    tradable_residuals_test = residuals_test[tradable_tickers]
    clean_test_returns = test_returns_raw[tradable_tickers]
    
    daily_pnl = simulate_hedged_trading_pca(
        clean_test_returns, tradable_residuals_test, 
        tradable_ou_params, betas, F_test,
        entry_threshold=entry_threshold, exit_threshold=exit_threshold
    )
    
    return daily_pnl

# ------------------------------------------------------------------------------
# 4. THE MASTER ORCHESTRATORS
# ------------------------------------------------------------------------------
def run_full_strategy(returns_wide, volume_wide, returns_wide_etf, t2e_dict, param_grid,
                          train_days=252, val_days=63, test_days=21):
    """
    Executes a nested Walk-Forward Optimization for the ETF strategy.
    If no tuning is desired, pass a param_grid with a single combination.
    """
    returns_wide = returns_wide.sort_index()
    returns_wide_etf = returns_wide_etf.sort_index()
    total_trading_days = len(returns_wide)
    current_idx = 0
    
    final_oos_portfolio_returns = []
    
    # Unpack the parameter grid
    keys = param_grid.keys()
    combinations = list(itertools.product(*param_grid.values()))
    param_list = [dict(zip(keys, comp)) for comp in combinations]
    
    print(f"Starting ETF Walk-Forward Simulation ({len(param_list)} param combinations)...")
    
    while current_idx + train_days + val_days + test_days <= total_trading_days:
        
        # ==========================================
        # PHASE 1: INNER VALIDATION LOOP
        # ==========================================
        val_train_start = returns_wide.index[current_idx]
        val_train_end = returns_wide.index[current_idx + train_days - 1]
        val_test_start = returns_wide.index[current_idx + train_days]
        val_test_end = returns_wide.index[current_idx + train_days + val_days - 1]
        
        best_sharpe = -np.inf
        best_params = param_list[0]
        
        # If there's only 1 parameter set, we can skip the validation loop processing
        # and just assign it directly to save compute time.
        if len(param_list) > 1:
            print(f"[Validation Phase] Tuning on {val_test_start.date()} to {val_test_end.date()}")
            for params in param_list:
                val_pnl = evaluate_window(
                    returns_wide, volume_wide, returns_wide_etf, t2e_dict,
                    val_train_start, val_train_end, val_test_start, val_test_end,
                    **params
                )
                
                if len(val_pnl) > 0:
                    sharpe = np.sqrt(252) * (val_pnl.mean() / val_pnl.std()) if val_pnl.std() > 0 else 0
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = params
            print(f"Selected Best Params: {best_params} (Val Sharpe: {best_sharpe:.2f})")
        else:
            best_params = param_list[0]
            
        # ==========================================
        # PHASE 2: OUTER TEST LOOP (Strictly Blind)
        # ==========================================
        true_train_start = returns_wide.index[current_idx + val_days]
        true_train_end = returns_wide.index[current_idx + val_days + train_days - 1]
        true_test_start = returns_wide.index[current_idx + val_days + train_days]
        true_test_end = returns_wide.index[current_idx + val_days + train_days + test_days - 1]
        
        print(f"[Testing Phase] Executing blindly on {true_test_start.date()} to {true_test_end.date()}")
        
        oos_pnl = evaluate_window(
            returns_wide, volume_wide, returns_wide_etf, t2e_dict,
            true_train_start, true_train_end, true_test_start, true_test_end,
            **best_params
        )
        
        if len(oos_pnl) > 0:
            final_oos_portfolio_returns.append(oos_pnl)
            
        current_idx += test_days
        
    print("\nETF Walk-Forward Simulation Complete.")
    strategy_returns = pd.concat(final_oos_portfolio_returns, axis=0).sort_index()
    
    final_sharpe = np.sqrt(252) * (strategy_returns.mean() / strategy_returns.std())
    print(f"FINAL OUT-OF-SAMPLE ETF SHARPE: {final_sharpe:.4f}")
    
    return strategy_returns


def run_full_strategy_pca(returns_wide, volume_wide, param_grid,
                                  train_days=252, val_days=63, test_days=21):
    """
    Executes a nested Walk-Forward Optimization to eliminate in-sample bias.
    val_days (e.g., 63 days / 3 months) is the window used to pick the best params.
    """
    returns_wide = returns_wide.sort_index()
    total_trading_days = len(returns_wide)
    current_idx = 0
    
    final_oos_portfolio_returns = []
    
    # Unpack the parameter grid into a list of dictionaries
    keys = param_grid.keys()
    combinations = list(itertools.product(*param_grid.values()))
    param_list = [dict(zip(keys, comp)) for comp in combinations]
    
    print(f"Starting Walk-Forward Optimization ({len(param_list)} param combinations)...")
    
    # Loop requires enough data for Train + Validation + Test
    while current_idx + train_days + val_days + test_days <= total_trading_days:
        
        # ==========================================
        # PHASE 1: INNER VALIDATION LOOP
        # ==========================================
        val_train_start = returns_wide.index[current_idx]
        val_train_end = returns_wide.index[current_idx + train_days - 1]
        val_test_start = returns_wide.index[current_idx + train_days]
        val_test_end = returns_wide.index[current_idx + train_days + val_days - 1]
        
        print(f"\n[Validation Phase] Tuning on {val_test_start.date()} to {val_test_end.date()}")
        
        best_sharpe = -np.inf
        best_params = param_list[0]
        
        for params in param_list:
            val_pnl = evaluate_window_pca(
                returns_wide, volume_wide, 
                val_train_start, val_train_end, val_test_start, val_test_end,
                **params
            )
            
            if len(val_pnl) > 0:
                sharpe = np.sqrt(252) * (val_pnl.mean() / val_pnl.std()) if val_pnl.std() > 0 else 0
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params
                    
        print(f"Selected Best Params: {best_params} (Val Sharpe: {best_sharpe:.2f})")
        
        # ==========================================
        # PHASE 2: OUTER TEST LOOP (Strictly Blind)
        # ==========================================
        # Slide the training window forward so it is strictly adjacent to the true test window
        true_train_start = returns_wide.index[current_idx + val_days]
        true_train_end = returns_wide.index[current_idx + val_days + train_days - 1]
        true_test_start = returns_wide.index[current_idx + val_days + train_days]
        true_test_end = returns_wide.index[current_idx + val_days + train_days + test_days - 1]
        
        print(f"[Testing Phase] Executing blindly on {true_test_start.date()} to {true_test_end.date()}")
        
        oos_pnl = evaluate_window_pca(
            returns_wide, volume_wide,
            true_train_start, true_train_end, true_test_start, true_test_end,
            **best_params
        )
        
        if len(oos_pnl) > 0:
            final_oos_portfolio_returns.append(oos_pnl)
            
        # Step the entire massive block forward by one Test Month
        current_idx += test_days
        
    print("\nWalk-Forward Optimization Complete.")
    strategy_returns = pd.concat(final_oos_portfolio_returns, axis=0).sort_index()
    
    # Final purely out-of-sample Sharpe
    final_sharpe = np.sqrt(252) * (strategy_returns.mean() / strategy_returns.std())
    print(f"FINAL OUT-OF-SAMPLE SHARPE: {final_sharpe:.4f}")
    
    return strategy_returns


