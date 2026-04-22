import pandas as pd
import numpy as np

# ------------------------------------------------------------------------------
# 1. UNIVERSE FILTERING
# ------------------------------------------------------------------------------
def get_rolling_universe(returns_wide, train_start, test_end):
    """Identifies stocks with 100% data coverage across the rolling window."""
    window_data = returns_wide.loc[train_start:test_end]
    
    # Pandas dropna(axis=1) is highly optimized to drop columns with any NaNs
    clean_window_returns = window_data.dropna(axis=1)
    valid_tickers = clean_window_returns.columns.tolist()
    
    return valid_tickers, clean_window_returns

# ------------------------------------------------------------------------------
# 2. THE ETF REGRESSION BASELINE
# ------------------------------------------------------------------------------
def compute_rolling_split(clean_stock_returns, etf_returns, t2e_dict, 
                                       train_start, train_end, test_start, test_end):
    """
    Runs OLS against the assigned Sector ETF. 
    Returns BOTH the training residuals (for OU calibration) 
    and testing residuals (for out-of-sample trading), plus the betas.
    """
    train_stocks = clean_stock_returns.loc[train_start:train_end]
    test_stocks = clean_stock_returns.loc[test_start:test_end]
    train_etfs = etf_returns.loc[train_start:train_end]
    test_etfs = etf_returns.loc[test_start:test_end]
    
    # Pre-allocate numeric arrays with dtype=float for speed
    residuals_train = pd.DataFrame(index=train_stocks.index, columns=train_stocks.columns, dtype=float)
    residuals_test = pd.DataFrame(index=test_stocks.index, columns=test_stocks.columns, dtype=float)
    betas_record = {} 
    
    for ticker in train_stocks.columns:
        etf_list = t2e_dict.get(ticker, [])
        assigned_etf = etf_list[0] if len(etf_list) > 0 else "SPY"
        
        # --- TRAIN ---
        y_train = train_stocks[ticker].values
        x_train = train_etfs[assigned_etf].values
        X_train = np.vstack([np.ones(len(x_train)), x_train]).T
        
        coeffs, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
        alpha, beta = coeffs[0], coeffs[1]
        
        betas_record[ticker] = {assigned_etf: beta}
        residuals_train[ticker] = y_train - (alpha + beta * x_train)
        
        # --- TEST (Out of Sample) ---
        y_test = test_stocks[ticker].values
        x_test = test_etfs[assigned_etf].values
        residuals_test[ticker] = y_test - (alpha + beta * x_test)
        
    return residuals_train, residuals_test, betas_record

# ------------------------------------------------------------------------------
# 3. THE PCA REGRESSION
# ------------------------------------------------------------------------------
def compute_rolling_split_pca(clean_stock_returns, train_start, train_end, test_start, test_end, num_factors=15):
    """
    Extracts statistical factors via PCA on the correlation matrix of the training window.
    Regresses stocks against the top 'num_factors' eigenportfolios to generate residuals.
    """
    train_stocks = clean_stock_returns.loc[train_start:train_end]
    test_stocks = clean_stock_returns.loc[test_start:test_end]
    
    # ---------------------------------------------------------
    # STEP 1: PCA on the Correlation Matrix (Training Window)
    # ---------------------------------------------------------
    # Standardize training returns to force variance = 1
    train_mean = train_stocks.mean()
    train_std = train_stocks.std()
    Y_train = (train_stocks - train_mean) / train_std
    
    # Compute Correlation Matrix
    corr_matrix = Y_train.corr().values
    
    # Extract Eigenvalues and Eigenvectors
    # eigh is highly optimized for symmetric matrices like correlation matrices
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    
    # eigh returns them in ascending order. We need descending (largest variance first)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Keep only the top 'm' factors
    top_eigenvectors = eigenvectors[:, :num_factors]
    
    # ---------------------------------------------------------
    # STEP 2: Construct the Eigenportfolio Returns (F_j)
    # ---------------------------------------------------------
    # Avellaneda-Lee Eq 9: F_jk = Sum( v_i / sigma_i * R_ik )
    # This transforms abstract eigenvectors into physical portfolio returns
    
    # Shape of top_eigenvectors is (N_stocks, num_factors)
    # We divide each row by its stock's physical volatility (train_std)
    portfolio_weights = top_eigenvectors / train_std.values[:, np.newaxis]
    
    # Multiply the raw training returns (T x N) by the weights (N x m)
    # Resulting F_train is (T x m), the daily returns of our 15 hidden factors
    F_train = np.dot(train_stocks.values, portfolio_weights)
    
    # Do the same for the Test window (using the FROZEN training weights!)
    F_test = np.dot(test_stocks.values, portfolio_weights)
    
    # ---------------------------------------------------------
    # STEP 3: Multi-Factor Regression via Vectorized SVD
    # ---------------------------------------------------------
    # Add a column of 1s for the intercept
    X_train_reg = np.hstack([np.ones((len(F_train), 1)), F_train])
    X_test_reg = np.hstack([np.ones((len(F_test), 1)), F_test])
    
    # Vectorized lstsq: Solves for ALL stocks simultaneously!
    # train_stocks.values shape is (T, N). coeffs shape will be (m+1, N)
    coeffs, _, _, _ = np.linalg.lstsq(X_train_reg, train_stocks.values, rcond=None)
    
    # Calculate residuals for all stocks simultaneously using dot product
    res_train_matrix = train_stocks.values - np.dot(X_train_reg, coeffs)
    res_test_matrix = test_stocks.values - np.dot(X_test_reg, coeffs)
    
    # Rebuild DataFrames from the computed matrices
    residuals_train = pd.DataFrame(res_train_matrix, index=train_stocks.index, columns=train_stocks.columns)
    residuals_test = pd.DataFrame(res_test_matrix, index=test_stocks.index, columns=test_stocks.columns)
    
    # Build the betas_record dictionary from the coeffs matrix
    # coeffs[0, :] are the alphas. coeffs[1:, :] are the 15 factor betas.
    betas_record = {ticker: coeffs[1:, i] for i, ticker in enumerate(train_stocks.columns)}
    
    # We also return F_test so the simulator knows the returns of the hedging portfolios
    return residuals_train, residuals_test, betas_record, F_test, portfolio_weights