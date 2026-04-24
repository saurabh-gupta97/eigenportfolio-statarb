import pandas as pd
import numpy as np

# ------------------------------------------------------------------------------
# 1. UNIVERSE FILTERING
# ------------------------------------------------------------------------------
def get_rolling_universe(returns_wide, train_start, train_end, test_start, test_end):
    """
    Identifies stocks with 100% data coverage strictly within the training window.
    Extracts those specific stocks for both the train and test windows.
    """
    # 1. Slice the raw data into train and test windows
    train_window_data = returns_wide.loc[train_start:train_end]
    test_window_data = returns_wide.loc[test_start:test_end]

    # 2. Drop any stock (column) that has a NaN in the TRAINING window
    # pandas dropna(axis=1) is highly optimized and perfectly handles this.
    clean_train_window_data = train_window_data.dropna(axis=1)
    
    # 3. Extract the list of surviving tickers
    valid_tickers = clean_train_window_data.columns.tolist()
    
    # 4. Apply this exact list of tickers to the TEST window
    # By passing the list of valid tickers, pandas automatically slices the test 
    # dataframe to only include those columns.
    clean_test_window_data = test_window_data[valid_tickers]
    
    return valid_tickers, clean_train_window_data, clean_test_window_data

# ------------------------------------------------------------------------------
# 2. THE ETF REGRESSION BASELINE
# ------------------------------------------------------------------------------
def compute_rolling_split(train_stocks, test_stocks_raw, returns_wide_etf, t2e_dict):
    """
    Runs OLS against the assigned Sector ETF. 
    Accepts pre-split train and test data to match the PCA function architecture.
    Allows NaNs in the test window to naturally flow through to the test residuals.
    """
    # 1. Align the ETF returns to the exact indices of the provided stock data
    train_etfs = returns_wide_etf.loc[train_stocks.index]
    test_etfs = returns_wide_etf.loc[test_stocks_raw.index]
    
    # 2. Pre-allocate DataFrames
    residuals_train = pd.DataFrame(index=train_stocks.index, columns=train_stocks.columns, dtype=float)
    
    # Notice we use test_stocks_raw.index, but train_stocks.columns. 
    # We only calculate residuals for stocks that survived the training window.
    residuals_test = pd.DataFrame(index=test_stocks_raw.index, columns=train_stocks.columns, dtype=float)
    
    betas_record = {} 
    
    for ticker in train_stocks.columns:
        # Get assigned ETF
        etf_list = t2e_dict.get(ticker, [])
        assigned_etf = etf_list[0] if len(etf_list) > 0 else "SPY"
        
        # --- TRAIN ---
        y_train = train_stocks[ticker].values
        x_train = train_etfs[assigned_etf].values
        
        X_train = np.vstack([np.ones(len(x_train)), x_train]).T
        coeffs, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
        alpha, beta = coeffs[0], coeffs[1]
        
        # Store beta in a dictionary keyed by ETF name to match the explicit hedger's expectations
        betas_record[ticker] = {assigned_etf: beta}
        residuals_train[ticker] = y_train - (alpha + beta * x_train)
        
        # --- TEST (Out of Sample) ---
        # We only process the test data if the stock actually exists in the raw test dataframe
        if ticker in test_stocks_raw.columns:
            y_test = test_stocks_raw[ticker].values
            x_test = test_etfs[assigned_etf].values
            
            # If the stock halted and y_test is NaN, the math becomes: NaN - float = NaN.
            # This perfectly flags it for the simulator's guard clause.
            residuals_test[ticker] = y_test - (alpha + beta * x_test)
            
    return residuals_train, residuals_test, betas_record

# ------------------------------------------------------------------------------
# 3. THE PCA REGRESSION
# ------------------------------------------------------------------------------
def compute_rolling_split_pca(train_stocks, test_stocks_raw, num_factors=10):
    """
    Trains PCA on train_stocks. 
    Calculates test factor returns day-by-day by dynamically slicing the weight 
    matrix to exclude stocks with NaN returns on that specific day.
    """
    # ---------------------------------------------------------
    # 1. PCA on Training Window
    # ---------------------------------------------------------
    train_mean = train_stocks.mean()
    train_std = train_stocks.std()
    
    # Standardize to force variance = 1
    Y_train = (train_stocks - train_mean) / train_std

    # Compute Correlation Matrix
    corr_matrix = Y_train.corr().values

    # Extract Eigenvalues and Eigenvectors
    # eigh is highly optimized for symmetric matrices like correlation matrices
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    
    # eigh returns them in ascending order. We need descending (largest variance first)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Keep only the top 'm' factors
    top_eigenvectors = eigenvectors[:, :num_factors]

    # Avellaneda-Lee Eq 9: F_jk = Sum( v_i / sigma_i * R_ik )
    # This transforms abstract eigenvectors into physical portfolio returns
    # Shape: (N_train, m)
    portfolio_weights = top_eigenvectors / train_std.values[:, np.newaxis]
    F_train = np.dot(train_stocks.values, portfolio_weights)
    
    # ---------------------------------------------------------
    # 2. Dynamic Factor Calculation for Test Window (No Zero-Fills)
    # ---------------------------------------------------------
    F_test_list = []
    
    # Iterate through the test window day-by-day (typically ~21 iterations)
    for date, row in test_stocks_raw.iterrows():
        
        # Create a boolean mask of non-NaN values for this specific day
        # Shape: (N_train,)
        valid_mask = row.notna().values
        
        # Dynamically slice the returns and the weight matrix
        # This completely drops the rows/columns of the missing stocks
        valid_returns = row.values[valid_mask]                    # Shape: (N_valid,)
        valid_weights = portfolio_weights[valid_mask, :]          # Shape: (N_valid, m)
        
        if valid_returns.shape[0] > 0:
            # (1, N_valid) dot (N_valid, m) -> (1, m)
            daily_F = np.dot(valid_returns, valid_weights)
        else:
            # Failsafe: if every single stock is NaN (e.g., market holiday anomaly)
            daily_F = np.zeros(num_factors)
            
        F_test_list.append(daily_F)
        
    # Convert back to a 2D numpy array. Shape: (T_test, m)
    F_test = np.array(F_test_list)
    
    # ---------------------------------------------------------
    # 3. Multi-Factor Regression
    # ---------------------------------------------------------
    residuals_train = pd.DataFrame(index=train_stocks.index, columns=train_stocks.columns)
    residuals_test = pd.DataFrame(index=test_stocks_raw.index, columns=test_stocks_raw.columns)
    
    X_train_reg = np.hstack([np.ones((len(F_train), 1)), F_train])
    X_test_reg = np.hstack([np.ones((len(F_test), 1)), F_test])
    
    betas_record = {}
    
    for ticker in train_stocks.columns:
        y_train = train_stocks[ticker].values
        
        # Fit OLS on the training window
        coeffs, _, _, _ = np.linalg.lstsq(X_train_reg, y_train, rcond=None)
        
        # Train residual
        residuals_train[ticker] = y_train - np.dot(X_train_reg, coeffs)
        
        # Test residual
        # Because we are using the raw test stocks, if the raw return is NaN, 
        # the math naturally becomes: NaN - Predicted_Float = NaN.
        residuals_test[ticker] = test_stocks_raw[ticker].values - np.dot(X_test_reg, coeffs)
        
        # Store the m factor loadings (excluding the alpha intercept)
        betas_record[ticker] = coeffs[1:]
        
    residuals_train = residuals_train.astype(float)
    residuals_test = residuals_test.astype(float)
        
    return residuals_train, residuals_test, betas_record, F_test, portfolio_weights