import pandas as pd
import numpy as np
import os
import yfinance as yf

def download_raw(start_date="2019-01-01", end_date="2024-01-01", file_path="data/raw/raw_data.csv"):
    # Full list of S&P 100 Tickers
    tickers = [
        "AAPL", "ABBV", "ABT"  , "ACN" , "ADBE", "AMAT", "AMD", "AMGN", "AMT"  , "AMZN",
        "AVGO", "AXP" , "BA"   , "BAC" , "BK"  , "BKNG", "BLK", "BMY" , "BRK-B", "C"   ,
        "CAT" , "CL"  , "CMCSA", "COF" , "COP" , "COST", "CRM", "CSCO", "CVS"  , "CVX" ,
        "DE"  , "DHR" , "DIS"  , "DUK" , "EMR" , "FDX" , "GD" , "GE"  , "GILD" , "GEV" ,
        "GM"  , "GOOG", "GOOGL", "GS"  , "HD"  , "HON" , "IBM", "INTC", "INTU" , "ISRG",
        "JNJ" , "JPM" , "KO"   , "LIN" , "LLY" , "LMT" , "LOW", "LRCX", "MA"   , "MCD" ,
        "MDLZ", "MDT" , "META" , "MMM" , "MO"  , "MRK" , "MS" , "MSFT", "MU"   , "NEE" ,
        "NFLX", "NKE" , "NOW"  , "NVDA", "ORCL", "PEP" , "PFE", "PG"  , "PLTR" , "PM"  ,
        "QCOM", "RTX" , "SBUX" , "SCHW", "SO"  , "SPG" , "T"  , "TMO" , "TMUS" , "TSLA",
        "TXN" , "UBER", "UNH"  , "UNP" , "UPS" , "USB" , "V"  , "VZ"  , "WFC"  , "WMT" , 
        "XOM" ,
    ]
    
    print(f"Downloading data for {len(tickers)} tickers...")
    
    raw = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        group_by="column"
    )

    if raw.empty:
        raise ValueError("Downloaded data is empty.")
    
    # Save locally to avoid re-downloading
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    raw.to_csv(file_path)
    
    print(f"Data saved to {file_path}")
    return raw

def load_raw(file_path="data/raw/raw_data.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find {file_path}. Ensure it is in your project directory.")
        
    print(f"Loading local data from {file_path}...")
    
    # Adding header=[0, 1] reads the yfinance 2-row header (Price Type, Ticker) correctly.
    # This also fixes the date parsing warning because it won't try to parse headers as dates.
    raw = pd.read_csv(file_path, header=[0, 1], index_col=0, parse_dates=True) 
    
    print(f"Original data shape: {raw.shape}")
    print("Raw data loading complete.")
    return raw

def get_prices_returns_volume(raw, prices_file_path="data/raw/prices.csv", returns_file_path="data/raw/returns.csv", volume_file_path="data/raw/volume.csv", save=True):
    
    # Because we read the CSV with a MultiIndex header, we can just extract these cleanly.
    prices = raw['Close'].copy()
    volume = raw['Volume'].copy()
    
    # name the index
    prices.index.name = "date"
    volume.index.name = "date"
    
    # convert values to numeric
    prices = prices.apply(pd.to_numeric, errors="coerce")
    volume = volume.apply(pd.to_numeric, errors="coerce")

    # Calculate daily log returns: ln(P_t / P_{t-1})
    prices = prices.replace([np.inf, -np.inf], np.nan)
    
    log_prices = np.log(prices)
    returns = log_prices.diff().iloc[1:]
    
    print(f"Returns calculated. Shape: {returns.shape}")
    
    if save:
        os.makedirs(os.path.dirname(prices_file_path), exist_ok=True)
        prices.to_csv(prices_file_path)
        print(f"Price data saved to {prices_file_path}")

        returns.to_csv(returns_file_path)
        print(f"Returns data saved to {returns_file_path}")
    
        volume.to_csv(volume_file_path)
        print(f"Volume data saved to {volume_file_path}")
    
    return prices, returns, volume

def get_prices_long(prices_wide, file_path="data/raw/prices_long.csv", save=True):
    prices_wide = prices_wide.sort_index()
    prices_wide.index.name = "date"
    
    prices_long = (
        prices_wide
        .reset_index()
        .melt(id_vars="date", var_name="ticker", value_name="adj_close")
    )
    
    prices_long["adj_close"] = pd.to_numeric(prices_long["adj_close"], errors="coerce")

    if save:
        prices_long.to_csv(file_path, index=False)
        print(f"Prices data saved to {file_path} in long-format")
        
    return prices_long

def get_returns_long(returns_wide, file_path="data/raw/returns_long.csv", save=True):
    returns_wide = returns_wide.sort_index()
    returns_wide.index.name = "date"

    returns_long = (
        returns_wide
        .reset_index()
        .melt(id_vars="date", var_name="ticker", value_name="return")
    )

    if save:
        returns_long.to_csv(file_path, index=False)
        print(f"Returns data saved to {file_path} in long-format")
        
    return returns_long

def get_volume_long(volume_wide, file_path="data/raw/volume_long.csv", save=True):
    volume_wide = volume_wide.sort_index()
    volume_wide.index.name = "date"
    
    volume_long = (
        volume_wide
        .reset_index()
        .melt(id_vars="date", var_name="ticker", value_name="volume")
    )
    
    volume_long["volume"] = pd.to_numeric(volume_long["volume"], errors="coerce")

    if save:
        volume_long.to_csv(file_path, index=False)
        print(f"Volume data saved to {file_path} in long-format")
        
    return volume_long

def get_metadata(df, file_path="data/raw/universe_metadata.csv", save=True):
    def longest_missing_streak(mask):
        max_streak = current = 0
        for x in mask:
            if x:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak
    
    dates = df.index
    expected_days = len(dates)
    
    # Check if we can load static info locally to avoid hitting Yahoo Finance
    local_metadata_exists = os.path.exists(file_path)
    if local_metadata_exists:
        existing_meta = pd.read_csv(file_path).set_index("ticker")
        print("Local metadata found. Updating statistical metrics without network calls.")

    report = []
    for ticker in df.columns:
        c = df[ticker]
        non_na = c.notna().sum()
        missing_mask = c.isna().to_numpy()
        mu = c.mean()
        sigma = c.std()
        outliers = (np.abs(c) > 5 * sigma).sum()
    
        base_info = {
            "ticker": ticker,
            "return_non_na": non_na,
            "return_coverage": non_na / expected_days,
            "first_valid_date": c.first_valid_index(),
            "last_valid_date": c.last_valid_index(),
            "missing_days": expected_days - non_na,
            "longest_missing_streak": longest_missing_streak(missing_mask),
            "return_mean": mu,
            "return_std": sigma,
            "outlier_count_5sigma": outliers,
            "annualized_vol": sigma * np.sqrt(252),
        }

        # If we don't have local fundamental data, we must fetch it (one-time setup)
        if not local_metadata_exists:
            try:
                info = yf.Ticker(ticker).info
                base_info.update({
                    "company_name": info.get("shortName"),
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                    "market_cap": info.get("marketCap"),
                    "country": info.get("country"),
                    "exchange": info.get("exchange"),
                })
            except Exception:
                base_info.update({
                    "company_name": None, "sector": None, "industry": None,
                    "market_cap": None, "country": None, "exchange": None,
                })
        else:
            # Preserve existing static data
            if ticker in existing_meta.index:
                for col in ["company_name", "sector", "industry", "market_cap", "country", "exchange"]:
                    if col in existing_meta.columns:
                        base_info[col] = existing_meta.loc[ticker, col]
        
        report.append(base_info)
    
    metadata = pd.DataFrame(report).sort_values("return_coverage", ascending=False)
    
    if save:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        metadata.to_csv(file_path, index=False)
        print(f"Metadata saved to {file_path}")
        
    return metadata

def load_metadata(file_path="data/raw/universe_metadata.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find {file_path}. Please generate it first.")
    metadata = pd.read_csv(file_path)
    return metadata

def process_universe_etfs(metadata_path="data/raw/universe_metadata.csv"):
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Cannot find {metadata_path}. Please run the metadata generation first.")
        
    print(f"Loading metadata from {metadata_path}...")
    metadata = pd.read_csv(metadata_path)

    sector_to_etf = {
        "Technology": "XLK",
        "Financial Services": "XLF",
        "Healthcare": "XLV",
        "Consumer Cyclical": "XLY",
        "Industrials": "XLI",
        "Communication Services": "XLC",
        "Consumer Defensive": "XLP",
        "Energy": "XLE",
        "Utilities": "XLU",
        "Real Estate": "XLRE",
        "Basic Materials": "XLB"
    }

    metadata['etf'] = metadata['sector'].map(sector_to_etf)
    metadata.to_csv(metadata_path, index=False)
    print(f"Successfully added 'etf' column and updated {metadata_path}")
    
    unique_etfs = metadata['etf'].dropna().unique().tolist()
    print(f"Identified {len(unique_etfs)} unique sector ETFs: {unique_etfs}")

    ticker_to_etfs = {}
    etf_to_tickers = {etf: [] for etf in sector_to_etf.values()}

    for _, row in metadata.iterrows():
        ticker = row['ticker']
        etf = row['etf']
        
        if pd.notna(etf):
            ticker_to_etfs[ticker] = [etf]
            etf_to_tickers[etf].append(ticker)
        else:
            ticker_to_etfs[ticker] = []

    etf_to_tickers = {k: v for k, v in etf_to_tickers.items() if len(v) > 0}

    return metadata, unique_etfs, ticker_to_etfs, etf_to_tickers

def download_prices_etf(etfs, start_date="2019-01-01", end_date="2024-01-01", file_path="data/raw/prices_etf.csv"):
    if not etfs:
        raise ValueError("ETF list is empty. Cannot download data.")
        
    print(f"\nDownloading data for {len(etfs)} ETFs...")
    
    prices_etf = yf.download(
        etfs,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        group_by="column"
    )['Close']
    
    if prices_etf.empty:
        raise ValueError("Downloaded ETF data is empty.")

    prices_etf = prices_etf.apply(pd.to_numeric, errors="coerce")
    prices_etf = prices_etf.replace([np.inf, -np.inf], np.nan)
    prices_etf.index.name = "date"

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    prices_etf.to_csv(file_path)
    print(f"ETF prices data saved to {file_path}")
    
    return prices_etf

def load_prices_etf(file_path="data/raw/prices_etf.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find {file_path}. Ensure it is in your project directory.")
        
    print(f"Loading local data from {file_path}...")
    
    prices_etf = pd.read_csv(file_path, index_col=0, parse_dates=True) 
    
    print(f"Original data shape: {prices_etf.shape}")
    print("Raw ETF data loading complete.")
    return prices_etf

def get_returns_etf(prices_etf, file_path="data/raw/returns_etf.csv", save=True):
    log_prices_etf = np.log(prices_etf)
    returns_etf = log_prices_etf.diff().iloc[1:]
    
    print(f"Returns calculated. Shape: {returns_etf.shape}")
    
    if save:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        returns_etf.to_csv(file_path)
        print(f"ETF returns data saved to {file_path}")
    
    return returns_etf

def get_returns_long_etf(returns_etf, file_path="data/raw/returns_long_etf.csv", save=True):
    returns_long_etf = (
        returns_etf
        .reset_index()
        .melt(id_vars="date", var_name="ticker", value_name="return")
    )
    
    returns_long_etf["return"] = pd.to_numeric(returns_long_etf["return"], errors="coerce")
    
    if save:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        returns_long_etf.to_csv(file_path, index=False)
        print(f"ETF returns data saved to {file_path} in long-format")
    
    return returns_long_etf