import yfinance as yf
import numpy as np
import pandas as pd


class MarketDataFetcher:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(self):
        """Fetches data and automatically shrinks the date range to fit the tickers."""
        print(f"Fetching data for: {', '.join(self.tickers)}...")

        raw_data = yf.download(
            self.tickers, start=self.start_date, end=self.end_date, progress=False
        )

        if raw_data.empty:
            raise ValueError(
                "No data found for these tickers. Check symbols/connection."
            )

        # Extract 'Adj Close' or 'Close'
        if isinstance(raw_data.columns, pd.MultiIndex):
            target = (
                "Adj Close" if "Adj Close" in raw_data.columns.levels[0] else "Close"
            )
            data = raw_data.xs(target, level=0, axis=1)
        else:
            data = (
                raw_data[["Adj Close"]]
                if "Adj Close" in raw_data.columns
                else raw_data[["Close"]]
            )

        # 1. Drop columns that are entirely empty (delisted/typos)
        data = data.dropna(axis=1, how="all")

        # 2. Update self.tickers to reflect only the ones that actually have data
        self.tickers = list(data.columns)

        # 3. ADAPTIVE ALIGNMENT: Find the start date of the "youngest" stock
        # Instead of dropping everything, we find the first row where all remaining tickers have a price
        initial_len = len(data)
        data = data.ffill().dropna(axis=0, how="any")

        if data.empty:
            raise ValueError(
                f"No overlapping history found for {self.tickers}. "
                "One or more tickers may be too new or have no trading overlap."
            )

        print(f"Successfully loaded {len(data)} days of data.")
        print(f"Analysis Period: {data.index[0].date()} to {data.index[-1].date()}")

        return data

    def get_live_risk_free_rate(self):
        """Fetches the current 3-month Treasury Bill yield as a risk-free rate proxy."""
        print("Fetching live risk-free rate (3-Month T-Bill)...")
        t_bill = yf.Ticker("^IRX")
        # Get the most recent closing price
        hist = t_bill.history(period="1d")
        if hist.empty:
            print("Warning: Could not fetch live rate. Falling back to 4%.")
            return 0.04
        return hist["Close"].iloc[-1] / 100  # Convert from percentage to decimal

    def calculate_annualized_metrics(self, data, trading_days=252):
        """
        Calculate annualized returns and covariance matrix from historical price data.
        non_positive_mask = data <= 0
        if non_positive_mask.to_numpy().any():
            print(
                "Warning: Zero or negative values detected in price data. Sanitizing data..."
            )
            data = data.where(~non_positive_mask, np.nan)
            log_return_t = np.log(data / data.shift(1))

        The mean and covariance of these log returns are then scaled by the number of
        trading days per year to obtain annualized statistics.

        Parameters
        ----------
        data : pandas.DataFrame or pandas.Series
            Historical price data indexed by date. For a DataFrame, each column is
            assumed to represent the price series of a single asset. All prices must
            be strictly positive; non-positive values are treated as invalid and
            removed from the computation.
        trading_days : int, optional
            Number of trading days used to annualize the statistics. Defaults to 252,
            which is a common approximation for the number of trading days in a year.

        Returns
        -------
        annual_returns : pandas.Series
            Annualized logarithmic mean returns for each asset, expressed on a
            per-year basis.
        annual_cov_matrix : pandas.DataFrame
            Annualized covariance matrix of logarithmic returns between all pairs
            of assets.
        """
        # 3. Validation: Prevent division by zero and invalid log operations
        # Stock prices cannot be <= 0. If bad data exists, convert it to NaN so it gets safely dropped.
        if (data <= 0).any().any():
            print(
                "Warning: Zero or negative values detected in price data. Sanitizing data..."
            )
            data = data.where(data > 0, np.nan)

        # Logarithmic returns for accurate continuous compounding
        # Any NaN values introduced by our sanitization step above are safely removed by .dropna()
        log_returns = np.log(data / data.shift(1)).dropna()
        annual_returns = log_returns.mean() * trading_days
        annual_cov_matrix = log_returns.cov() * trading_days

        return annual_returns, annual_cov_matrix
