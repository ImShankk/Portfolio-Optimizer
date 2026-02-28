import sys
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np

from loadData import MarketDataFetcher
from optimize import PortfolioOptimizer
from visualize import PortfolioVisualizer


def main():
    print("=" * 60)
    print("   SECURE PORTFOLIO OPTIMIZER & INVESTMENT PLANNER")
    print("=" * 60)

    # 1. User Configuration & Input Scrubbing
    try:
        ticker_input = input("\nEnter tickers (e.g., AAPL, MSFT, TSLA, XOM): ")
        # Remove duplicates while preserving order
        input_list = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        unique_tickers = list(dict.fromkeys(input_list))

        if not unique_tickers:
            raise ValueError("No tickers provided.")

        capital_input = input("Enter total investment amount (e.g., $21,000 CAD): $")
        # Clean numeric input: keep only digits and decimal points
        clean_capital = "".join(c for c in capital_input if c.isdigit() or c == ".")
        total_capital = float(clean_capital)

        # Detect currency label for display purposes
        currency_label = (
            "".join(c for c in capital_input if c.isalpha()).upper() or "USD"
        )

        if total_capital <= 0:
            raise ValueError("Investment capital must be greater than zero.")

    except ValueError as e:
        print(f"[ERROR] Invalid input: {e}")
        sys.exit(1)

    # Date Range: Last 5 Years
    end_date = datetime.today()
    start_date = end_date - relativedelta(years=5)

    # 2. Initialize Objects
    fetcher = MarketDataFetcher(unique_tickers, start_date, end_date)
    rf_rate = fetcher.get_live_risk_free_rate()
    optimizer = PortfolioOptimizer(risk_free_rate=rf_rate)

    # 3. Execute Pipeline
    try:
        # DATA FETCHING & VALIDATION
        data = fetcher.fetch_data()

        # After fetch_data(), the fetcher.tickers list may have shrunk
        # because it drops invalid/delisted tickers. We use the updated list.
        valid_tickers = fetcher.tickers

        if not valid_tickers:
            raise ValueError(
                "The ticker list is empty after cleaning. Cannot optimize."
            )

        print("Calculating annualized metrics...")
        annual_returns, annual_cov_matrix = fetcher.calculate_annualized_metrics(data)

        print(f"Optimizing for {len(valid_tickers)} assets...")
        # NOTE: If you want to change max allocation for any one stock,
        # adjust the bounds in optimize.py (e.g., 0.40 for 40% max).
        optimal_weights = optimizer.maximize_sharpe_ratio(
            annual_returns, annual_cov_matrix
        )

        opt_return, opt_volatility = optimizer.portfolio_performance(
            optimal_weights, annual_returns, annual_cov_matrix
        )

        opt_sharpe = optimizer.calculate_sharpe_ratio(opt_return, opt_volatility)
        last_prices = data.iloc[-1]

    except Exception as e:
        print(f"\n[CRITICAL ERROR]: {str(e)}")
        sys.exit(1)

    # 4. Final Output Table
    print("\n" + "=" * 60)
    print(f"INVESTMENT PLAN ({currency_label})")
    print("=" * 60)

    allocation_results = []
    for i, ticker in enumerate(valid_tickers):
        weight = optimal_weights[i]
        # Hide assets with less than 0.1% allocation
        if weight > 0.001:
            position_value = total_capital * weight
            price = last_prices[ticker]
            shares = position_value / price

            allocation_results.append(
                {
                    "Ticker": ticker,
                    "Weight": f"{weight:.2%}",
                    "Value": f"{position_value:,.2f}",
                    "Price": f"{price:,.2f}",
                    "Shares": round(shares, 4),
                }
            )

    df_plan = pd.DataFrame(allocation_results)
    if df_plan.empty:
        print("Optimization resulted in no valid allocations.")
    else:
        print(df_plan.to_string(index=False))

    print("-" * 60)
    print(f"Expected Annual Return: {opt_return:.2%}")
    print(f"Portfolio Volatility:   {opt_volatility:.2%}")
    print(f"Sharpe Ratio:           {opt_sharpe:.2f}")
    print(f"Risk-Free Rate Bench:   {rf_rate:.2%}")
    print("=" * 60)

    # 5. Visualization
    try:
        PortfolioVisualizer.plot_asset_allocation(
            valid_tickers, optimal_weights, chart_title="Optimized Asset Allocation"
        )
    except Exception as e:
        print(f"\n[WARNING] Could not generate chart: {e}")


if __name__ == "__main__":
    main()
