import sys
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import yfinance as yf
import numpy as np

from loadData import MarketDataFetcher
from optimize import PortfolioOptimizer
from visualize import PortfolioVisualizer


def get_valid_tickers(initial_tickers):
    valid_tickers = []
    current_list = initial_tickers
    i = 0
    while i < len(current_list):
        ticker = current_list[i]
        print(f"Checking {ticker}...")
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            if hist.empty:
                raise ValueError
            valid_tickers.append(ticker)
        except:
            print(f"\n[!] ALERT: '{ticker}' is invalid.")
            choice = input(
                f"Would you like to (R)eplace it, (S)kip it, or (Q)uit? "
            ).lower()
            if choice == "r":
                new_ticker = input("Enter replacement: ").strip().upper()
                current_list[i] = new_ticker
                continue
            elif choice == "q":
                sys.exit(0)
        i += 1
    return valid_tickers


def main():
    print("=" * 60)
    print("   PROFESSIONAL PORTFOLIO OPTIMIZER")
    print("=" * 60)

    # 1. Inputs
    raw_input = input("\nEnter tickers (e.g., AAPL, NVDA, XOM): ")
    tickers = list(
        dict.fromkeys([t.strip().upper() for t in raw_input.split(",") if t.strip()])
    )
    valid_tickers = get_valid_tickers(tickers)

    amount_input = input("\nEnter investment amount (e.g., $100,000): ")
    capital = float("".join(c for c in amount_input if c.isdigit() or c == "."))

    # 2. Set Diversification Rules
    print("\n--- Diversification Settings ---")
    min_p = float(input("Minimum % per stock (e.g., 5): ")) / 100
    max_p = float(input("Maximum % per stock (e.g., 35): ")) / 100

    # Quick check: min_p * num_assets cannot exceed 100%
    if min_p * len(valid_tickers) > 1.0:
        print(
            f"[ERROR] Min allocation of {min_p:.0%} for {len(valid_tickers)} stocks is impossible (>100%)."
        )
        return

    # 3. Process
    end_date = datetime.today()
    start_date = end_date - relativedelta(years=5)
    fetcher = MarketDataFetcher(valid_tickers, start_date, end_date)
    optimizer = PortfolioOptimizer(risk_free_rate=fetcher.get_live_risk_free_rate())

    try:
        data = fetcher.fetch_data()
        returns, cov = fetcher.calculate_annualized_metrics(data)

        # 4. Optimized with Diversification Bounds
        # We pass the custom bounds here
        bounds = tuple((min_p, max_p) for _ in range(len(valid_tickers)))

        # We need to tweak optimize.py slightly to accept these bounds as an argument
        # For now, let's assume we update optimize.py maximize_sharpe_ratio(self, annual_returns, annual_cov_matrix, bounds=None)
        optimal_weights = optimizer.maximize_sharpe_ratio(returns, cov, bounds=bounds)

        p_ret, p_vol = optimizer.portfolio_performance(optimal_weights, returns, cov)
        last_prices = data.iloc[-1]

        # 5. Output Results
        print("\n" + "=" * 60)
        print(f"DIVERSIFIED INVESTMENT PLAN (${capital:,.2f})")
        print("=" * 60)

        plan = []
        for i, ticker in enumerate(valid_tickers):
            weight = optimal_weights[i]
            val = capital * weight
            plan.append(
                [
                    ticker,
                    f"{weight:.2%}",
                    f"${val:,.2f}",
                    round(val / last_prices[ticker], 2),
                ]
            )

        df = pd.DataFrame(plan, columns=["Ticker", "Weight", "Value", "Shares"])
        print(df.to_string(index=False))
        print("-" * 60)
        print(f"Expected Return: {p_ret:.2%} | Volatility: {p_vol:.2%}")
        print(f"Sharpe Ratio: {(p_ret - optimizer.risk_free_rate)/p_vol:.2f}")
        print("=" * 60)

        PortfolioVisualizer.plot_asset_allocation(valid_tickers, optimal_weights)

    except Exception as e:
        print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()
