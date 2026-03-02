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
    valid = []
    curr = initial_tickers
    i = 0
    while i < len(curr):
        ticker = curr[i]
        print(f"Checking {ticker}...")
        try:
            if yf.Ticker(ticker).history(period="1d").empty:
                raise ValueError
            valid.append(ticker)
        except:
            print(f"\n[!] '{ticker}' is invalid.")
            choice = input("(R)eplace, (S)kip, or (Q)uit? ").lower()
            if choice == "r":
                curr[i] = input("New Ticker: ").strip().upper()
                continue
            elif choice == "q":
                sys.exit(0)
        i += 1
    return valid


def main():
    print("=" * 60 + "\n   INTELLIGENT PORTFOLIO OPTIMIZER v4.5\n" + "=" * 60)

    # 1. Tickers
    t_in = input("\nEnter tickers (e.g. AAPL, NVDA, SPY): ")
    tickers = list(
        dict.fromkeys([t.strip().upper() for t in t_in.split(",") if t.strip()])
    )
    valid_tickers = get_valid_tickers(tickers)

    # 2. Capital (Updated Robust Logic)
    while True:
        cap_in = input("\nEnter total investment amount (e.g. 100,000.00): ")
        # Remove commas first
        temp_cap = cap_in.replace(",", "")
        # Now clean any other symbols
        clean = "".join(c for c in temp_cap if c.isdigit() or c == ".")
        try:
            total_capital = float(clean)
            if total_capital > 0:
                break
        except:
            print("[!] Invalid amount. Please enter a number like 100000.")

    # 3. Diversification
    print("\n--- Diversification Settings ---")
    min_p, max_p = 0.0, 1.0
    if input("Set a Minimum % per stock? (y/n): ").lower() == "y":
        min_p = float(input("Min %: ")) / 100
    if input("Set a Maximum % per stock? (y/n): ").lower() == "y":
        max_p = float(input("Max %: ")) / 100

    # 4. Strategy
    print(
        "\n--- Strategy ---\n1. Max Sharpe (Aggressive)\n2. Min Volatility (Conservative + ETF Preference)"
    )
    strat = input("Choice (1 or 2): ")

    # 5. Pipeline
    end_date = datetime.today()
    start_date = end_date - relativedelta(years=5)
    fetcher = MarketDataFetcher(valid_tickers, start_date, end_date)
    optimizer = PortfolioOptimizer(fetcher.get_live_risk_free_rate())

    try:
        data = fetcher.fetch_data()
        rets, cov = fetcher.calculate_annualized_metrics(data)
        bounds = tuple((min_p, max_p) for _ in range(len(valid_tickers)))

        if strat == "2":
            print("Optimizing for Minimum Risk (with Strong ETF Preference)...")
            weights = optimizer.minimize_volatility(rets, cov, valid_tickers, bounds)
        else:
            weights = optimizer.maximize_sharpe_ratio(rets, cov, bounds)

        p_ret, p_vol = optimizer.portfolio_performance(weights, rets, cov)

        # 6. Benchmark Logic
        print("Fetching Benchmark Data (SPY)...")
        bench_df = yf.download("SPY", period="1y", progress=False)
        spy_prices = (
            bench_df["Adj Close"]
            if "Adj Close" in bench_df.columns
            else bench_df["Close"]
        )

        # Using .iloc[0].item() ensures we get a single number, not a "Series"
        spy_start = (
            float(spy_prices.iloc[0].item())
            if hasattr(spy_prices.iloc[0], "item")
            else float(spy_prices.iloc[0])
        )
        spy_end = (
            float(spy_prices.iloc[-1].item())
            if hasattr(spy_prices.iloc[-1], "item")
            else float(spy_prices.iloc[-1])
        )
        spy_1y_ret = (spy_end / spy_start) - 1

        # 7. Final Report
        print(
            "\n"
            + "=" * 95
            + f"\nINTELLIGENT INVESTMENT PLAN (${total_capital:,.2f})\n"
            + "=" * 95
        )
        res = []
        for i, t in enumerate(valid_tickers):
            w = weights[i]
            v = total_capital * w
            last_p = float(data.iloc[-1][t])
            proj_v = v * np.exp(rets[t])
            res.append(
                [t, f"{w:.2%}", f"${v:,.2f}", round(v / last_p, 2), f"${proj_v:,.2f}"]
            )

        df = pd.DataFrame(
            res, columns=["Ticker", "Weight", "Value", "Shares", "Proj. 1Y Value"]
        )
        print(df.to_string(index=False))
        print("-" * 95)

        p_simple_ret = np.exp(p_ret) - 1
        print(f"Portfolio Expected 1Y Return: {float(p_simple_ret):.2%}")
        print(f"S&P 500 (SPY) 1Y Performance:  {float(spy_1y_ret):.2%}")
        print(
            f"Status vs Benchmark:           {'OUTPERFORMING' if p_simple_ret > spy_1y_ret else 'UNDERPERFORMING'}"
        )
        print(f"Portfolio Volatility:          {float(p_vol):.2%}")
        print("-" * 95)

        proj_total = total_capital * np.exp(p_ret)
        print(
            f"ESTIMATED 1-YEAR TOTAL: ${float(proj_total):,.2f} (Gain: ${float(proj_total-total_capital):,.2f})"
        )
        print("=" * 95)

        PortfolioVisualizer.plot_asset_allocation(valid_tickers, weights)

    except Exception as e:
        print(f"\n[ERROR]: {str(e)}")


if __name__ == "__main__":
    main()
