from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

from loadData import MarketDataFetcher
from optimize import PortfolioOptimizer

app = Flask(__name__)
CORS(app)


@app.route("/api/optimize", methods=["POST"])
def optimize_portfolio():
    data = request.json
    tickers = data.get("tickers", [])
    capital = float(data.get("capital", 100000))
    strategy = data.get("strategy", "1")  # '1' for Sharpe, '2' for Volatility

    # 2. Pipeline setup
    end_date = datetime.today()
    start_date = end_date - relativedelta(years=5)
    fetcher = MarketDataFetcher(tickers, start_date, end_date)
    optimizer = PortfolioOptimizer(fetcher.get_live_risk_free_rate())
    try:
        price_data = fetcher.fetch_data()
        rets, cov = fetcher.calculate_annualized_metrics(price_data)

        # Hardcoding bounds for now, can make this dynamic later (i just wanna make sure this works first)
        bounds = tuple((0.0, 1.0) for _ in range(len(fetcher.tickers)))

        # 4. Strategy execution
        if str(strategy) == "2":
            weights = optimizer.minimize_volatility(rets, cov, fetcher.tickers, bounds)
        else:
            weights = optimizer.maximize_sharpe_ratio(rets, cov, bounds)

        p_ret, p_vol = optimizer.portfolio_performance(weights, rets, cov)

        # 5. Format the final report into JSON for React
        res = []
        for i, t in enumerate(fetcher.tickers):
            w = weights[i]
            if w > 0.001:  # Only send active assets to keep the frontend clean
                v = capital * w
                res.append({"ticker": t, "weight": float(w), "value": float(v)})

        p_simple_ret = np.exp(p_ret) - 1

        return jsonify(
            {
                "status": "success",
                "metrics": {
                    "expectedReturn": float(p_simple_ret),
                    "volatility": float(p_vol),
                    "projectedTotal": float(capital * np.exp(p_ret)),
                },
                "allocations": res,
            }
        )

    except Exception as e:
        print(f"\n[API ERROR]: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
