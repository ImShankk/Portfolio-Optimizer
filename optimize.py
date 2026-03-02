import numpy as np
from scipy.optimize import minimize


class PortfolioOptimizer:
    def __init__(self, risk_free_rate=0.04):
        self.risk_free_rate = risk_free_rate
        self.etf_proxies = [
            "SPY",
            "QQQ",
            "VFV",
            "VTI",
            "VOO",
            "IVV",
            "ZSP",
            "XIU",
            "XUU",
            "TEC",
        ]

    def portfolio_performance(self, weights, returns, cov_matrix):
        weights = np.array(weights)
        returns_p = np.dot(weights, returns)
        variance_p = np.dot(weights.T, np.dot(cov_matrix, weights))
        volatility_p = np.sqrt(max(0, variance_p))
        return returns_p, volatility_p

    def minimize_volatility(
        self, annual_returns, annual_cov_matrix, tickers, bounds=None
    ):
        num_assets = len(annual_returns)
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        if bounds is None:
            bounds = tuple((0, 1) for _ in range(num_assets))

        def objective(weights):
            p_ret, vol = self.portfolio_performance(
                weights, annual_returns, annual_cov_matrix
            )

            # --- AUTO-ADAPTIVE DIVERSITY ---
            # We apply a penalty based on asset type.
            # Single stocks get a 'Concentration Tax' that grows as they get bigger.
            concentration_penalty = 0
            etf_weights = []

            for i, ticker in enumerate(tickers):
                is_etf = any(etf in ticker.upper() for etf in self.etf_proxies)

                if not is_etf:
                    # Single stocks: The more you buy, the higher the 'risk' cost
                    # 0.35 is the 'Auto-Tuned' scalar for professional balance
                    concentration_penalty += 0.35 * (weights[i] ** 2)
                else:
                    etf_weights.append(weights[i])
                    # ETFs: Very low penalty (0.01) to allow them to take the lead
                    concentration_penalty += 0.01 * (weights[i] ** 2)

            # Diversity Reward: If you have SPY and QQQ, this forces them to split
            # the load rather than one hogging 37% and the other 3%.
            if len(etf_weights) > 1:
                concentration_penalty += 0.15 * np.var(etf_weights)

            # Momentum Nudge (0.15): Allows NVDA/AMD to stay in the mix if they win big
            return vol + concentration_penalty - (0.15 * p_ret)

        init_guess = np.full(num_assets, 1.0 / num_assets)
        result = minimize(
            objective,
            init_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        return result.x
