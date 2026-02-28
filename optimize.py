import numpy as np
from scipy.optimize import minimize


class PortfolioOptimizer:
    """
    A quantitative optimization engine that calculates the optimal asset allocation
    to maximize the risk-adjusted return (Sharpe ratio) of a portfolio.
    """

    # Class-level constant to heavily penalize mathematically invalid portfolios
    # (e.g., zero volatility) during the SciPy minimization process.
    INVALID_PORTFOLIO_PENALTY = float("inf")

    def __init__(self, risk_free_rate=0.04):
        """
        Parameters
        ----------
        risk_free_rate : float, optional
            The theoretical rate of return of an investment with zero risk. Defaults to 0.04 (4%).
        """
        self.risk_free_rate = risk_free_rate

    def portfolio_performance(self, weights, returns, cov_matrix):
        """
        Calculates the expected annualized return and volatility of a portfolio.
        """
        # Ensure inputs have compatible dimensions to avoid obscure NumPy errors.
        weights_arr = np.asarray(weights).reshape(-1)
        n_assets = weights_arr.shape[0]

        # Validate returns length
        try:
            n_returns = len(returns)
        except TypeError:
            raise ValueError(
                "`returns` must be a 1-D array-like object with length equal to the number of assets."
            )

        if n_returns != n_assets:
            raise ValueError(
                f"Dimension mismatch: weights has length {n_assets} but returns has length {n_returns}."
            )

        # Validate covariance matrix shape
        try:
            cov_shape = cov_matrix.shape
        except AttributeError:
            raise ValueError(
                "`cov_matrix` must be an array-like object with a 2-D shape."
            )

        if len(cov_shape) != 2:
            raise ValueError(f"`cov_matrix` must be 2-D, but has shape {cov_shape}.")

        rows, cols = cov_shape
        if rows != cols:
            raise ValueError(f"`cov_matrix` must be square, but has shape {cov_shape}.")

        if rows != n_assets:
            raise ValueError(
                f"Dimension mismatch: weights has length {n_assets} but cov_matrix has shape {cov_shape}."
            )

        returns_p = np.dot(weights_arr, returns)
        variance_p = float(np.dot(weights_arr.T, np.dot(cov_matrix, weights_arr)))

        # Guard against negative variance due to numerical issues or invalid covariance matrices.
        if variance_p < 0.0:
            # Allow tiny negative values caused by floating-point rounding;
            # treat them as zero variance. Larger negatives indicate invalid input.
            if variance_p >= -1e-10:
                variance_p = 0.0
            else:
                raise ValueError(
                    f"Computed negative portfolio variance ({variance_p}); "
                    "check the covariance matrix and asset weights."
                )

        volatility_p = np.sqrt(variance_p)
        return returns_p, volatility_p

    def calculate_sharpe_ratio(self, expected_return, volatility):
        """
        Calculates the expected Sharpe ratio for a given return and volatility profile.
        """
        eps = np.finfo(float).eps
        if abs(volatility) < eps:
            return float("-inf")

        return (expected_return - self.risk_free_rate) / volatility

    def _negative_sharpe_ratio(self, weights, returns, cov_matrix):
        """
        The objective function to minimize (minimizing the negative Sharpe ratio
        is equivalent to maximizing the positive Sharpe ratio).
        """
        p_ret, p_vol = self.portfolio_performance(weights, returns, cov_matrix)

        eps = np.finfo(float).eps
        if abs(p_vol) < eps:
            return self.INVALID_PORTFOLIO_PENALTY

        return -(p_ret - self.risk_free_rate) / p_vol

    def maximize_sharpe_ratio(self, annual_returns, annual_cov_matrix, bounds=None):
        """
        Maximizes the Sharpe ratio with support for custom diversification bounds.
        """
        try:
            num_assets = len(annual_returns)
        except TypeError:
            raise ValueError("`annual_returns` must be a 1-D array-like object.")

        args = (annual_returns, annual_cov_matrix)
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

        # Default to 0-100% if no specific bounds are provided
        if bounds is None:
            bounds = tuple((0, 1) for _ in range(num_assets))

        init_guess = np.full(num_assets, 1.0 / num_assets)

        result = minimize(
            self._negative_sharpe_ratio,
            init_guess,
            args=args,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if not result.success:
            raise RuntimeError(f"Sharpe ratio optimization failed: {result.message}")

        return result.x
