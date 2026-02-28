import matplotlib.pyplot as plt
import numpy as np


class PortfolioVisualizer:
    """
    A utility class for generating visual representations of quantitative
    portfolio data and analytics.
    """

    @staticmethod
    def plot_asset_allocation(
        tickers,
        weights,
        output_filename="allocation_chart.png",
        chart_title="Optimal Asset Allocation",
        min_weight_threshold=0.001,
    ):
        """
        Generates and saves a pie chart detailing the optimal asset allocation.

        Assets with an allocation weight below the minimum threshold are
        filtered out to maintain chart readability.

        Parameters
        ----------
        tickers : list of str
            The ticker symbols of the assets in the portfolio.
        weights : numpy.ndarray or list of float
            The optimized fractional weights corresponding to each asset.
            Must sum to ~1.0 and contain no negative values.
        output_filename : str, optional
            The filepath where the chart image will be saved.
            Defaults to 'allocation_chart.png'.
        chart_title : str, optional
            The title displayed at the top of the chart.
            Defaults to 'Optimal Asset Allocation'.
        min_weight_threshold : float, optional
            The fractional threshold below which assets are excluded from the chart.
            Defaults to 0.001 (0.1%).

        Raises
        ------
        ValueError
            If inputs are empty, dimensions mismatch, weights are negative,
            or weights do not sum to 1.0.
        """
        # 1. Input Validation: Check for empty arrays
        if not tickers or len(weights) == 0:
            raise ValueError("Input arrays `tickers` and `weights` cannot be empty.")

        # 2. Input Validation: Check for dimension mismatch
        if len(tickers) != len(weights):
            raise ValueError(
                f"Dimension mismatch: `tickers` has length {len(tickers)}, "
                f"but `weights` has length {len(weights)}."
            )

        # 3. Input Validation: Check for negative weights
        if np.any(np.asarray(weights) < 0):
            raise ValueError(
                "Weights cannot contain negative values (short selling is not supported)."
            )

        # 4. Input Validation: Ensure fractional weights sum to 1.0 (with tolerance for float math)
        if not np.isclose(np.sum(weights), 1.0, atol=1e-4):
            raise ValueError(
                f"Weights must sum to 1.0. Current sum: {np.sum(weights):.4f}"
            )

        # Filter out assets below the visibility threshold
        active_assets = [
            (t, w) for t, w in zip(tickers, weights) if w >= min_weight_threshold
        ]

        # Early exit if the optimizer assigned 0% to everything (edge case handling)
        if not active_assets:
            print(
                f"No assets exceed the minimum allocation threshold ({min_weight_threshold:.1%}); "
                "allocation chart will not be generated."
            )
            return

        labels = [ticker for ticker, _ in active_assets]
        sizes = [weight for _, weight in active_assets]

        fig, ax = plt.subplots(figsize=(8, 8))

        # Generate a color for each active asset using a larger colormap
        colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))

        ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
        )
        ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle

        # Apply the dynamic title
        plt.title(chart_title)

        # Save the figure with an explicit try/except block for I/O safety
        try:
            plt.savefig(output_filename, bbox_inches="tight")
        except (OSError, IOError) as e:
            print(f"Failed to save allocation chart to '{output_filename}': {e}")
            raise
        else:
            print(f"Saved allocation chart to '{output_filename}'")

        # Free up memory by closing the figure
        plt.close(fig)
