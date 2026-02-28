import matplotlib.pyplot as plt
import numpy as np
import os


class PortfolioVisualizer:
    @staticmethod
    def plot_asset_allocation(
        tickers,
        weights,
        output_filename="allocation_chart.png",
        chart_title="Optimal Asset Allocation",
    ):
        # Force clear the current figure to prevent ghosting from old charts
        plt.clf()
        plt.close("all")

        # Filter for visibility
        active_assets = [(t, w) for t, w in zip(tickers, weights) if w > 0.001]
        labels = [item[0] for item in active_assets]
        sizes = [item[1] for item in active_assets]

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))

        # Create the pie
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            pctdistance=0.85,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )

        # Draw a white circle at the center to create a donut chart
        centre_circle = plt.Circle((0, 0), 0.70, fc="white")
        ax.add_artist(centre_circle)

        plt.title(chart_title, pad=20, fontsize=16, fontweight="bold")
        ax.axis("equal")

        try:
            # Explicitly remove old file if it exists to force a refresh
            if os.path.exists(output_filename):
                os.remove(output_filename)
            plt.savefig(output_filename, bbox_inches="tight")
            print(f"Updated allocation chart saved as '{output_filename}'")
        except Exception as e:
            print(f"Failed to save chart: {e}")
        finally:
            plt.close(fig)
