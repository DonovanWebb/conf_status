"""
Graph style settings for matplotlib figures.
"""

import matplotlib.pyplot as plt
import numpy as np


def set_graph_style():
    line_width_pt = 246  # latex linewidth in points
    line_width_in = 1 / 72 * line_width_pt  # latex linewidth in points
    fig_width = 0.95 * line_width_in
    phi = (1 + 5**0.5) / 2  # golden ratio
    fig_height = fig_width / phi
    plt.style.use("seaborn-v0_8-muted")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
            "figure.figsize": (fig_width, fig_height),
            "lines.linewidth": 2,
            "lines.markersize": 3,
            "lines.linestyle": "--",
            "axes.grid": True,
            "grid.alpha": 1.0,
            "grid.linestyle": ":",
            "grid.linewidth": 1.0,
        }
    )


if __name__ == "__main__":
    # Example usage
    set_graph_style()
    data_x = np.random.randn(50)
    data_y = np.random.randn(50)
    plt.plot(data_x)
    plt.plot(data_y)
    plt.xlabel("X-axis / MHz")
    plt.ylabel("Y-axis / AU")
    plt.show()
