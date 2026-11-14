import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_results(df, subplot_configs, save_path, dpi=300, show_plt = False, ncols=1):
    """
    Plots time-series simulation data from a DataFrame based on subplot configurations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the time series data (must include 't_s' column).
    subplot_configs : list
        List specifying the layout. Elements can be:
        - A string: plots a single line for that column.
        - A list or tuple of strings: overlays multiple lines on that subplot.
    save_path : str, optional
        File path to save the generated figure. Set to None to skip saving.
    dpi : int, optional
        Resolution for the saved figure.
    """
    neon_palette = [
        "#00FFFF",  # Electric Cyan
        "#FF5E00",  # Neon Orange
        "#FF00FF",  # Neon Magenta
        "#39FF14",  # Neon Green / Lime
        "#FFFF00",  # Bright Yellow
        "#BF00FF",  # Electric Purple
    ]

    sns.set_theme(
        style="darkgrid",
        palette=neon_palette,
        rc={
            "figure.facecolor": "#000000",   # Outer plot background
            "axes.facecolor": "#000000",     # Inner plot area background
            "axes.edgecolor": "#333333",     # Subtle border around the plot
            "text.color": "#FFFFFF",         # White title and text
            "axes.labelcolor": "#FFFFFF",    # White x and y labels
            "legend.facecolor": "#0D0D0D",   # Near-black legend box
            "legend.edgecolor": "#333333",
            # --- grid line settings ---
            "grid.color": "#FFFFFF",         # Faint gridlines so neon pops
            "grid.linewidth": 1.8,          # <-- Controls grid thickness (default is ~0.8)
            "grid.linestyle": ":",
            # --- tick settings --- 
            "xtick.color": "#FFFFFF",        # Light gray tick marks
            "ytick.color": "#FFFFFF",
            "xtick.labelsize": 11,           # Larger tick numbers (default is ~10)
            "ytick.labelsize": 11,
            "xtick.major.size": 6,           # Length of the tick mark (0 hides them)
            "ytick.major.size": 6,
            "xtick.major.width": 1.5,        # Thickness of the tick mark
            "ytick.major.width": 1.5,
            "xtick.direction": "inout",        # Point ticks outward ("in", "out", or "inout")
            "ytick.direction": "inout",
        }
    )

    # plt.style.use('dark_background')

    num_plots = len(subplot_configs)
    nrows = math.ceil(num_plots / ncols)

    # Create subplots
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(12, nrows * 3.5),
        sharex=True,
    )
    
    # Ensure axes is always iterable (even if nrows == 1)
    if num_plots == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    # Loop through configurations and plot
    for i, config in enumerate(subplot_configs):
        ax = axes[i]

        # Check if this subplot needs multiple lines overlaid
        if isinstance(config, (list, tuple)):
            for col in config:
                ax.plot(
                    df["t_s"],
                    df[col],
                    # marker="o",
                    linewidth=2.,
                    label=col,  # Label for legend
                )
            ax.legend()  # Show legend to distinguish lines
            title_str = ", ".join(config)
            ax.set_title(title_str, fontsize=11, fontweight="bold")
        else:
            # Single line plot
            ax.plot(
                df["t_s"],
                df[config],
                # marker="s",
                color="tab:orange",
                linewidth=2.5,
                label=config,
            )
            ax.set_title(config, fontsize=11, fontweight="bold")

        ax.set_ylabel("Value", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.6)

    # Hide unused empty subplot slots
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    # Formatting and layout
    fig.autofmt_xdate()
    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",  # Prevents labels/titles from getting cut off
        )

    if show_plt:
        plt.show()

    else:
        plt.close(fig)