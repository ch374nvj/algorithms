import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_results(df, subplot_configs, save_path, dpi=300, show_plt = False):
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
    sns.set_theme(style="whitegrid")

    num_plots = len(subplot_configs)
    ncols = 1
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
                    marker="o",
                    linewidth=2,
                    label=col,  # Label for legend
                )
            ax.legend(loc="upper left")  # Show legend to distinguish lines
            title_str = ", ".join(config)
            ax.set_title(title_str, fontsize=11, fontweight="bold")
        else:
            # Single line plot
            ax.plot(
                df["t_s"],
                df[config],
                marker="s",
                color="tab:orange",
                linewidth=2,
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