from dataclasses import dataclass
from typing import Optional, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def n_plot(xlab,
           ylab,
           xs=14,
           ys=14):
    """
    Configures the appearance of a matplotlib plot.

    This function sets up minor ticks, major ticks, and axis labels for a plot.
    It adjusts the tick parameters and applies a tight layout to ensure proper
    spacing.

    Parameters
    ----------
    xlab : str
        Label for the x-axis.
    ylab : str
        Label for the y-axis.
    xs : int, optional
        Font size for the x-axis label (default is 14).
    ys : int, optional
        Font size for the y-axis label (default is 14).

    Returns
    -------
    None
    """
    plt.minorticks_on()
    plt.tick_params(axis='both', which='major', labelsize=ys - 2, direction='in', length=6, width=2)
    plt.tick_params(axis='both', which='minor', labelsize=ys - 2, direction='in', length=4, width=2)
    plt.tick_params(axis='both', which='both', top=True, right=True)
    plt.xlabel(xlab, fontsize=xs)
    plt.ylabel(ylab, fontsize=ys)
    plt.tight_layout()
    return None


def ax_plot(fig,
            ax,
            xlab,
            ylab,
            xs=14,
            ys=14):
    """
    Configures the appearance of a matplotlib plot using a given figure and axes.

    This function sets up minor ticks, major ticks, and axis labels for the provided
    matplotlib axes. It adjusts the tick parameters and applies a tight layout to
    ensure proper spacing.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    ax : matplotlib.axes.Axes
        The matplotlib axes object to configure.
    xlab : str
        Label for the x-axis.
    ylab : str
        Label for the y-axis.
    xs : int, optional
        Font size for the x-axis label (default is 14).
    ys : int, optional
        Font size for the y-axis label (default is 14).

    Returns
    -------
    None
    """
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', labelsize=ys - 2, direction='in', length=6, width=2)
    ax.tick_params(axis='both', which='minor', labelsize=ys - 2, direction='in', length=4, width=2)
    ax.tick_params(axis='both', which='both', top=True, right=True)
    ax.set_xlabel(xlab, fontsize=xs)
    ax.set_ylabel(ylab, fontsize=ys)
    fig.tight_layout()
    return None


def plot_fes_series_1d(fes_arrays: list[np.ndarray],
                       fig=None,
                       ax=None,
                       slices: list[float] = None,
                       labels: list[str] = None,
                       max_slices: int = 5,
                       save: bool = True,
                       show: bool = True,
                       filename: str = "fes_1d",
                       x_lab: str = r"CV1",
                       y_lab: str = r"$F$ (eV)",
                       fig_size: tuple = (8, 3)):
    """
    Plots a series of 1D free energy surfaces (FES) over time.

    This function generates a plot of multiple 1D FES arrays, optionally labeling
    each curve with a corresponding time slice. It allows customization of the
    figure, axis, labels, and other plot parameters. The plot can be saved to
    files and/or displayed.

    Parameters
    ----------
    fes_arrays : list[np.ndarray]
        A list of 1D FES arrays, where each array contains x and y data for plotting.
    fig : matplotlib.figure.Figure, optional
        The matplotlib figure object to use for the plot. If None, a new figure is created.
    ax : matplotlib.axes.Axes, optional
        The matplotlib axes object to use for the plot. If None, new axes are created.
    slices : list[float], optional
        A list of time slices corresponding to each FES array. If None, indices are used.
    labels : list[str], optional
        A list of labels for each FES curve. If None, labels are generated based on slices.
    max_slices : int, optional
        The maximum number of slices to plot (default is 5).
    save : bool, optional
        Whether to save the plot as a file (default is True).
    show : bool, optional
        Whether to display the plot (default is True).
    filename : str, optional
        The base filename for saving the plot (default is "fes_1d").
    x_lab : str, optional
        The label for the x-axis (default is "CV1").
    y_lab : str, optional
        The label for the y-axis (default is "$F$ (eV)").
    fig_size : tuple, optional
        The size of the figure in inches (default is (8, 3)).

    Returns
    -------
    tuple
        A tuple containing the matplotlib figure and axes objects.
    """
    if slices is None:
        slices = np.arange(len(fes_arrays))

    # If there are more than max_times, select only the last max_times
    if len(slices) > max_slices:
        fes_arrays = fes_arrays[-max_slices:]
        slices = slices[-max_slices:]
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

    for i, xy in enumerate(fes_arrays):
        if labels is not None:
            label = labels[i]
        else:
            label = fr"$t={slices[i]}$ ps"
        ax.plot(xy[0], xy[1], label=label)

    ax.legend()
    ax_plot(fig, ax, x_lab, y_lab)
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    return fig, ax


def plot_fes_series_1d_compare(fes_arrays_a: list[np.ndarray],
                               fes_arrays_b: list[np.ndarray],
                               fig=None,
                               ax=None,
                               labels: list[str] = None,
                               save: bool = True,
                               show: bool = True,
                               filename: str = "fes_1d_compare",
                               x_lab: str = r"CV1",
                               y_lab: str = r"$F$ (eV)",
                               fig_size: tuple = (8, 3)):
    """
    Plots and compares two series of 1D free energy surfaces (FES).

    This function generates a plot comparing two 1D FES arrays. It allows customization
    of the figure, axis, labels, and other plot parameters. The plot can be saved to
    files and/or displayed.

    Parameters
    ----------
    fes_arrays_a : list[np.ndarray]
        The first 1D FES array, containing x and y data for plotting.
    fes_arrays_b : list[np.ndarray]
        The second 1D FES array, containing x and y data for plotting.
    fig : matplotlib.figure.Figure, optional
        The matplotlib figure object to use for the plot. If None, a new figure is created.
    ax : matplotlib.axes.Axes, optional
        The matplotlib axes object to use for the plot. If None, new axes are created.
    labels : list[str], optional
        A list of labels for the two FES curves. If None, default labels ["MD", "PIMD"] are used.
    save : bool, optional
        Whether to save the plot as a file (default is True).
    show : bool, optional
        Whether to display the plot (default is True).
    filename : str, optional
        The base filename for saving the plot (default is "fes_1d_compare").
    x_lab : str, optional
        The label for the x-axis (default is "CV1").
    y_lab : str, optional
        The label for the y-axis (default is "$F$ (eV)").
    fig_size : tuple, optional
        The size of the figure in inches (default is (8, 3)).

    Returns
    -------
    tuple
        A tuple containing the matplotlib figure and axes objects.
    """
    if labels is None:
        labels = ["MD", "PIMD"]
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

    ax.plot(*fes_arrays_a, '-', label=labels[0], lw=2)
    ax.plot(*fes_arrays_b, '--', label=labels[1], lw=2)

    ax.legend(loc="best")
    ax_plot(fig, ax, x_lab, y_lab)
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    return fig, ax


def plot_fes_contourf_series(fes_arrays: list[np.ndarray],
                             fig=None,
                             ax=None,
                             times: list[float] = None,
                             max_times=5,
                             save=True,
                             show=True,
                             filename="fes_contourf",
                             x_lab="CV1",
                             y_lab="CV2",
                             fig_size=(8, 3)):
    """
    Plots a series of 2D free energy surfaces (FES) as contour plots over time.

    This function generates a series of contour plots for multiple 2D FES arrays.
    It allows customization of the figure, axis, labels, and other plot parameters.
    The plot can be saved to files and/or displayed.

    Parameters
    ----------
    fes_arrays : list[np.ndarray]
        A list of 2D FES arrays, where each array contains x, y, and z data for plotting.
    fig : matplotlib.figure.Figure, optional
        The matplotlib figure object to use for the plot. If None, a new figure is created.
    ax : matplotlib.axes.Axes, optional
        The matplotlib axes object to use for the plot. If None, new axes are created.
    times : list[float], optional
        A list of time points corresponding to each FES array. If None, indices are used.
    max_times : int, optional
        The maximum number of time points to plot (default is 5).
    save : bool, optional
        Whether to save the plot as a file (default is True).
    show : bool, optional
        Whether to display the plot (default is True).
    filename : str, optional
        The base filename for saving the plot (default is "fes_contourf").
    x_lab : str, optional
        The label for the x-axis (default is "CV1").
    y_lab : str, optional
        The label for the y-axis (default is "CV2").
    fig_size : tuple, optional
        The size of the figure in inches (default is (8, 3)).

    Returns
    -------
    tuple
        A tuple containing the matplotlib figure and axes objects.
    """
    if times is None:
        times = np.arange(len(fes_arrays))

    # If there are more than max_times, select only the last max_times
    if len(times) > max_times:
        fes_arrays = fes_arrays[-max_times:]
        times = times[-max_times:]
    if fig is None or ax is None:
        fig, ax = plt.subplots(
            nrows=1,
            ncols=len(fes_arrays),
            figsize=fig_size,
            sharex=True,
            sharey=True,
            constrained_layout=True
        )

    contours = []
    for i, xyz in enumerate(fes_arrays):
        cf = ax[i].contourf(*xyz)
        contours.append(cf)
        ax[i].set_xlabel(x_lab)
        ax[i].set_title(fr"$t={times[i]}$ ps")

    ax[0].set_ylabel(y_lab)
    fig.colorbar(contours[-1], ax=ax, orientation="vertical", label=r"$F$ (eV)")
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    return fig, ax


def plot_fes_contourf_compare(fes_a,
                              fes_b,
                              fig=None,
                              ax=None,
                              labels=None,
                              save=True,
                              show=True,
                              filename="fes_contourf_compare",
                              x_lab="CV1",
                              y_lab="CV2",
                              fig_size=(8, 3)):
    """
    Plots and compares two 2D free energy surfaces (FES) as contour plots.

    This function generates side-by-side contour plots for two 2D FES arrays.
    It allows customization of the figure, axis, labels, and other plot parameters.
    The plot can be saved to files and/or displayed.

    Parameters
    ----------
    fes_a : np.ndarray
        The first 2D FES array, containing x, y, and z data for plotting.
    fes_b : np.ndarray
        The second 2D FES array, containing x, y, and z data for plotting.
    fig : matplotlib.figure.Figure, optional
        The matplotlib figure object to use for the plot. If None, a new figure is created.
    ax : np.ndarray of matplotlib.axes.Axes, optional
        The matplotlib axes objects to use for the plot. If None, new axes are created.
    labels : list[str], optional
        A list of labels for the two FES plots. If None, default labels ["MD", "PIMD"] are used.
    save : bool, optional
        Whether to save the plot as a file (default is True).
    show : bool, optional
        Whether to display the plot (default is True).
    filename : str, optional
        The base filename for saving the plot (default is "fes_contourf_compare").
    x_lab : str, optional
        The label for the x-axis (default is "CV1").
    y_lab : str, optional
        The label for the y-axis (default is "CV2").
    fig_size : tuple, optional
        The size of the figure in inches (default is (8, 3)).

    Returns
    -------
    tuple
        A tuple containing the matplotlib figure and axes objects.
    """
    fes_arrays = [fes_a, fes_b]
    if labels is None:
        labels = ["MD", "PIMD"]
    if fig is None or ax is None:
        fig, ax = plt.subplots(
            nrows=1,
            ncols=2,
            figsize=fig_size,
            sharex=True,
            sharey=True,
            constrained_layout=True
        )

    contours = []
    for i, xyz in enumerate(fes_arrays):
        cf = ax[i].contourf(*xyz)
        contours.append(cf)
        ax[i].set_xlabel(x_lab)
        ax[i].set_title(fr"{labels[i]}")

    ax[0].set_ylabel(y_lab)
    fig.colorbar(contours[-1], ax=ax, orientation="vertical", label=r"$F$ (eV)")
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    return fig, ax


def plot_fes_contourf(fes,
                      fig=None,
                      ax=None,
                      save=True,
                      show=True,
                      filename="fes_contourf",
                      x_lab="CV1",
                      y_lab="CV2",
                      fig_size=(8, 3),
                      ):
    """
    Plots a 2D free energy surface (FES) as a contour plot.

    This function generates a contour plot for a given 2D FES array. It allows
    customization of the figure, axis, labels, and other plot parameters. The
    plot can be saved to files and/or displayed.

    Parameters
    ----------
    fes : np.ndarray
        A 2D FES array containing x, y, and z data for plotting.
    fig : matplotlib.figure.Figure, optional
        The matplotlib figure object to use for the plot. If None, a new figure is created.
    ax : matplotlib.axes.Axes, optional
        The matplotlib axes object to use for the plot. If None, new axes are created.
    save : bool, optional
        Whether to save the plot as a file (default is True).
    show : bool, optional
        Whether to display the plot (default is True).
    filename : str, optional
        The base filename for saving the plot (default is "fes_contourf").
    x_lab : str, optional
        The label for the x-axis (default is "CV1").
    y_lab : str, optional
        The label for the y-axis (default is "CV2").
    fig_size : tuple, optional
        The size of the figure in inches (default is (8, 3)).

    Returns
    -------
    tuple
        A tuple containing the matplotlib figure and axes objects.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

    cf = ax.contourf(*fes)
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)
    fig.colorbar(cf, ax=ax, orientation="vertical", label=r"$F$ (eV)")
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    return fig, ax


def plot_fes_contour_compare(fes_a,
                             fes_b,
                             fig=None,
                             ax=None,
                             labels=None,
                             save=True,
                             show=True,
                             filename="fes_contour_compare",
                             x_lab="CV1",
                             y_lab="CV2",
                             fig_size=(8, 3),
                             ):
    """
    Plots and compares two 2D free energy surfaces (FES) as contour plots.

    This function generates a contour plot for two 2D FES arrays, overlaying them
    with different colors. It allows customization of the figure, axis, labels,
    and other plot parameters. The plot can be saved to files and/or displayed.

    Parameters
    ----------
    fes_a : np.ndarray
        The first 2D FES array, containing x, y, and z data for plotting.
    fes_b : np.ndarray
        The second 2D FES array, containing x, y, and z data for plotting.
    fig : matplotlib.figure.Figure, optional
        The matplotlib figure object to use for the plot. If None, a new figure is created.
    ax : matplotlib.axes.Axes, optional
        The matplotlib axes object to use for the plot. If None, new axes are created.
    labels : list[str], optional
        A list of labels for the two FES plots. If None, default labels ["MD", "PIMD"] are used.
    save : bool, optional
        Whether to save the plot as a file (default is True).
    show : bool, optional
        Whether to display the plot (default is True).
    filename : str, optional
        The base filename for saving the plot (default is "fes_contour_compare").
    x_lab : str, optional
        The label for the x-axis (default is "CV1").
    y_lab : str, optional
        The label for the y-axis (default is "CV2").
    fig_size : tuple, optional
        The size of the figure in inches (default is (8, 3)).

    Returns
    -------
    tuple
        A tuple containing the matplotlib figure and axes objects.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
    if labels is None:
        labels = ["MD", "PIMD"]

    levels = np.linspace(0, 0.5, 6)
    ax.contour(*fes_a, colors="b", levels=levels)
    ax.contour(*fes_b, colors="r", levels=levels)

    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)

    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="b", label=labels[0]),
            plt.Line2D([0], [0], color="r", label=labels[1]),
        ]
    )
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    return fig, ax


def plot_fes_sep(fes_a,
                 fes_b,
                 fig=None,
                 ax=None,
                 save=True,
                 show=True,
                 filename="energy_sep",
                 fig_size=(8, 3)):
    """
    Plots and compares two 2D free energy surfaces (FES) at specific slices.

    This function generates a plot comparing two FES datasets at specific slices
    along the third dimension. It allows customization of the figure, axis, and
    other plot parameters. The plot can be saved to files and/or displayed.

    Parameters
    ----------
    fes_a : np.ndarray
        The first 3D FES array, where the third dimension represents slices.
    fes_b : np.ndarray
        The second 3D FES array, where the third dimension represents slices.
    fig : matplotlib.figure.Figure, optional
        The matplotlib figure object to use for the plot. If None, a new figure is created.
    ax : matplotlib.axes.Axes, optional
        The matplotlib axes object to use for the plot. If None, new axes are created.
    save : bool, optional
        Whether to save the plot as a file (default is True).
    show : bool, optional
        Whether to display the plot (default is True).
    filename : str, optional
        The base filename for saving the plot (default is "energy_sep").
    fig_size : tuple, optional
        The size of the figure in inches (default is (8, 3)).

    Returns
    -------
    tuple
        A tuple containing the matplotlib figure and axes objects.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

    ax.plot(
        fes_a[1, :, 50],
        fes_a[2, :, 50],
        "b",
        label=r"MD, $d_\mathrm{OO}=2.6 $Å"
    )
    ax.plot(
        fes_b[1, :, 50],
        fes_b[2, :, 50],
        "r",
        label=r"PIMD, $d_\mathrm{OO}=2.6 $Å",
    )
    ax.plot(
        fes_a[1, :, 60],
        fes_a[2, :, 60],
        "b--",
        label=r"MD, $d_\mathrm{OO}=2.7 $Å",
    )
    ax.plot(
        fes_b[1, :, 60],
        fes_b[2, :, 60],
        "r--",
        label=r"PIMD, $d_\mathrm{OO}=2.7 $Å",
    )
    ax.set_ylim(0.08, 0.6)
    ax.legend(ncols=2, loc="upper right", fontsize=9)
    ax.set_ylabel(r"$F$ (eV)")
    ax.set_xlabel(r"$\Delta C_\mathrm{H}$")
    if save:
        plt.savefig(f"{filename}.png", dpi=600)
        plt.savefig(f"{filename}.pdf")
    if show:
        plt.show()
    return fig, ax


@dataclass
class PlumedFES:
    data: np.ndarray  # numeric columns (after dropping der_* if possible)
    fields: List[str]  # matching names (after dropping der_*), may be []


def load_plumed_fes_drop_der(path: str) -> PlumedFES:
    fields_raw: List[str] = []
    numeric_lines: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            if s.startswith("#!"):
                parts = s.split()
                if len(parts) >= 3 and parts[1] == "FIELDS":
                    fields_raw = parts[2:]
                continue

            if s.startswith("#") or s.startswith("@"):
                continue

            numeric_lines.append(s)

    if not numeric_lines:
        raise ValueError(f"No numeric data found in {path}")

    data = np.loadtxt(numeric_lines)
    if data.ndim == 1:
        data = data[None, :]

    # Drop der_* only if fields align with data columns
    if fields_raw and len(fields_raw) == data.shape[1]:
        keep_idx = [i for i, name in enumerate(fields_raw) if not name.startswith("der_")]
        data = data[:, keep_idx]
        fields = [fields_raw[i] for i in keep_idx]
        return PlumedFES(data=data, fields=fields)

    return PlumedFES(data=data, fields=[])


def plot_plumed_fes(
        path: str,
        ax: Optional[plt.Axes] = None,
        shift_min_to_zero: bool = True,
        levels: int = 30,
) -> Tuple[plt.Figure, plt.Axes]:
    fes = load_plumed_fes_drop_der(path)
    data, fields = fes.data, fes.fields

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ncol = data.shape[1]
    if ncol < 2:
        raise ValueError(f"Need at least 2 columns to plot, got {ncol}")

    # Labels: use FIELDS if available, else defaults
    def lab(i: int, default: str) -> str:
        return fields[i] if fields and i < len(fields) else default

    if ncol == 2:
        x = data[:, 0]
        z = data[:, 1].copy()
        if shift_min_to_zero and np.isfinite(z).any():
            z -= np.nanmin(z)
        order = np.argsort(x)
        ax.plot(x[order], z[order])
        ax.set_xlabel(lab(0, "CV"))
        ax.set_ylabel(lab(1, "FES"))

    else:
        # Use first 3 columns for 2D plot even if more columns exist
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2].copy()

        if shift_min_to_zero and np.isfinite(z).any():
            z -= np.nanmin(z)

        xu = np.unique(x)
        yu = np.unique(y)

        if xu.size * yu.size == z.size:
            # regular grid: sort then reshape
            idx = np.lexsort((x, y))
            xs, ys, zs = x[idx], y[idx], z[idx]
            X = xs.reshape(yu.size, xu.size)
            Y = ys.reshape(yu.size, xu.size)
            Z = zs.reshape(yu.size, xu.size)
            m = ax.contourf(X, Y, Z, levels=levels)
        else:
            # irregular grid
            m = ax.tricontourf(x, y, z, levels=levels)

        ax.set_xlabel(lab(0, "CV1"))
        ax.set_ylabel(lab(1, "CV2"))
        cbar = fig.colorbar(m, ax=ax)
        cbar.set_label(lab(2, "FES"))

    return fig, ax


def plot_plumed_colvar(filename, x_axis='time', figsize=(10, 8)):
    col_names = None
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith("#! FIELDS"):
                # Split the line and remove "#!" and "FIELDS" to get column names
                col_names = line.split()[2:]
                break

    if col_names is None:
        raise ValueError(f"Could not find '#! FIELDS' header in {filename}. Ensure it is a valid PLUMED file.")

    try:
        data = pd.read_csv(filename, sep=r'\s+', comment='#', names=col_names, engine='python')
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None

    # Check if x_axis exists
    if x_axis not in data.columns:
        print(f"Warning: '{x_axis}' column not found. Using index as X-axis.")
        x_data = data.index
        x_label = "Step (Index)"
    else:
        x_data = data[x_axis]
        x_label = x_axis

    plot_cols = [col for col in data.columns if col != x_axis]
    n_plots = len(plot_cols)

    if n_plots == 0:
        print("No variables found to plot.")
        return data, None

    # Create subplots
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)

    # Handle the case where there is only one variable (axes is not a list)
    if n_plots == 1:
        axes = [axes]

    for ax, col in zip(axes, plot_cols):
        ax.plot(x_data, data[col], label=col, linewidth=1.5)
        ax.set_ylabel(col)
        ax.legend(loc='upper right')

    # Set common X label on the bottom plot
    axes[-1].set_xlabel(x_label)
    plt.tight_layout()

    # Show plot
    plt.show()

    return data, fig
