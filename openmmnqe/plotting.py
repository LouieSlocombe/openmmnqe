import matplotlib.pyplot as plt
import numpy as np


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
