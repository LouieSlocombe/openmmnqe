import matplotlib.pyplot as plt
import numpy as np

def n_plot(xlab,
           ylab,
           xs=14,
           ys=14):
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
    if times is None:
        times = np.arange(len(fes_arrays))

    # If there are more than 5 times, select only the last 5
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
