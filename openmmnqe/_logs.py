"""Shared readers for the package's tab-separated reporter logs.

Every reporter in this package writes the same shape of file: a header row
of tab-separated column names beginning with ``Step``, then one row per
report.  The helpers here read that format back and resolve column
selections against it, so :mod:`openmmnqe.reporters` and
:mod:`openmmnqe.adqtb` share one parser rather than growing two that drift
apart.  Nothing here is public; the modules that own each log wrap these in
their own readers so error messages name the log the caller actually asked
for.
"""
from __future__ import annotations

import os
from collections.abc import Iterable
from numbers import Real

import numpy as np

from ._validation import require_integer


def _read_reporter_log(file: str | os.PathLike[str],
                       description: str,
                       ) -> tuple[list[str], np.ndarray]:
    """
    Read any of this package's tab-separated, Step-first reporter logs.

    Parameters
    ----------
    file : str or os.PathLike
        Log to read.
    description : str
        What the log is, used verbatim in error messages, e.g.
        ``"expansion log"``.

    Returns
    -------
    header : list of str
        Column names, the first of which is ``"Step"``.
    values : numpy.ndarray
        Row values, shaped ``(n_rows, len(header))``.

    Raises
    ------
    ValueError
        If the header is malformed or duplicated, the file holds no data
        rows, or a row does not match the header.
    """
    with open(file) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        has_data = any(line.strip() for line in handle)
    if len(header) < 2 or header[0] != "Step":
        raise ValueError(f"{description} must start with a Step column")
    if len(set(header)) != len(header):
        raise ValueError(f"{description} contains duplicate column names")
    if not has_data:
        raise ValueError(f"{description} contains no data rows")

    try:
        values = np.loadtxt(file, delimiter="\t", skiprows=1, ndmin=2)
    except ValueError as exc:
        raise ValueError(
            f"could not parse {description} {file!s}"
        ) from exc
    if values.shape[1] != len(header):
        raise ValueError(f"{description} rows do not match its header")
    return header, values


def _select_log_columns(header: list[str],
                        requested: str | Iterable[str] | None,
                        prefixes: tuple[str, ...],
                        description: str) -> list[str]:
    """
    Resolve a requested column selection against a log header.

    Parameters
    ----------
    header : list of str
        Column names read from the log.
    requested : str or iterable of str or None
        Columns wanted. A bare string selects one column; None selects every
        column carrying one of *prefixes*.
    prefixes : tuple of str
        Name prefixes that mark a column as belonging to this observable.
    description : str
        Observable name, used in error messages.

    Returns
    -------
    list of str
        The selected column names.

    Raises
    ------
    ValueError
        If a requested column is absent, or the log carries no expansion
        columns at all.
    """
    available = [
        name for name in header
        if any(name.startswith(prefix) for prefix in prefixes)
    ]
    if requested is None:
        selected = available
    elif isinstance(requested, str):
        selected = [requested]
    else:
        selected = list(requested)

    missing = [name for name in selected if name not in available]
    if missing:
        raise ValueError(
            f"unknown {description} column(s): {', '.join(missing)}"
        )
    if not selected and description == "expansion":
        raise ValueError("expansion log contains no expansion columns")
    return selected


def _column_label(column: str) -> str:
    """
    Turn a reporter column name into a compact legend label.

    Parameters
    ----------
    column : str
        Column name, e.g. ``"Rg_Proton_H1(nm)"`` or ``"KE_cv(kJ/mol)"``.

    Returns
    -------
    str
        The name without its observable prefix or unit suffix, e.g.
        ``"Proton_H1"`` or ``"KE_cv"``.
    """
    label = column
    for prefix in ("Expansion_", "Rg_", "Distance_", "Kcv_"):
        if label.startswith(prefix):
            label = label[len(prefix):]
            break
    if label.endswith(")") and "(" in label:
        label = label[:label.rindex("(")]
    return label


def _block_averaged_columns(file: str | os.PathLike[str],
                            description: str, *,
                            discard: float = 0.0,
                            blocks: int = 5,
                            ) -> dict[str, tuple[float, float]]:
    """
    Average every non-``Step`` column of a reporter log, with block errors.

    Consecutive samples from one trajectory are correlated, so the naive
    ``std / sqrt(n)`` understates the uncertainty. The retained rows are
    instead split into *blocks* contiguous chunks and the error taken from
    the scatter of the block means.

    Parameters
    ----------
    file : str or os.PathLike
        Log to average.
    description : str
        What the log is, used verbatim in error messages, e.g.
        ``"thermodynamic log"``.
    discard : float, optional
        Leading fraction of the rows to drop as equilibration, in ``[0, 1)``.
        Default is 0.0.
    blocks : int, optional
        Number of blocks the retained rows are split into. Default is 5.

    Returns
    -------
    dict of str to tuple of float
        ``{column: (mean, standard_error)}`` for every column but ``"Step"``.

    Raises
    ------
    ValueError
        If *discard* is outside ``[0, 1)``, *blocks* is below 2, or too few
        rows survive to fill the blocks.
    TypeError
        If *blocks* is not an integer.
    """
    blocks = require_integer(blocks, name="blocks", minimum=2)
    if isinstance(discard, bool) or not isinstance(discard, Real):
        raise ValueError("discard must be a number in [0, 1)")
    discard = float(discard)
    if not np.isfinite(discard) or not 0.0 <= discard < 1.0:
        raise ValueError("discard must be a number in [0, 1)")

    header, values = _read_reporter_log(file, description)
    retained = values[int(discard * len(values)):]
    if len(retained) < blocks:
        raise ValueError(
            f"{description} has {len(retained)} rows after discarding, "
            f"too few for {blocks} blocks"
        )

    # Drop the leading remainder rather than the trailing one: the tail is the
    # better-equilibrated end of a trajectory.
    block_size = len(retained) // blocks
    retained = retained[len(retained) - block_size * blocks:]
    block_means = retained.reshape(blocks, block_size, -1).mean(axis=1)

    means = retained.mean(axis=0)
    errors = block_means.std(axis=0, ddof=1) / np.sqrt(blocks)
    return {
        name: (float(means[index]), float(errors[index]))
        for index, name in enumerate(header)
        if name != "Step"
    }
