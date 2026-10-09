"""File ownership and row formatting for Step-first text reporters."""

from __future__ import annotations

import os
from collections.abc import Iterable
from types import TracebackType
from typing import Self


class _TabularReporter:
    """Own a text log; subclasses validate and sample their observables.

    Call the constructor only after validating the reporter's arguments so
    invalid configurations never create or truncate an output file.
    """

    def __init__(self, file: str | os.PathLike[str], header: str) -> None:
        self._out = open(file, "w")
        self._out.write(header + "\n")

    def _write_row(self, step: int, values: Iterable[float]) -> None:
        """Write and flush a complete row in the package's log format."""
        line = f"{step}" + "".join(f"\t{value:.6f}" for value in values)
        self._out.write(line + "\n")
        self._out.flush()

    def close(self) -> None:
        """Close the output file, safely allowing repeated calls."""
        out = getattr(self, "_out", None)
        if out is not None and not out.closed:
            out.close()

    def __enter__(self) -> Self:
        """Return this reporter for use as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the output file when leaving a context."""
        self.close()

    def __del__(self) -> None:
        """Best-effort fallback for callers that did not close the reporter."""
        try:
            self.close()
        except Exception:
            pass
