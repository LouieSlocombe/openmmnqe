"""The shared reporter-log helpers every module's log reader is built on."""

from __future__ import annotations

import pytest

from openmmnqe._logs import _column_label


@pytest.mark.parametrize(
    ("column", "label"),
    [
        ("Rg_Proton_H1(nm)", "Proton_H1"),
        ("Expansion_H1(nm)", "H1"),
        ("Distance_H1-O2(nm)", "H1-O2"),
        ("Kcv_0(kJ/mol)", "0"),
        ("KE_cv(kJ/mol)", "KE_cv"),  # no observable prefix: only the unit goes
        ("Rg_Bare", "Bare"),  # no unit suffix: only the prefix goes
        ("Step", "Step"),
    ],
)
def test_column_labels_strip_only_a_known_prefix_and_the_unit(column: str, label: str) -> None:
    assert _column_label(column) == label
