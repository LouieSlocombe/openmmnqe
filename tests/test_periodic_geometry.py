"""Characterize the shared reduced-box displacement operation."""

import numpy as np
import pytest

from openmmnqe.tools import _minimum_image_displacements


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("box", [
    np.diag([2.0, 2.2, 2.6]),
    np.array([[2.0, 0, 0], [0.5, 2.2, 0], [0.3, 0.4, 2.6]]),
])
def test_displacements_remove_lattice_translations_without_changing_box(dtype, box):
    box = box.astype(dtype)
    original_box = box.copy()
    local = np.array([[0.2, 0.3, 0.4], [-0.1, 0.25, -0.3]], dtype=dtype)
    translations = np.array([[2, -3, 4], [-2, 1, -3]], dtype=dtype)
    displaced = local + translations @ box
    assert _minimum_image_displacements(displaced, box) is None
    np.testing.assert_allclose(displaced, local, atol=2e-6, rtol=0)
    assert displaced.dtype == dtype
    np.testing.assert_array_equal(box, original_box)


def test_half_box_ties_keep_numpy_nearest_even_convention():
    displacements = np.array([
        [1, -1, 3], [-3, 5, -5],
    ], dtype=float)
    _minimum_image_displacements(displacements, np.eye(3) * 2)
    np.testing.assert_array_equal(displacements, [[1, -1, -1], [1, 1, -1]])


def test_empty_displacements_keep_their_shape():
    displacements = np.empty((0, 3))
    _minimum_image_displacements(displacements, np.eye(3))
    assert displacements.shape == (0, 3)
