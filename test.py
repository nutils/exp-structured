import unittest
import numpy as np

from nutils.experimental.structured import create_grid
from nutils.units.typing import Length


class TestGrid(unittest.TestCase):
    def test_grid(self):
        def esize(x):
            return 0.2 if x < 1 else 0.5

        grid, where = create_grid([0, 2, 1], esize)
        np.testing.assert_almost_equal(
            actual=grid, desired=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0], decimal=10
        )
        self.assertEqual(where.tolist(), [0, 7, 5])

    def test_merge(self):
        def esize(x):
            return 0.2 if x < 1 else 0.5

        grid, where = create_grid([0, 2, 1, 1e-10], esize)
        np.testing.assert_almost_equal(
            actual=grid, desired=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0], decimal=10
        )
        self.assertEqual(where.tolist(), [0, 7, 5, 0])

    def test_units(self):
        m = Length("m")

        def esize(x):
            return 0.2 * m if x < m else 0.5 * m

        grid, where = create_grid([0 * m, 2 * m, 1 * m], esize)
        np.testing.assert_almost_equal(
            actual=grid / m,
            desired=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0],
            decimal=10,
        )
        self.assertEqual(where.tolist(), [0, 7, 5])
