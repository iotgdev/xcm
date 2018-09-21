import unittest

import numpy

from xcm.utilities.evaluation import roc_curve


class TestRocCurve(unittest.TestCase):

    def test_value_error(self):
        with self.assertRaises(ValueError):
            roc_curve([], numpy.ndarray([1, 2, 3]))
