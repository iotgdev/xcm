import unittest
import mock


from xcm.classifiers.bopr import gaus_probit_ratio, weighting, BOPRClassifier


class TestGausProbitRatio(unittest.TestCase):

    def test_lt_min(self):
        self.assertEqual(gaus_probit_ratio(-10), 10)

    def test_gt_max(self):
        self.assertEqual(gaus_probit_ratio(100), 0)


class TestWeighting(unittest.TestCase):

    def test_lt_min(self):
        self.assertEqual(weighting(-10), 1)

    def test_gt_max(self):
        self.assertEqual(weighting(100), 0)


class TestForget(unittest.TestCase):

    def setUp(self):
        with mock.patch('xcm.classifiers.bopr.DEFAULT_HASH_SIZE', new=1):
            self.bopr = BOPRClassifier()

    def test_runtime_error(self):
        with self.assertRaises(ValueError):
            self.bopr.forget(1000)

    def test_no_runtime_error(self):
        self.assertIsNone(self.bopr.forget(1))


class TestPartialFit(unittest.TestCase):

    def setUp(self):
        with mock.patch('xcm.classifiers.bopr.DEFAULT_HASH_SIZE', new=1):
            self.bopr = BOPRClassifier()

    def test_runtime_error(self):
        with self.assertRaises(ValueError):
            self.bopr.partial_fit([[1]], [])

    def test_no_error(self):
        self.assertIsNone(self.bopr.partial_fit([], []))
