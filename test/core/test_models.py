import unittest

import mock
import numpy

from xcm.core.models import XCMModel


# noinspection PyTypeChecker
class TestXCMModelSerialise(unittest.TestCase):

    def test_bad_sizes(self):
        model = XCMModel('test', None, numpy.ndarray(0), numpy.ndarray(0), 5.0, 1.0, 1.0, [])
        model.classifier.weights = []
        model.classifier.n_features = 2
        with self.assertRaises(ValueError):
            model.serialise(mock.Mock())


class TestXCMModelPredict(unittest.TestCase):

    def test_missing_attributes(self):
        model = XCMModel('test', None, numpy.ndarray(0), numpy.ndarray(0), 5.0, None, None, [])
        with self.assertRaises(ValueError):
            model.predict({})
