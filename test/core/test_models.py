import unittest

from xcm.core.models import XCMModel, XCMTrainingModel


class TestXCMModelPredict(unittest.TestCase):

    def test_missing_attributes(self):
        model = XCMModel('test')
        with self.assertRaises(ValueError):
            model.predict({})


class TestXCMTrainingModelPredict(unittest.TestCase):

    def test_missing_attributes(self):
        model = XCMTrainingModel('test')
        with self.assertRaises(ValueError):
            model.predict({})


class TestXCMTrainingModelTrainingSetCTR(unittest.TestCase):

    def test_denom_is_zero(self):
        model = XCMTrainingModel('test')
        self.assertEqual(model.training_set_ctr, 0.0)


class TestXCMTrainingModelDownsamplingRate(unittest.TestCase):

    def test_denom_is_zero(self):
        model = XCMTrainingModel('test')
        self.assertEqual(model.downsampling_rate, 1.0)


class TestXCMTrainingModelOversamplingRate(unittest.TestCase):

    def test_denom_is_zero(self):
        model = XCMTrainingModel('test')
        self.assertEqual(model.oversampling_rate, 0.0)
