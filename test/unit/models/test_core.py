#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from xcm.models.core import XCMModel


class TestXCMModel(unittest.TestCase):

    def setUp(self):
        self.default_model = XCMModel('default', None)

    def test_default_training_set_ctr(self):
        self.assertEqual(self.default_model.training_set_ctr, 0.0)

    def test_default_downsampling_rate(self):
        self.assertEqual(self.default_model.downsampling_rate, 0.0)

    def test_default_oversampling_rate(self):
        self.assertEqual(self.default_model.oversampling_rate, 1.0)
