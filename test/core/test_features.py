import unittest
import mock

import datetime

from xcm.core.features import get_next_version


class TestGetNextVersion(unittest.TestCase):

    def test_attribute_error(self):
        with mock.patch('xcm.core.features.logging') as f, mock.patch('xcm.core.features.datetime') as d:
            d.date.today.return_value = datetime.datetime(2017, 1, 1)
            self.assertEqual(get_next_version(), '2017_01_01_00')
            self.assertEqual(f.warning.called, True)

    def test_next_version(self):
        with mock.patch('xcm.core.features.datetime') as d:
            d.date.today.return_value = datetime.datetime(2017, 1, 1)
            self.assertEqual(get_next_version('2017_01_01_00'), '2017_01_01_01')
