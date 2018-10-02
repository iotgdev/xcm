import unittest

import mock

from xcm.stores.s3 import S3XCMStore


class TestS3XCMStoreList(unittest.TestCase):

    def setUp(self):
        mock_s3 = mock.Mock()
        mock_s3.key = 1  # using bad type to raise the error
        mock_s3.Bucket.return_value = mock_s3.objects = mock_s3.filter = mock_s3
        mock_s3.filter.return_value = [mock_s3]
        self.mock_s3 = mock_s3

    def test_bad_model_id(self):
        store = S3XCMStore('', '', mock.Mock())
        store.logger = mock.Mock()
        with mock.patch('xcm.stores.s3.get_s3_connection', return_value=self.mock_s3):
            self.assertEqual(store.list(), [])
            self.assertEqual(store.logger.info.called, True)