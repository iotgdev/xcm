import unittest

import mock

from xcm.stores.s3 import S3XCMStore


class TestS3XCMStoreValidateModelId(unittest.TestCase):

    def test_invalid_model_type(self):
        store = S3XCMStore('', '', mock.Mock)
        with self.assertRaises(ValueError):
            store._validate_model_id('a:b:c')
