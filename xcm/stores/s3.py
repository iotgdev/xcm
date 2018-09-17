#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Store for accessing XCM models.

Each store is accountable for only one type of model and as such can only interact with 1 type of model

Usage:
>>> from xcm.core.models import XCMTrainingModel
>>> from xcm.stores.s3 import S3XCMStore
>>> store = S3XCMStore('s3_bucket_name', 's3_directory', XCMTrainingModel)
>>> store.list()
['XCMBaseline:XCMTrainingModel:2018_09_04_00_00', ...]
"""
from __future__ import unicode_literals

import ujson
from logging import getLogger

import boto3

from xcm.core.base_classes import XCMStore
from xcm.core.features import get_next_version

_ID_DELIMETER = ':'
_VERSION_DELIMETER = '_'
_S3 = None


def get_s3_connection():
    global _S3
    if _S3 is None:
        _S3 = boto3.resource('s3')
    return _S3


def is_substring_match(string, substring):
    return substring is None or substring in string


def is_valid_model_query(model_id, name=None, version=None):
    """Checks that a model id is valid for the model class (and name and version)"""
    model_name, model_class, model_version = xcm_model_properties(model_id)
    return is_substring_match(model_name, name) and is_substring_match(model_version, version)


def get_model_directory(prefix, model_id):
    model_name, model_class, model_version = xcm_model_properties(model_id)
    return '/'.join((prefix, model_name, model_version.replace(_VERSION_DELIMETER, '/')))


def xcm_model_properties(model_id):
    model_name, model_type, model_id = model_id.split(_ID_DELIMETER)
    return model_name, model_type, model_id


class S3XCMStore(XCMStore):
    """S3 Store for XCM classifiers"""

    def __init__(self, s3_bucket, s3_prefix, model_class):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix.strip('/')
        self.model_class = model_class
        self.logger = getLogger('xcm.store')

    @property
    def model_type(self):
        """get the model_class name as the model type"""
        return self.model_class.__name__

    @property
    def metadata_prefix(self):
        """get the s3 prefix corresponding to the metadata location of the models"""
        return '/'.join((self.s3_prefix, 'metadata', ''))

    @property
    def s3_conn(self):
        """get a singleton s3 connection"""
        return get_s3_connection()

    def _validate_model_id(self, model_id):
        """
        Check if a model id is valid for the store.
        Each store only accesses a specific kind of model

        :type model_id: str
        """
        model_name, model_type, model_version = xcm_model_properties(model_id)
        if model_type != self.model_type:
            raise ValueError('Invalid model_type! Expected {}'.format(self.model_type))
        return model_id

    def list(self, model_name=None, model_version=None):
        """
        List all models subject to filtering

        :type model_name: str|unicode
        :type model_version: str|unicode
        :rtype: list[str]
        """

        models = []

        for obj in self.s3_conn.Bucket(self.s3_bucket).objects.filter(Prefix=self.metadata_prefix):

            try:
                model_id = self._validate_model_id(obj.key[len(self.metadata_prefix):])
                if is_valid_model_query(model_id, model_name, model_version):
                    models.append(model_id)
            except ValueError:
                self.logger.info('bad key name in S3 XCM store: ({})'.format(obj.key))

        return sorted(models, key=lambda x: x.split(':')[2], reverse=True)  # sort by version, newest first

    def retrieve(self, model_id):
        """
        Get a model by its id

        :type model_id: str
        :rtype: xcm.core.base_classes.XCM
        """
        model_name, model_type, model_version = xcm_model_properties(self._validate_model_id(model_id))
        model = self.model_class(model_name, model_version)
        model.load(self)

        return model

    def update(self, model):
        """
        Update a model. Do not bump version
        :type model: xcm.core.base_classes.XCM
        :rtype: str
        """
        self._create_key(self.metadata_prefix + self._validate_model_id(model.id))
        model.serialise(self)
        return model.id

    def delete(self, model_id):
        """
        delete a model by id
        remove model metadata

        :type model_id: str
        :rtype: str
        """
        data_location = get_model_directory(self.s3_prefix, self._validate_model_id(model_id))

        for obj in self.s3_conn.Bucket(self.s3_bucket).objects.filter(Prefix=data_location):
            self._delete_key(obj.key)

        self._delete_key(self.metadata_prefix + model_id)

        return model_id

    def create(self, model):
        """
        create a model using a model class
        assign the model an id and update the metadata

        :type model: xcm.core.base_classes.XCM
        """
        model.version = get_next_version(model.version)
        return self.update(model)

    def _get_buffer(self, model_id, location):
        """
        Get a buffer file for an xcm model

        :type model_id: str|unicode
        :type location: str|unicode
        :rtype: str
        """
        model_id = self._validate_model_id(model_id)
        file_path = get_model_directory(self.s3_prefix, model_id) + '/{}.buffer'.format(location)
        obj = self.s3_conn.Object(self.s3_bucket, file_path)

        return obj.get()['Body'].read()

    def _get_dict(self, model_id, location):
        """
        get a dict file for an xcm model

        :type model_id: str|unicode
        :type location: str|unicode
        :rtype: dict
        """
        model_id = self._validate_model_id(model_id)
        file_path = get_model_directory(self.s3_prefix, model_id) + '/{}.dict'.format(location)
        obj = self.s3_conn.Object(self.s3_bucket, file_path)

        return ujson.loads(obj.get()['Body'].read())

    def _set_buffer(self, model_id, location, data):
        """
        set a buffer file for an xcm model

        :type model_id: str
        :type location: str
        :type data: str
        """
        model_id = self._validate_model_id(model_id)
        file_path = get_model_directory(self.s3_prefix, model_id) + '/{}.buffer'.format(location)
        self._create_key(file_path, body=data)

    def _set_dict(self, model_id, location, data):
        """
        set a dictionary (as json) for an xcm model

        :type model_id: str|unicode
        :type location: str|unicode
        :type data: dict
        """
        model_id = self._validate_model_id(model_id)
        file_path = get_model_directory(self.s3_prefix, model_id) + '/{}.dict'.format(location)
        self._create_key(file_path, body=ujson.dumps(data))

    def _delete_key(self, key_name):
        """
        Delete an S3 key.

        :type key_name: str|unicode
        """
        self.s3_conn.Bucket(self.s3_bucket).delete_objects(Delete={'Objects': [{'Key': key_name}], 'Quiet': True})

    def _create_key(self, key_name, body=""):
        """
        create an S3 key with optional contents

        :type key_name: str|unicode
        :type body: str|unicode
        """
        self.s3_conn.Object(self.s3_bucket, key_name).put(Body=body)
