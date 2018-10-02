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
from functools import partial
from logging import getLogger

import boto3

from xcm.core.base_classes import XCMStore
from xcm.core.features import get_next_version
from xcm.utilities.serialisers import unpack_float_array, pack_float_array

_ID_DELIMETER = ':'
_S3 = None


def get_s3_connection():
    """
    Get singleton S3 Connection
    :rtype boto3.resources.base.ServiceResource:
    """
    global _S3
    if _S3 is None:
        _S3 = boto3.resource('s3')
    return _S3


def is_substring_match(string, substring):
    """Checks if a substring (optional, returns True if absent) can be found in a string"""
    return substring is None or substring in string


def is_valid_model_query(model_id, name=None, version=None):
    """Checks that a model id is valid for the model class (and name and version)"""
    model_name, model_version = xcm_model_properties(model_id)
    return is_substring_match(model_name, name) and is_substring_match(model_version, version)


def get_model_directory(prefix, model_id):
    """Get the location of the model prefixes"""
    model_name, model_version = xcm_model_properties(model_id)
    return '/'.join((prefix, 'models', model_name, model_version))


def xcm_model_properties(model_id):
    model_name, model_version = model_id.split(_ID_DELIMETER)
    return model_name, model_version


class S3XCMStore(XCMStore):
    """S3 Store for XCM classifiers"""

    CLASSIFIER_PARAMS = 'BOPR_PARAMS'
    TRAINING_DATA_VALUES = 'TRAINING_DATA_VALUES'

    def __init__(self, s3_bucket, s3_prefix, model_class):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix.strip('/')
        self.model_class = model_class
        self.logger = getLogger('xcm.store')

    @property
    def metadata_prefix(self):
        """get the s3 prefix corresponding to the metadata location of the models"""
        return '/'.join((self.s3_prefix, 'metadata', ''))

    def list(self, model_name=None, model_version=None):
        """
        List all models subject to filtering

        :type model_name: str|unicode
        :type model_version: str|unicode
        :rtype: list[str]
        """

        models = []

        for obj in get_s3_connection().Bucket(self.s3_bucket).objects.filter(Prefix=self.metadata_prefix):

            try:
                model_id = obj.key[len(self.metadata_prefix):]
                if is_valid_model_query(model_id, model_name, model_version):
                    models.append(model_id)
            except (Exception, ):
                self.logger.info('bad key name in S3 XCM store: ({})'.format(obj.key))

        return sorted(models, key=lambda x: x.split(':')[1], reverse=True)  # sort by version, newest first

    def retrieve(self, model_id):
        """
        Get a model by its id

        :type model_id: str
        :rtype: xcm.core.base_classes.XCM
        """
        model_name, model_version = xcm_model_properties(model_id)

        get_dict = partial(self._get_dict, model_id)
        get_buffer = partial(self._get_buffer, model_id)

        classifier_params = get_dict(self.CLASSIFIER_PARAMS)

        initial_variance = unpack_float_array(classifier_params['initial_variance_shape'],
                                              get_buffer('initial_variance'))
        initial_weights = unpack_float_array(classifier_params['initial_weights_shape'], get_buffer('initial_weights'))
        variance = unpack_float_array(classifier_params['variance_shape'], get_buffer('variance'))
        weights = unpack_float_array(classifier_params['weights_shape'], get_buffer('weights'))

        training_params = get_dict(self.TRAINING_DATA_VALUES)

        serialized_model = {
            'beta': classifier_params['beta'],
            'initial_variance': initial_variance,
            'initial_weights': initial_weights,
            'variance': variance,
            'weights': weights,

            'trained_days': training_params['completed_training_days'],
            'good_records': training_params['good_record_count'],
            'normal_records': training_params['normal_record_count'],
            'sample_records': training_params['sample_record_count'],
            'features': training_params['record_features'],
        }

        model = self.model_class(model_name, model_version)
        model.deserialize(**serialized_model)

        return model

    def update(self, model):
        """
        Update a model. Do not bump version
        :type model: xcm.core.base_classes.XCM
        :rtype: str
        """
        serialized_model = model.serialize()

        set_dict = partial(self._set_dict, model.id)
        set_buffer = partial(self._set_buffer, model.id)

        initial_variance_shape, initial_variance = pack_float_array(serialized_model['initial_variance'])
        initial_weights_shape, initial_weights = pack_float_array(serialized_model['initial_weights'])
        variance_shape, variance = pack_float_array(serialized_model['variance'])
        weights_shape, weights = pack_float_array(serialized_model['weights'])

        classifier_params = {
            'initial_variance_shape': initial_variance_shape,
            'initial_weights_shape': initial_weights_shape,
            'variance_shape': variance_shape,
            'weights_shape': weights_shape,
            'beta': serialized_model['beta']
        }

        training_params = {
            'completed_training_days': serialized_model['trained_days'],
            'good_record_count': serialized_model['good_records'],
            'normal_record_count': serialized_model['normal_records'],
            'sample_record_count': serialized_model['sample_records'],
            'record_features': serialized_model['features']
        }

        set_buffer('initial_variance', initial_variance)
        set_buffer('initial_weights', initial_weights)
        set_buffer('variance', variance)
        set_buffer('weights', weights)

        set_dict(self.CLASSIFIER_PARAMS, classifier_params)
        set_dict(self.TRAINING_DATA_VALUES, training_params)

        return model.id

    def delete(self, model_id):
        """
        delete a model by id
        remove model metadata

        :type model_id: str
        :rtype: str
        """
        data_location = get_model_directory(self.s3_prefix, model_id)

        for obj in get_s3_connection().Bucket(self.s3_bucket).objects.filter(Prefix=data_location):
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
        model_id = self.update(model)
        self._create_key(self.metadata_prefix + model.id)
        return model_id

    def _get_buffer(self, model_id, location):
        """
        Get a buffer file for an xcm model

        :type model_id: str|unicode
        :type location: str|unicode
        :rtype: str
        """
        file_path = get_model_directory(self.s3_prefix, model_id) + '/{}.buffer'.format(location)
        obj = get_s3_connection().Object(self.s3_bucket, file_path)

        return obj.get()['Body'].read()

    def _get_dict(self, model_id, location):
        """
        get a dict file for an xcm model

        :type model_id: str|unicode
        :type location: str|unicode
        :rtype: dict
        """
        file_path = get_model_directory(self.s3_prefix, model_id) + '/{}.dict'.format(location)
        obj = get_s3_connection().Object(self.s3_bucket, file_path)

        return ujson.loads(obj.get()['Body'].read())

    def _set_buffer(self, model_id, location, data):
        """
        set a buffer file for an xcm model

        :type model_id: str
        :type location: str
        :type data: str
        """
        file_path = get_model_directory(self.s3_prefix, model_id) + '/{}.buffer'.format(location)
        self._create_key(file_path, body=data)

    def _set_dict(self, model_id, location, data):
        """
        set a dictionary (as json) for an xcm model

        :type model_id: str|unicode
        :type location: str|unicode
        :type data: dict
        """
        file_path = get_model_directory(self.s3_prefix, model_id) + '/{}.dict'.format(location)
        self._create_key(file_path, body=ujson.dumps(data))

    def _delete_key(self, key_name):
        """
        Delete an S3 key.

        :type key_name: str|unicode
        """
        get_s3_connection().Bucket(self.s3_bucket).delete_objects(
            Delete={'Objects': [{'Key': key_name}], 'Quiet': True})

    def _create_key(self, key_name, body=""):
        """
        create an S3 key with optional contents

        :type key_name: str|unicode
        :type body: str|unicode
        """
        get_s3_connection().Object(self.s3_bucket, key_name).put(Body=body)
