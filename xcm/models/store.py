#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
classes for storing xcm models
"""
from __future__ import unicode_literals

import logging
import re
import struct
from builtins import bytes

import numpy
from retrying import retry

from ioteclabs_wrapper.core.access import get_labs_dal
from ioteclabs_wrapper.modules.xcm import XCM as LabsXCMAPI
from xcm.models.core import XCMModel

try:
    # noinspection PyCompatibility
    from urllib.parse import urlparse
except ImportError:
    # noinspection PyUnresolvedReferences,PyCompatibility
    from urlparse import urlparse


logger = logging.getLogger('xcm.models.store')


class XCMStore(object):

    _api_to_model = {
        'id': 'model_id',
        'account': 'account',
        'name': 'name',
        'classifierBeta': 'classifier_beta',
        'hashSize': 'hash_size',
        'goodRecords': 'good_records',
        'normalRecords': 'normal_records',
        'sampleRecords': 'sample_records',
        'features': 'features',
        'createdAt': 'created_at',
        'updatedAt': 'updated_at',
    }

    _byte_fields = {
        'preTrainMean': 'pre_train_mean',
        'postTrainMean': 'post_train_mean',
        'preTrainVariance': 'pre_train_variance',
        'postTrainVariance': 'post_train_variance'
    }

    def __init__(self):
        """create API session"""
        self.api = LabsXCMAPI(dal=get_labs_dal())

    @property
    def model_fields(self):
        """
        :rtype: list[tuple[str, str]]
        """
        return list(self._api_to_model.items()) + list(self._byte_fields.items())

    @property
    def params(self):
        """
        :rtype: dict
        """
        return {'complete': True, 'verbose': True}

    @staticmethod
    def snake_case(string):
        """
        Converts string from camelCase to snake_case
        Legacy API responded with camelCase, current API responds with snake_case
        :type string: str
        :rtype: str
        """
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', string)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def _to_model(self, response):
        """converts an api response to a model object"""
        self._format_model_bytes(response)
        return XCMModel(**{v: response.get(k, response.get(self.snake_case(k))) for k, v in self.model_fields})

    def _format_model_bytes(self, response):
        """
        gets a bytes representation from the response
        :type response: dict
        """
        model_id = response['id']
        for i, j in self._byte_fields.items():
            if self._is_internal_resource(response[i]):
                response[i] = numpy.frombuffer(self.api.resources.retrieve(model_id, j, **self.params), dtype='<f4')

    def _to_response(self, model):
        """
        converts a model object to an api response
        :type model: XCMModel
        :rtype: dict
        """
        response = {k: getattr(model, v, None) for k, v in self.model_fields}
        self._format_response_bytes(response)
        return response

    def _format_response_bytes(self, response):
        """
        formats the byte fields in a response
        :type response: dict
        """
        hash_size = response['hashSize']
        for i in self._byte_fields:
            response[i] = struct.pack('<%sf' % hash_size, *response[i].ravel())
            if isinstance(response[i], str):  # python 2 to 3
                response[i] = bytes(response[i])

    def _is_internal_resource(self, value):
        """checks if a model value directs to another API endpoint"""
        try:
            url_value = urlparse(value)
        except (AttributeError, TypeError):
            return False

        # noinspection PyProtectedMember
        current_host_value = urlparse(self.api._dal.url)
        return url_value.netloc == current_host_value.netloc and url_value.scheme == current_host_value.scheme

    # noinspection PyShadowingBuiltins
    @retry(stop_max_attempt_number=3)
    def retrieve(self, id):
        """retrieve an xcm model from the labs API by id"""

        return self._to_model(self.api.retrieve(id, **self.params))

    @retry(stop_max_attempt_number=3)
    def create(self, model):
        """register a model from an object on the labs API"""
        if model.model_id:
            raise ValueError('This model has an id!')

        return self._to_model(self.api.create(params=self.params, **self._to_response(model)))

    @retry(stop_max_attempt_number=3)
    def update(self, model):
        """update a model from an object on the labs API"""
        if not model.model_id:
            raise ValueError('This model has no id!')

        return self._to_model(self.api.update(params=self.params, **self._to_response(model)))

    @retry(stop_max_attempt_number=3)
    def _list_iter(self, **kwargs):
        """
        iterate through models one at a time, yielding
        :type kwargs: dict[str, str]
        :rtype: list[XCMModel]
        """
        params = dict(self.params, offset=0, limit=100)
        # noinspection PyTypeChecker
        params.update(kwargs, fields='id')
        continuing = True

        while continuing:
            response = self.api.list(**params)
            for i in response['results']:
                yield self.retrieve(i['id'])
            params['offset'] += params['limit']
            continuing = response['next']

    def list(self, as_list=False, **kwargs):
        """
        List xcm models, choosing an iterator or list output
        :type as_list: bool
        :type kwargs: dict
        :rtype: list[XCMModel]
        """
        if as_list:
            return list(self._list_iter(**kwargs))
        else:
            return self._list_iter(**kwargs)
