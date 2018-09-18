#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base classes for XCM extendable properties
XCMReader so different log files can be used to create XCM data
XCMClassifier so different machine learning techniques (gradient descent for example) can be applied
XCM for new model classes (training, active, current etc) can be created and used in XCM
"""
from __future__ import unicode_literals

import logging
from abc import ABCMeta, abstractmethod
from six import with_metaclass


logger = logging.getLogger('xcm.base_classes')


class XCMReader(with_metaclass(ABCMeta, object)):
    """All XCM readers must have these methods"""

    @abstractmethod
    def get_ml_features_dict(self, source_date, **kwargs):
        """
        Method to get a dict of machine learning fields from a data source
        returns a dict capable of making an XCMRecord and an AdDataRecord
        """
        pass

    @abstractmethod
    def get_label(self, record, **kwargs):
        pass

    @abstractmethod
    def get_labelled_data(self, source_date, **kwargs):
        """iterate over log data"""
        pass


class XCMClassifier(with_metaclass(ABCMeta, object)):
    """All XCM Classifiers must have these methods"""

    @abstractmethod
    def predict(self, prediction_input):
        pass

    @abstractmethod
    def partial_fit(self, prediction_inputs, labels):
        pass

    @abstractmethod
    def forget(self, strength):
        pass


class XCM(with_metaclass(ABCMeta, object)):
    """All XCM model classes must inherit from this"""

    @property
    def id(self):

        try:
            name = getattr(self, 'name')
            version = getattr(self, 'version')
        except AttributeError:
            logger.exception('Model could not be saved {}'.format(dir(self)))
            raise AttributeError('This Model does not have an id assigned to it yet!')

        return ':'.join((name, self.xcm_class, version))

    @property
    def xcm_class(self):
        return self.__class__.__name__

    @abstractmethod
    def load(self, datastore):
        """Loads a model from a datastore into a model class"""
        pass

    @abstractmethod
    def serialise(self, datastore):
        """Serialises a model from a datastore into a model class"""
        pass

    # noinspection PyMethodMayBeStatic
    def forget(self):
        """method for "blurring" data in the model"""
        pass  # not all models need to forget (Active models, for example)

    @abstractmethod
    def predict(self, feature_dict, exploration=0.0):
        pass


class XCMStore(with_metaclass(ABCMeta, object)):
    """All stores should have this implementation"""

    @abstractmethod
    def list(self, **kwargs):
        pass

    @abstractmethod
    def retrieve(self, model_id):
        pass

    @abstractmethod
    def update(self, model):
        pass

    @abstractmethod
    def create(self, model):
        pass

    @abstractmethod
    def delete(self, model_id):
        pass
