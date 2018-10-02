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

from xcm.core.features import get_xcm_feature_hashes

logger = logging.getLogger('xcm.base_classes')


class XCMReader(with_metaclass(ABCMeta, object)):
    """All XCM readers must have these methods"""

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

        return ':'.join((name, version))

    @abstractmethod
    def deserialize(self, **kwargs):
        """Loads a model from a datastore into a model class"""
        pass

    @abstractmethod
    def serialize(self):
        """Serialises a model from a datastore into a model class"""
        pass

    # noinspection PyTypeChecker, PyUnresolvedReferences
    def predict(self, feature_dict, exploration=0.):
        """
        Given an auction, make predictions about the likely response.

        :param dict feature_dict: a bidder auction dictionary
        :param float exploration: perturbation factor between 0 and 1
        :rtype: tuple[float, float, float]
        :return: uncalibrated model score, calibrated CTR prediction, CTR boost over the training CTR
        """
        if not self.training_set_ctr or not self.downsampling_rate:
            raise ValueError("Cannot perform prediction: missing defining attribute")

        hashed_features = get_xcm_feature_hashes(feature_dict, self.classifier.n_features, self.features)

        fitness = self.classifier.predict(hashed_features, exploration)

        ctr_estimate = fitness / (fitness + (1. - fitness) / self.downsampling_rate)
        ctr_boost = ctr_estimate / self.training_set_ctr

        return fitness, ctr_estimate, ctr_boost


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
