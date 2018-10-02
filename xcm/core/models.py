#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import logging

from xcm.classifiers.bopr import BOPRClassifier
from xcm.core.base_classes import XCM
from xcm.core.features import get_xcm_feature_hashes


class XCMModel(XCM):
    """
    Implementation of the XCM (Cross Customer Model) interface for
    - loading and storing the model
    - predicting on bidder auctions
    """

    def __init__(self, name, version=None):
        """
        :type name: str
        :type version: str|None
        """
        self.name = name
        self.version = version
        self.classifier = BOPRClassifier(None, None, 0.1)

        self.downsampling_rate = None
        self.training_set_ctr = None

        self.features = None

    # noinspection PyUnusedLocal
    def deserialize(self, beta, variance, weights, good_records, normal_records, sample_records, features, **kwargs):
        """
        Loads the model's buffers and metadata from the data store

        :type beta: int
        :type variance: numpy.ndarray
        :type weights: numpy.ndarray
        :type good_records: int
        :type normal_records: int
        :type sample_records: int
        :type features: list[str]
        :type kwargs: dict
        """
        self.classifier = self.classifier.__class__(weights, variance, beta)
        self.downsampling_rate = float(normal_records) / float(normal_records + sample_records)
        self.training_set_ctr = good_records / (good_records + normal_records + sample_records + self.downsampling_rate)
        self.features = features

    def serialize(self):
        """Persists the model to the data store"""
        raise RuntimeError('XCMModels are not serializable and can not be saved!')


class XCMTrainingModel(XCM):
    """
    Implementation of the XCM training model.
    """

    def __init__(self, name, version=None):
        self.name = name
        self.version = version

        self.classifier = BOPRClassifier(None, None, 0.1)

        # Fields to store counters of label counts in training data.
        self.good_behaviour_count = 0
        self.normal_behaviour_count = 0
        self.unused_sample_count = 0

        self.trained_days = []
        self.features = None

    def deserialize(self, beta, initial_variance, initial_weights, variance, weights, trained_days, good_records,
                    normal_records, sample_records, features):
        """
        deserializes the model's features as kwargs into the object

        :type beta: float
        :type initial_variance: numpy.ndarray
        :type initial_weights: numpy.ndarray
        :type variance: numpy.ndarray
        :type weights: numpy.ndarray
        :type trained_days: list[int]
        :type good_records: int
        :type normal_records: int
        :type sample_records: int
        :type features: list[str]
        """
        self.classifier = self.classifier.__class__(initial_weights, initial_variance, beta)
        self.classifier.variance = variance
        self.classifier.weights = weights

        self.good_behaviour_count = good_records
        self.normal_behaviour_count = normal_records
        self.unused_sample_count = sample_records

        self.trained_days = trained_days
        self.features = features

    def serialize(self):
        """Serialise the model so a datastore can save it"""
        return {
            'beta': self.classifier.beta,
            'initial_weights': self.classifier.initial_weights,
            'initial_variance': self.classifier.initial_variance,
            'variance': self.classifier.variance,
            'weights': self.classifier.weights,

            'trained_days': self.trained_days,
            'good_records': self.good_behaviour_count,
            'normal_records': self.normal_behaviour_count,
            'sample_records': self.unused_sample_count,
            'features': self.features
        }

    def partial_fit(self, data, labels):
        """
        Converts data to a list of hashed features and updates the encapsulated FFM model.

        :param list[dict] data: should contain a list of canonical auctions.
        :param list[int|None] labels: list of the labels corresponding to each element of data.
        """
        hashed_data = []
        valid_labels = []
        for i, (record, label) in enumerate(zip(data, labels)):
            if label is not None:
                features = get_xcm_feature_hashes(record, self.classifier.n_features, self.features)
                if not features.shape[0]:
                    continue

                hashed_data.append(features)
                valid_labels.append(label)

        self.good_behaviour_count += valid_labels.count(1)
        self.normal_behaviour_count += valid_labels.count(0)
        self.unused_sample_count += labels.count(None)

        logging.info('Oversampling clicks by %s', int(0.1 / self.oversampling_rate))
        for i in range(len(valid_labels)):
            if valid_labels[i] == 1:
                for j in range(int(0.1 / self.oversampling_rate)):
                    valid_labels.append(1)
                    hashed_data.append(hashed_data[i])

        logging.info('Applying dataset with %s clicks and %s imps', valid_labels.count(1), valid_labels.count(0))

        for j in range(10):
            for i in range(0, len(hashed_data), 1000):
                self.classifier.forget(0.001)
                self.classifier.partial_fit(hashed_data[i:i + 1000], valid_labels[i:i + 1000])

    @property
    def training_set_ctr(self):
        """
        Return the measured CTR in the unsampled data used to run.
        """
        record_count = self.good_behaviour_count + self.normal_behaviour_count + self.unused_sample_count
        num = self.good_behaviour_count + 1.
        denom = record_count + self.downsampling_rate
        return float(num) / denom

    @property
    def downsampling_rate(self):
        """
        Return the measured downsampling rate.

        NOTE: this assumes only the impressions are downsampled. IF this is ever different, then calculation of
        the calibration correction would need to change.
        """
        if self.normal_behaviour_count + self.unused_sample_count == 0:
            return 1.

        return float(self.normal_behaviour_count) / float(self.normal_behaviour_count + self.unused_sample_count)

    @property
    def oversampling_rate(self):
        """Return the measured oversampling rate for good behaviour"""
        inv_downsampling_rate = 1 / self.downsampling_rate

        num = self.good_behaviour_count + 1
        denom = self.good_behaviour_count + self.normal_behaviour_count + inv_downsampling_rate

        return float(num) / denom
