#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Models for xcm

Usage:
>>> from xcm.models.core import XCMModel
>>> model = XCMModel('name', 'account')
>>> model.partial_fit([{'test': 1}], [1])
>>> model.predict({'test': 1})
"""


import logging
import mmh3 as mmh3
import numpy
from six import iteritems

from xcm.ml_utils.classifiers import BOPRClassifier


logger = logging.getLogger('xcm.models.core')


class XCMModel(object):

    def __init__(self, name, account, model_id=None, beta=5, pre_train_mean=None, post_train_mean=None,
                 pre_train_variance=None, post_train_variance=None, hash_size=1000000, good_records=0, normal_records=0,
                 sample_records=0, features=None, created_at=None, updated_at=None):
        self.name = name
        self.model_id = model_id
        self.account = account

        self.classifier = BOPRClassifier(
            pre_train_mean, pre_train_variance, post_train_mean, post_train_variance, beta, hash_size)
        self.hash_size = hash_size

        self.good_records = good_records
        self.normal_records = normal_records
        self.sample_records = sample_records

        self.features = features or []

        self.created_at = created_at
        self.updated_at = updated_at

    @property
    def training_set_ctr(self):
        """
        Return the measured CTR in the unsampled data used to run.
        """
        record_count = self.good_records + self.normal_records + self.sample_records
        num = self.good_records
        denom = record_count + self.downsampling_rate
        return (float(num) / denom) if denom else 0.0

    @property
    def downsampling_rate(self):
        """
        Return the measured downsampling rate.

        NOTE: this assumes only the impressions are downsampled. IF this is ever different, then calculation of
        the calibration correction would need to change.
        """
        denom = self.good_records + self.sample_records
        return (float(self.normal_records) / denom) if denom else 0.0

    @property
    def oversampling_rate(self):
        """Return the measured oversampling rate for good behaviour"""
        try:
            inv_downsampling_rate = 1 / self.downsampling_rate
        except ZeroDivisionError:
            return 1

        num = self.good_records
        denom = self.good_records + self.normal_records + inv_downsampling_rate

        return float(num) / denom

    def get_xcm_feature_hashes(self, canon_auction):
        """
        Given an auction in bidder format returns a list of hashed features

        :param dict canon_auction: an auction dictionary as per iotec auction canonical
        :return: a list of hashed features
        :rtype: numpy.array
        """
        filtered_auction = {k: v for k, v in canon_auction.items() if k in self.features}
        return numpy.unique(numpy.array(self.get_hashed_features(filtered_auction)))

    def get_hashed_features(self, record):
        """
        Returns the hashed features corresponding to a given XCM record dictionary.
        The values are expected to be strings or lists

        :param dict record: an XCM record dictionary
        :rtype: numpy.array
        """
        indices = []
        for key, value in iteritems(record):
            if isinstance(value, (list, set)):
                indices += [mmh3.hash("%s-%s" % (key, item)) for item in value]
            elif value:
                indices.append(mmh3.hash("%s-%s" % (key, value)))

        return numpy.array(indices) % self.hash_size

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
                features = self.get_xcm_feature_hashes(record)
                if not features.shape[0]:
                    continue

                hashed_data.append(features)
                valid_labels.append(label)

        self.good_records += valid_labels.count(1)
        self.normal_records += valid_labels.count(0)
        self.sample_records += labels.count(None)

        logger.info('Oversampling clicks by %s', int(0.1 / self.oversampling_rate))
        for i in range(len(valid_labels)):
            if valid_labels[i] == 1:
                for j in range(int(0.1 / self.oversampling_rate)):  # todo: better oversampling
                    valid_labels.append(1)
                    hashed_data.append(hashed_data[i])

        logger.info('Applying dataset with %s clicks and %s imps', valid_labels.count(1), valid_labels.count(0))

        for j in range(10):  # todo: raise question of j
            for i in range(0, len(hashed_data), 1000):
                self.classifier.forget(0.001)
                self.classifier.partial_fit(hashed_data[i:i + 1000], valid_labels[i:i + 1000])

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

        hashed_features = self.get_xcm_feature_hashes(feature_dict)

        fitness = self.classifier.predict(hashed_features, exploration)

        ctr_estimate = fitness / (fitness + (1. - fitness) / self.downsampling_rate)
        ctr_boost = ctr_estimate / self.training_set_ctr

        return fitness, ctr_estimate, ctr_boost
