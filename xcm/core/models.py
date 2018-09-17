#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import logging
import struct
from functools import partial

import numpy as np

from xcm.canonical.xcm_time import xcmd_dt
from xcm.classifiers.bopr import BOPRClassifier
from xcm.core.base_classes import XCM
from xcm.core.features import get_xcm_feature_hashes, DEFAULT_FEATURES
from xcm.utilities.serialisers import pack_float_array, unpack_float_array


class XCMModel(XCM):
    """
    Implementation of the XCM (Cross Customer Model) interface for
    - loading and storing the model
    - predicting on bidder auctions
    """

    def __init__(self, name, version=None, weights=None, variances=None, beta=None, downsampling_rate=None,
                 training_set_ctr=None, features=None):
        """
        :type name: str
        :type version: str|None
        :param np.array weights: array of values for the weights
        :param np.array variances: array of variances for the weights
        :param float beta: defines the learning rate (lower implies faster learning)
        :param float downsampling_rate: downsampling rate used for the negative class samples while learning
            (used to predict the CTR)
        :param float training_set_ctr: the CTR observed in the training set
        """
        self.name = name
        self.version = version
        self.classifier = BOPRClassifier(weights, variances, beta)

        self.downsampling_rate = downsampling_rate
        self.training_set_ctr = training_set_ctr
        self.features = features

        self.logger = logging.getLogger('xcm.xcmmodel')

    def load(self, data_store):
        """
        Loads the model's buffers and metadata from the data store

        :type data_store: xcm.stores.s3.S3XCMStore
        """
        # noinspection PyProtectedMember
        weights = data_store._get_buffer(self.id, 'MODEL_WEIGHTS')
        # noinspection PyProtectedMember
        variances = data_store._get_buffer(self.id, 'MODEL_VARIANCES')
        # noinspection PyProtectedMember
        model_values = data_store._get_dict(self.id, 'MODEL_VALUES')

        weights = np.ndarray(shape=(len(weights) // 4,), dtype='<f4', buffer=weights)  # len(buffer)=4e6, len(shape)=1e6
        variances = np.ndarray(shape=(len(weights) // 4,), dtype='<f4', buffer=variances)
        beta = model_values['beta']

        self.classifier = BOPRClassifier(weights, variances, beta)

        self.downsampling_rate = model_values['downsampling_rate']
        self.training_set_ctr = model_values['training_set_ctr']
        self.features = model_values.get('features', DEFAULT_FEATURES)

    def serialise(self, data_store):
        """
        Persists the model to the data store

        :type data_store: xcm.stores.s3.S3XCMStore
        """
        n_weights = len(self.classifier.weights)
        n_variances = len(self.classifier.weights)
        if n_weights != self.classifier.n_features or n_variances != self.classifier.n_features:
            raise ValueError('Incompatible sizes for weights and variances arrays')

        # noinspection PyProtectedMember
        set_buffer = partial(data_store._set_buffer, self.id)

        set_buffer('MODEL_WEIGHTS', struct.pack("<%sf" % self.classifier.n_features, *self.classifier.weights))
        set_buffer('MODEL_VARIANCES', struct.pack("<%sf" % self.classifier.n_features, *self.classifier.variance))

        model_data = {'beta': self.classifier.beta, 'downsampling_rate': self.downsampling_rate,
                      'training_set_ctr': self.training_set_ctr, 'features': self.features}

        # noinspection PyProtectedMember
        data_store._set_dict(self.id, 'MODEL_VALUES', model_data)

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

        try:
            ctr_boost = ctr_estimate / self.training_set_ctr
        except KeyError:
            ctr_boost = 0

        return fitness, ctr_estimate, ctr_boost


class XCMTrainingModel(XCM):
    """
    Implementation of the XCM training model.
    """
    DATA_STORE_KEY_CLASSIFIER_PARAMS = 'BOPR_PARAMS'
    DATA_STORE_KEY_TRAINING_DATA_VALUES = 'TRAINING_DATA_VALUES'
    DATA_STORE_KEY_AUC_SCORES = 'AUC_SCORES'
    DATA_STORE_KEY_FEATURE_SUPPORT = 'FEATURE_SUPPORT'

    def __init__(self, name, version=None, weights=None, variances=None, beta=None, features=None):
        self.name = name
        self.version = version

        self.classifier = BOPRClassifier(weights, variances, beta)

        # Fields to store counters of label counts in training data.
        self.good_behaviour_count = 0
        self.normal_behaviour_count = 0
        self.unused_sample_count = 0
        self.trained_days = []
        self.features = features

        # AUC scores:
        self.auc_scores = {}

    def configure_classifier(self, classifier_data):
        pass

    def load(self, data_store):
        """
        If the mnemonic member variable has been set, loads the model data from the data
        store otherwise it raises an ACEException. The model data is assigned to the member
        variables _vectors and _intercept.

        :type data_store: xcm.stores.s3.S3XCMStore
        """
        # noinspection PyProtectedMember
        get_dict = partial(data_store._get_dict, self.id)
        # noinspection PyProtectedMember
        get_buffer = partial(data_store._get_buffer, self.id)

        model_params = get_dict(self.DATA_STORE_KEY_CLASSIFIER_PARAMS)

        beta = model_params['beta']
        initial_variance = unpack_float_array(model_params['initial_sigma2_shape'], get_buffer('initial_sigma2_data'))
        initial_weights = unpack_float_array(model_params['initial_mu_shape'], get_buffer('initial_mu_data'))

        variance = unpack_float_array(model_params['sigma2_shape'], get_buffer('sigma2_data'))
        weights = unpack_float_array(model_params['mu_shape'], get_buffer('mu_data'))

        self.classifier = BOPRClassifier(initial_weights, initial_variance, beta)
        self.classifier.weights = weights
        self.classifier.variance = variance

        training_params = get_dict(self.DATA_STORE_KEY_TRAINING_DATA_VALUES)

        self.normal_behaviour_count = training_params['label_0_count']
        self.good_behaviour_count = training_params['label_1_count']
        self.unused_sample_count = training_params['unused_sample_count']
        self.trained_days = training_params["completed_iods"]
        self.features = training_params.get("features", DEFAULT_FEATURES)

        self.auc_scores = get_dict(self.DATA_STORE_KEY_AUC_SCORES)

    def serialise(self, data_store):
        """
        Given an instance of the data store, stores the model parameters

        :type data_store: xcm.stores.s3.S3XCMStore
        """
        # noinspection PyProtectedMember
        set_dict = partial(data_store._set_dict, self.id)
        # noinspection PyProtectedMember
        set_buffer = partial(data_store._set_buffer, self.id)

        initial_variance_shape, initial_variance_data = pack_float_array(self.classifier.initial_variance)
        initial_weights_shape, initial_weights_data = pack_float_array(self.classifier.initial_weights)
        variance_shape, variance_data = pack_float_array(self.classifier.variance)
        weights_shape, weights_data = pack_float_array(self.classifier.weights)

        bopr_fields = {
            "n_features": self.classifier.n_features, "beta": self.classifier.beta,
            "initial_sigma2_shape": initial_variance_shape, "initial_mu_shape": initial_weights_shape,
            "sigma2_shape": variance_shape, "mu_shape": weights_shape
        }
        set_dict(self.DATA_STORE_KEY_CLASSIFIER_PARAMS, bopr_fields)

        set_buffer('initial_sigma2_data', initial_variance_data)
        set_buffer('initial_mu_data', initial_weights_data)
        set_buffer('sigma2_data', variance_data)
        set_buffer('mu_data', weights_data)

        set_dict(self.DATA_STORE_KEY_TRAINING_DATA_VALUES, {
            "label_0_count": self.normal_behaviour_count,
            "label_1_count": self.good_behaviour_count,
            "unused_sample_count": self.unused_sample_count,
            "completed_iods": self.trained_days,
            "features": self.features
        })

        set_dict(self.DATA_STORE_KEY_AUC_SCORES, self.auc_scores)

    def predict(self, bidder_auction, exploration=0.):
        """
        Given an auction, make predictions about the likely response.

        :param dict bidder_auction: a bidder auction dictionary
        :param float exploration: perturbation factor between 0 and 1
        :rtype: tuple[float, float, float]
        :return: uncalibrated model score, calibrated CTR prediction, CTR boost over the training CTR
        """
        hashed_features = get_xcm_feature_hashes(bidder_auction, self.classifier.n_features, self.features)

        if not hashed_features.shape[0]:
            return 0., 0., 0.

        fitness = self.classifier.predict(hashed_features, exploration)

        ctr_estimate = fitness / (fitness + (1. - fitness) / self._get_downsampling_rate())
        training_set_ctr = self._get_training_set_ctr()
        ctr_boost = ctr_estimate / training_set_ctr

        return fitness, ctr_estimate, ctr_boost

    def partial_fit(self, data, labels):
        """
        Converts data to a list of hashed features and updates the encapsulated FFM model.

        :param list[dict] data: should contain a list of canonical auctions.
        :param list[int] labels: list of the labels corresponding to each element of data.
        """
        hashed_data = []
        valid_labels = []
        for i, record in enumerate(data):
            features = get_xcm_feature_hashes(record, self.classifier.n_features, self.features)
            if not features.shape[0]:
                continue

            hashed_data.append(features)
            valid_labels.append(labels[i])

        self.good_behaviour_count += valid_labels.count(1)
        self.normal_behaviour_count += valid_labels.count(0)

        oversample_rate = self._get_oversampling_rate()

        logging.info('Oversampling clicks by %s', int(0.1 / oversample_rate))
        for i in range(len(valid_labels)):
            if valid_labels[i] == 1:
                for j in range(int(0.1 / oversample_rate)):
                    valid_labels.append(1)
                    hashed_data.append(hashed_data[i])

        logging.info('Applying dataset with %s clicks and %s imps', valid_labels.count(1), valid_labels.count(0))

        for j in range(10):
            for i in range(0, len(hashed_data), 1000):
                self.forget()
                self.classifier.partial_fit(hashed_data[i:i + 1000], valid_labels[i:i + 1000])

    def forget(self):
        self.classifier.forget(0.001)

    def create_prediction_model(self, mnemonic, version=None):
        """
        Create a instance of an XCMModel for use in prediction scenarios.

        :type mnemonic: str
        :type version: str|None
        :rtype: XCMModel
        """
        return XCMModel(
            mnemonic, version, self.classifier.weights, self.classifier.variance, self.classifier.beta,
            self._get_downsampling_rate(), self._get_training_set_ctr()
        )

    def _get_training_set_ctr(self):
        """
        Return the measured CTR in the unsampled data used to run.
        """
        record_count = self.good_behaviour_count + self.normal_behaviour_count + self.unused_sample_count
        num = self.good_behaviour_count + 1.
        denom = record_count + self._get_downsampling_rate()
        return float(num) / denom

    def _get_downsampling_rate(self):
        """
        Return the measured downsampling rate.

        NOTE: this assumes only the impressions are downsampled. IF this is ever different, then calculation of
        the calibration correction would need to change.
        """
        if self.normal_behaviour_count + self.unused_sample_count == 0:
            return 1.

        return float(self.normal_behaviour_count) / float(self.normal_behaviour_count + self.unused_sample_count)

    def _get_oversampling_rate(self):
        """Return the measured oversampling rate for good behaviour"""
        inv_downsampling_rate = 1 / self._get_downsampling_rate()

        num = self.good_behaviour_count + 1
        denom = self.good_behaviour_count + self.normal_behaviour_count + inv_downsampling_rate

        return float(num) / denom

    def train(self, log_reader, end_day, auction_filter=None, downsampling_rate=0.02, pagination=10000):
        """
        Train the model
        :param xcm.core.base_classes.XCMReader log_reader: a reader passing labelled data
        :param callable|None auction_filter: a function to restrict the dataset
        :param float downsampling_rate: the decimal fraction of baseline records to process
        :param int end_day: end day to train (incl)
        :param int pagination: the amount of records to train in one go
        """
        training_data = []
        labels = []

        for xcm_date in range(max(self.trained_days) + 1, end_day + 1):
            for record_index, (label, record) in enumerate(
                    log_reader.get_labelled_data(xcmd_dt(xcm_date), auction_filter, downsampling_rate), start=1):

                labels.append(label)
                training_data.append(record)

                if not record_index % pagination:
                    self.partial_fit(training_data, labels)
                    training_data = []
                    labels = []

            self.trained_days.append(xcm_date)

        if training_data:
            self.partial_fit(training_data, labels)
