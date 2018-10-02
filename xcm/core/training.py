"""
Module contains implementation of the training procedure for XCM.

Data resources we're going to be working with:
- XCM baseline models (stored in the ACE Training Model Repository)
- XCM 'current' models (stored in the ACE Training Model Repository)
- BeeswaxLog
- Prediction model repository
- Settings

"""
from __future__ import unicode_literals

import ujson

import mmh3

from xcm.canonical.xcm_time import xcmd, xcmd_dt

XCM_CURRENT_NAME = 'XCMDefault'


def build_xcm_model(model_store, log_reader, model_name=XCM_CURRENT_NAME, features=None):
    """
    Builds an XCM model
    Uses a baseline model to _train up to 6 days ago, then a current model trained until yesterday
    moves the current model to a new store for access/optimisation as an active model
    """

    _train_baseline_model(model_store, log_reader, model_name, features)
    _train_model(model_store, log_reader, model_name, features)


def _train_baseline_model(model_store, log_reader, model_name, features=None):
    """
    Trains a baseline model
    baseline models:
     - Have no auction filter
     - Have been trained for some days prior to "creation"
     - Atr trained up to 7 days ago to avoid noise
    :param xcm.stores.s3.S3XCMStore model_store: the datastore to interface with models
    :param xcm.core.base_classes.XCMReader log_reader: the log reader to provide data for.
    :param list features: the features to use to build the model
    :type model_name: str|unicode
    """
    today = xcmd()
    end_date = today - 7

    model_id = model_store.list(model_name + 'Baseline')[0]
    model = model_store.retrieve(model_id)
    """:type model: xcm.core.models.XCMTrainingModel"""
    if features:
        model.features = features

    next_training_day = max(model.trained_days) + 1
    for training_day in range(next_training_day, end_date + 1):
        model.classifier.forget(0.001)

    _train(model, log_reader, end_date, auction_filter=lambda x: mmh3.hash(ujson.dumps(x, sort_keys=True)) % 20 > 0)
    model_store.create(model)  # a new version is saved


def _train_model(model_store, log_reader, model_name, features=None):
    """
    Trains a current model
    current models:
     - have an auction filter of 95% so that the remaining 5% can be reserved for evaluation
     - are built from baseline models
     - create active models
     - trained up to yesterday given that the dataset is complete
    :type model_store: xcm.stores.s3.S3XCMStore
    :type log_reader: xcm.core.base_classes.XCMReader
    :type model_name: str|unicode
    :type features: list
    """
    today = xcmd()
    end_date = today - 1

    model_id = model_store.list(model_name + 'Baseline')[0]  # most recent baseline
    model = model_store.retrieve(model_id)
    """:type model: xcm.core.models.XCMTrainingModel"""
    model.name = model_name
    if features:
        model.features = features

    _train(model, log_reader, end_date, auction_filter=lambda x: mmh3.hash(ujson.dumps(x, sort_keys=True)) % 20 == 0)

    model_store.update(model)  # same version as the baseline model


def _train(model, log_reader, end_day, auction_filter=None, downsampling_rate=0.02, pagination=10000):
    """
    Train the model
    :param xcm.core.models.XCMTrainingModel model: an XCM model to _train
    :param xcm.core.base_classes.XCMReader log_reader: a reader passing labelled data
    :param callable|None auction_filter: a function to restrict the dataset
    :param float downsampling_rate: the decimal fraction of baseline records to process
    :param int end_day: end day to _train (incl)
    :param int pagination: the amount of records to _train in one go
    """
    training_data = []
    labels = []

    for xcm_date in range(max(model.trained_days) + 1, end_day + 1):
        for record_index, (label, record) in enumerate(
                log_reader.get_labelled_data(xcmd_dt(xcm_date), downsampling_rate), start=1):

            if auction_filter and auction_filter(record):
                labels.append(label)
                training_data.append(record)

                if not record_index % pagination:
                    model.partial_fit(training_data, labels)
                    training_data = []
                    labels = []

        model.trained_days.append(xcm_date)

    if training_data:
        model.partial_fit(training_data, labels)
