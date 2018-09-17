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

import mmh3

from xcm.canonical.xcm_time import xcmd


XCM_BASELINE_NAME = 'XCMBaseline'
XCM_CURRENT_NAME = 'XCMCurrent'


def build_xcm_model(training_store, active_store, log_reader):
    """
    Builds an XCM model
    Uses a baseline model to train up to 6 days ago, then a current model trained until yesterday
    moves the current model to a new store for access/optimisation as an active model
    """

    train_baseline_model(training_store, log_reader)
    train_current_model(training_store, log_reader)

    generate_active_model(training_store, active_store)


def train_baseline_model(training_store, log_reader):
    """
    Trains a baseline model
    baseline models:
     - Have no auction filter
     - Have been trained for some days prior to "creation"
     - Atr trained up to 7 days ago to avoid noise
    :param xcm.stores.s3.S3XCMStore training_store: the datastore to interface with models
    :param xcm.core.base_classes.XCMReader log_reader: the log reader to provide data for.
    """
    today = xcmd()
    end_date = today - 7

    baseline_id = training_store.list(XCM_BASELINE_NAME)[0]  # most recent baseline
    baseline_model = training_store.retrieve(baseline_id)
    """:type baseline_model: xcm.core.models.XCMTrainingModel"""

    next_training_day = max(baseline_model.trained_days) + 1
    for training_day in range(next_training_day, end_date + 1):
        baseline_model.forget()

    baseline_model.train(log_reader, end_date)
    training_store.create(baseline_model)  # a new version is saved


def train_current_model(training_store, log_reader):
    """
    Trains a current model
    current models:
     - have an auction filter of 95% so that the remaining 5% can be reserved for evaluation
     - are built from baseline models
     - create active models
     - trained up to yesterday given that the dataset is complete
    :type training_store: xcm.stores.s3.S3XCMStore
    :type log_reader:
    """
    today = xcmd()
    end_date = today - 1

    baseline_id = training_store.list(XCM_BASELINE_NAME)[0]  # most recent baseline
    current_model = training_store.retrieve(baseline_id)
    """:type current_model: xcm.core.models.XCMTrainingModel"""
    current_model.name = XCM_CURRENT_NAME

    record_filter = lambda x: mmh3.hash(x['AuctionId']) % 20 > 0
    current_model.train(log_reader, end_date, auction_filter=record_filter)

    training_store.update(current_model)  # same version as the baseline model


def generate_active_model(training_store, active_store):
    """
    generates an active model (optimised to evaluate, not train)
    :param training_store:
    :param xcm.stores.s3.S3XCMStore active_store:
    """
    old_active_model = active_store.list(XCM_CURRENT_NAME)[0]  # most recent active model

    current_id = training_store.list(XCM_CURRENT_NAME)[0]  # most recent current model
    current_model = training_store.retrieve(current_id)
    active_model = current_model.create_prediction_model(XCM_CURRENT_NAME, current_model.version)

    active_store.update(active_model)
    active_store.delete(old_active_model)
