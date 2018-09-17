#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions for stores
"""
from __future__ import unicode_literals

from xcm.core.models import XCMModel, XCMTrainingModel
from xcm.stores.s3 import S3XCMStore


_ACTIVE_STORE = None
_TRAINING_STORE = None


def get_active_model_store():
    global _ACTIVE_STORE
    if _ACTIVE_STORE is None:
        _ACTIVE_STORE = S3XCMStore('', '', XCMModel)
    return _ACTIVE_STORE


def get_training_model_store():
    global _TRAINING_STORE
    if _TRAINING_STORE is None:
        _TRAINING_STORE = S3XCMStore('', '', XCMTrainingModel)
    return _TRAINING_STORE
