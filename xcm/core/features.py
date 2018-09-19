#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions for XCM model training
"""
from __future__ import unicode_literals

import datetime
import logging
import mmh3

import numpy
from six import iteritems

from xcm.core.records import XCMRecord

logger = logging.getLogger('xcm.features')


DEFAULT_FEATURES = ['CoarseURL', 'AppId', 'AnonymousDomainId', 'IsAdInApp', 'IsInStreamVideo', 'IsInterstitial',
                    'DeviceType', 'SlotVisibility', 'SlotViewability', 'HalfHourIndex', 'WeekDay', 'GeoState',
                    'GeoCity', 'Browser', 'BrowserVersion', 'OS', 'OSVersion', 'AdHandle', 'AdHeight', 'AdWidth',
                    'AdvertiserId', 'PageVerticalCategories', 'SmoothTime', 'AdDimensions']


def get_next_version(version=None):
    """
    Given a version identifier, increment by a major version and return.
    NB that major versions are also bumped to today, where minor versions are not

    :param version: string version id
    :return: string version id
    """
    today = datetime.date.today()

    try:
        ver_parts = [int(part) for part in version.split('_')]
    except AttributeError:
        logging.warning('error when creating new version from {}'.format(version))
        ver_parts = [0, 0, 0]

    if ver_parts[0:3] == [today.year, today.month, today.day]:
        ver_parts[3] += 1
        ver_parts[4] = 0
    else:
        ver_parts = [today.year, today.month, today.day, 0, 0]

    return "{0:04}_{1:02}_{2:02}_{3:02}_{4:02}".format(*ver_parts)


def get_hashed_features(record, hash_size):
    """
    Returns the hashed features corresponding to a given XCM record dictionary.
    The values are expected to be strings or lists

    :param dict record: an XCM record dictionary
    :type hash_size: int
    :rtype: numpy.array
    """
    indices = []
    for key, value in iteritems(record):
        if isinstance(value, (list, set)):
            indices += [mmh3.hash("%s-%s" % (key, item)) for item in value]
        elif value:
            indices.append(mmh3.hash("%s-%s" % (key, value)))

    return numpy.array(indices) % hash_size


def get_xcm_feature_hashes(canon_auction, hash_size, features):
    """
    Given an auction in bidder format returns a list of hashed features

    :param dict canon_auction: an auction dictionary as per iotec auction canonical
    :type hash_size: int
    :param list features: a list of features to be used in the record for hashing
    :return: a list of hashed features
    :rtype: numpy.array
    """
    xcm_record = XCMRecord.as_dict(canon_auction, features)
    return numpy.unique(numpy.array(get_hashed_features(xcm_record, hash_size)))
