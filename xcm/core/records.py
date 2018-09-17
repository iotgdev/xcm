#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
record classes to manage connections with XCM
"""
from __future__ import unicode_literals

from collections import defaultdict

from xcm.canonical.xcm_time import ts_dt


class XCMRecord(object):
    """
    Parents for PVC and ASC allow us to include parents in the XCM record
    """
    def __call__(self, *args, **kwargs):
        return self.as_dict(*args, **kwargs)

    @staticmethod
    def as_dict(record, features):
        """
        Convert XCM standard fields into an XCM Record

        :param dict record: an iotec Auction Canonical record
        :param list features: the features to return in the XCM record
        :return: An XCM Record
        :rtype: dict
        """

        record = defaultdict(lambda: "0", **record)

        local_timestamp = record['AuctionRecTimestamp'] + record['TimezoneOffset'] * 60.

        xcm_dict = {
            'CoarseURL': record['CoarseURL'],
            'AppId': record['AppId'],
            'AnonymousDomainId': record['AnonymousDomainId'],
            'IsAdInApp': record['IsAdInApp'],
            'IsInStreamVideo': record['IsInStreamVideo'],
            'IsInterstitial': record['IsInterstitial'],
            'DeviceType': record['DeviceType'],
            'SlotVisibility': record['SlotVisibility'],
            'SlotViewability': record['SlotViewability'],
            'HalfHourIndex': XCMRecord.half_hour(local_timestamp),
            'WeekDay': str(ts_dt(local_timestamp).isoweekday()),
            'GeoState': record['GeoState'],
            'GeoCity': record['GeoCity'],
            'Browser': record['Browser'],
            'BrowserVersion': record['BrowserVersion'],
            'OS': record['OS'],
            'OSVersion': record['OSVersion'],
            'AdHandle': record['AdHandle'],
            'AdHeight': record['AdHeight'],
            'AdWidth': record['AdWidth'],
            'AdvertiserId': record['AdvertiserId'],
        }

        parent_categories = {c.split('-')[0] for c in record['PageVerticalCategories']}
        xcm_dict['PageVerticalCategories'] = [
            'PANEX-%s' % vert for vert in parent_categories.union(record['PageVerticalCategories'])]

        int_half_hour_index = int(xcm_dict['HalfHourIndex'])
        xcm_dict['SmoothTime'] = [str(t % 48) for t in range(int_half_hour_index - 2, int_half_hour_index + 3)]

        xcm_dict['AdDimensions'] = XCMRecord.ad_dimensions(xcm_dict['AdWidth'], xcm_dict['AdHeight'])

        return {k: v for k, v in xcm_dict.items() if k in features}

    @staticmethod
    def half_hour(timestamp):
        """
        Return a timestamp bucketed into the half hour into the day (0-47)
        :type timestamp: str|unicode|int|float
        :rtype: str
        """
        formatted_ts = int(float(timestamp))

        half_hour_index = (formatted_ts % 86400) / 1800
        return str(half_hour_index)

    @staticmethod
    def ad_dimensions(ad_width, ad_height):
        """Returns the ad dimensions from the ad width and height"""
        return '{}x{}'.format(ad_width, ad_height)
