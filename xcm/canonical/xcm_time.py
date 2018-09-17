# coding=utf-8
"""
Date and time module for XCM.
Concepts such as XCM hour, XCM day, and XCM week count since the start of XCM


The following contractions are used in this module :

    xcmw - xcm week
    xcmd - xcm day
    xcmh - io hour
    ds - datestring
    dt - datetime
    st - full datetime string
    ts - timestamp


The table below shows possible conversion from one type to another :

      *   | xcmw| xcmd| xcmh|  ts |  dt |  st |  ds |
    ------+-----+-----+-----+-----+-----+-----+-----+
     xcmw |  X  |  Y  |     |     |  Y  |     |     |
    ------+-----+-----+-----+-----+-----+-----+-----+
     xcmd |  Y  |  X  |  Y  |  Y  |  Y  |  Y  |  Y  |
    ------+-----+-----+-----+-----+-----+-----+-----+
     xcmh |     |  Y  |  X  |  Y  |  Y  |  Y  |  Y  |
    ------+-----+-----+-----+-----+-----+-----+-----+
      ts  |     |  Y  |  Y  |  X  |  Y  |  Y  |  Y  |
    ------+-----+-----+-----+-----+-----+-----+-----+
      dt  |  Y  |  Y  |  Y  |  Y  |  X  |  Y  |  Y  |
    ------+-----+-----+-----+-----+-----+-----+-----+
      st  |     |  Y  |  Y  |  Y  |  Y  |  X  |  Y  |
    ------+-----+-----+-----+-----+-----+-----+-----+
      ds  |     |  Y  |  Y  |  Y  |  Y  |  Y  |  X  |


There are functions to get the current time of the type desired :

    xcmw - current io week
    xcmd - current io day
    xcmh - current io hour
    ds - current date string  (YYYY/MM/DD (optional day_first=True to get DD/MM/YYYY)
    dt - current datetime object
    st - current time string  (YYYY-MM-DD HH:mm:ss.uuuuuu (u are microseconds))
    ts - current Unix epoch time (seconds)


**NOTE**
Datetimes generated in this module (either internally or returned)
  are timezone aware and use UTC as their time zone.
Naive datetimes (timezone unaware) will be assumed to be UTC and
  use that timezone information when doing conversions


Potential extenstions :
    add xcmm (XCM month)
    add xcmy (XCM year)
    add auto_conversion (automatic type detection of given input)
"""
from __future__ import unicode_literals

import datetime
import time

import pytz

XCM_EPOCH = 1367366400.0  # May 2013
MICROSEC_FACTOR = 10.0 ** 6
MICROSEC_LIMIT = XCM_EPOCH * MICROSEC_FACTOR
MICROSEC_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
STANDARD_FORMAT = '%Y-%m-%d %H:%M:%S'


def xcmh_st(xcmh):
    """Converts an XCM hour to a timestamp"""
    return ts_st(xcmh_ts(xcmh))


def xcmh_ds(xcmh, day_first=False):
    """Converts an XCM hour to a datestring, setting day_first to True returns a DD/MM/YYYY datestring"""
    return st_ds(xcmh_st(xcmh), day_first=day_first)


def xcmd_st(xcmd):
    """Converts an XCM day to a full date string"""
    return dt_st(xcmd_dt(xcmd))


def ts_st(ts):
    """Converts a timestamp to a full date string"""
    return dt_st(ts_dt(ts))


def ts_ds(ts, day_first=False):
    """Converts a timestamp to a datestring, setting day_first to True returns a DD/MM/YYYY datestring"""
    return st_ds(ts_st(ts), day_first=day_first)


def st_xcmd(st):
    """Converts a full date string to an XCM day"""
    return xcmh_xcmd(st_xcmh(st))


def ds_xcmh(ds):
    """Converts a datestring to an XCM hour"""
    return ts_xcmh(ds_ts(ds))


def ds_xcmd(ds):
    """Converts a datestring to an XCM day"""
    return ts_xcmd(ds_ts(ds))


def xcmd_ds(xcmd, day_first=False):
    """Convert an XCM day to a datestring, setting day_first to True returns a DD/MM/YYYY datestring"""
    return st_ds(dt_st(xcmd_dt(xcmd)), day_first=day_first)


def dt_ds(dt, day_first=False):
    """Converts a datetime object to a datestring, setting day_first to True returns a DD/MM/YYYY datestring"""
    return st_ds(dt_st(dt), day_first=day_first)


def st_ds(st, day_first=False):
    """Converts a full date string to a datestring, setting day_first to True returns a DD/MM/YYYY datestring"""
    if day_first:
        split_stds = st[:10].split('-')
        split_stds.reverse()
        return '/'.join(split_stds)
    else:
        return st[:10].replace('-', '/')


def ds_ts(ds):
    """Converts a datestring to a timestamp"""
    return dt_ts(ds_dt(ds))


def ds_dt(ds):
    """Converts a datestring to a datetime object"""
    return st_dt(ds_st(ds))


def ds_st(ds):
    """Converts a datestring to a full date string"""
    return ds.replace('/', '-') + ' 00:00:00'


def dt_st(dt):
    """Converts a datetime object to a full date string"""
    if dt.microsecond:
        return datetime.datetime.strftime(dt, MICROSEC_FORMAT)
    else:
        return datetime.datetime.strftime(dt, STANDARD_FORMAT)


def ts_xcmh(ts):
    """Converts a timestamp to an XCM hour"""
    if type(ts) == str or type(ts) == unicode:
        ts = float(ts)
    if ts > MICROSEC_LIMIT:
        ts /= MICROSEC_FACTOR
    return int((ts - XCM_EPOCH) / 3600)


def xcmh_ts(xcmhour):
    """Converts an XCM hour to a timestamp"""
    if type(xcmhour) == str or type(xcmhour) == unicode:
        xcmhour = float(xcmhour)
    return xcmhour * 3600 + XCM_EPOCH


def ts_dt(ts):
    """Converts a timestamp to a datetime"""
    if type(ts) == str or type(ts) == unicode:
        ts = float(ts)
    if ts > MICROSEC_LIMIT:
        ts /= MICROSEC_FACTOR

    return datetime.datetime.utcfromtimestamp(ts)


def dt_ts(dt):
    """Converts a datetime object to a timestamp"""
    if not dt.tzinfo:
        new_dt = dt.replace(tzinfo=pytz.utc)
    else:
        new_dt = dt

    return int((new_dt - datetime.datetime(1970, 1, 1, tzinfo=pytz.utc)).total_seconds()) + (dt.microsecond/MICROSEC_FACTOR if dt.microsecond else 0)


def dt_xcmh(dt):
    """Converts a datetime object to an XCM hour"""
    return ts_xcmh(dt_ts(dt))


def dt_xcmd(dt):
    """Converts a datetime object to an XCM day"""
    return xcmh_xcmd(dt_xcmh(dt))


def xcmh_dt(xcmhour):
    """Converts an XCM hour to a datetime object"""
    return ts_dt(xcmh_ts(xcmhour))


def st_dt(st):
    """Converts a full date string to an XCM day"""
    if '.' in st:
        return datetime.datetime.strptime(st, MICROSEC_FORMAT)
    else:
        return datetime.datetime.strptime(st, STANDARD_FORMAT)


def st_ts(st):
    """Converts a full date string to a timestamp"""
    return dt_ts(st_dt(st))


def st_xcmh(st):
    """Converts a full date string to an XCM hour"""
    return ts_xcmh(st_ts(st))


def xcmd_xcmh(xcmd):
    """Converts an XCM day to an XCM hour, giving the hour for midnight"""
    if type(xcmd) != int:
        xcmd = int(xcmd)
    return xcmd * 24


def xcmh_xcmd(xcmh):
    """Converts an XCM hour to an XCM day"""
    if type(xcmh) != int:
        xcmh = int(xcmh)
    return xcmh / 24


def xcmd_dt(xcmd):
    """Converts an XCM day to a datetime object"""
    return xcmh_dt(xcmd_xcmh(xcmd))


def ts_xcmd(ts):
    """Converts a timestamp to an XCM day"""
    return xcmh_xcmd(ts_xcmh(ts))


def xcmd_ts(xcmd):
    """Converts an XCM day to a timestamp, giving the timestamp for midnight"""
    return xcmh_ts(xcmd_xcmh(xcmd))


def xcmw_xcmd(xcmw):
    """
    Converts an io week into an io day, giving the monday for that week
    The -2 is because the XCM_EPOCH (start timestamp) is a Wednesday.
    XCM week 1 should return io day 5 (Monday)
    """
    if type(xcmw) != int:
        xcmw = int(xcmw)
    return (xcmw * 7) - 2


def xcmd_xcmw(xcmd):
    """
    Converts an io day to an io week.
    The +2 is because the XCM_EPOCH is a Wednesday.
    Io day 5 (Monday) should return io week 1
    """
    if type(xcmd) != int:
        xcmd = int(xcmd)
    return (xcmd + 2) / 7


def xcmw_dt(xcmw):
    """Converts an XCM week to a datetime object"""
    return xcmd_dt(xcmw_xcmd(xcmw))


def dt_xcmw(dt):
    """Converts a datetime object to an XCM week"""
    return xcmd_xcmw(dt_xcmd(dt))


def xcmh_xcmw(xcmh):
    """Converts an XCM hour to an XCM week"""
    return xcmd_xcmw(xcmh_xcmd(xcmh))


def xcmw_xcmh(xcmw):
    """Converts an XCM week to an XCM hour"""
    return xcmd_xcmh(xcmw_xcmd(xcmw))


def xcmw():
    """Returns the current XCM week"""
    return xcmd_xcmw(xcmd())


def xcmd():
    """Returns the current XCM day"""
    return xcmh_xcmd(ts_xcmh(time.time()))


def xcmh():
    """Return the current XCM hour"""
    return ts_xcmh(time.time())


def ts():
    """Returns the current timestamp"""
    return time.time()


def dt():
    """Returns the current datetime object"""
    return ts_dt(time.time())


def st():
    """Returns the current full date string"""
    return ts_st(time.time())


def ds():
    """Returns the current datestring"""
    return ts_ds(time.time())


def minute_adjust(minute, adj):
    """Adjusts minutes and returns a int in 0-59 range"""
    return (minute + adj) % 60


def hour_adjust(hour, adj):
    """Adjusts hours and returns a int in 0-23 range"""
    return (hour + adj) % 24


def ensure_xcm_day(day):
    """
    Convert a day to an xcm day.

    Intended to provide flexible date arg handling in scripts.

    :param str | datetime.datetime | datetime.date | int day: A date string (YYYY/MM/DD), an xcmday integer or a
    datetime.
    :return: The xcmday.
    :rtype: int

    """
    # convert dates to datetime
    if isinstance(day, datetime.date):
        day = datetime.datetime.fromordinal(day.toordinal())

    if isinstance(day, basestring) and '/' in day:
        xcm_day = ds_xcmd(day)
    elif isinstance(day, datetime.datetime):
        xcm_day = dt_xcmd(day)
    else:
        xcm_day = int(day)

    return xcm_day
