#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functions for interacting with byte packed numpy arrays
"""
from __future__ import unicode_literals

import numpy
import struct


def pack_float_array(ar):
    """
    Packs a numpy float array into a byte structure, returning the shape and packed data.

    :param numpy.ndarray ar: Array of float data.
    """

    total_size = numpy.product(ar.shape)

    packed_data = struct.pack('<%sf' % total_size, *ar.ravel())
    return ar.shape, packed_data


def unpack_float_array(shape, packed_data):
    """
    Unpacks byte packed data into a numpy float array of specified shape.
    """
    ar = numpy.fromstring(packed_data, dtype='<f4')
    ar.shape = shape
    return ar


def pack_int_array(ar):
    """
    Packs a numpy int array into a byte structure, returning the shape and packed data.

    :param numpy.ndarray ar: Array of int data.
    """

    total_size = numpy.product(ar.shape)

    packed_data = struct.pack('<%si' % total_size, *ar.ravel())
    return ar.shape, packed_data


def unpack_int_array(shape, packed_data):
    """
    Unpacks byte packed data into a numpy int array of specified shape.
    """
    ar = numpy.fromstring(packed_data, dtype='<i4')
    ar.shape = shape
    return ar
