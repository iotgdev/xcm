#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class for taking a random sample from normal data
"""
import numpy


class Sampler(object):

    def __init__(self, sample_size, sample_usages):
        self.sample_size = sample_size
        self.sample_usages = sample_usages

        self.random_sample = None

    def get(self):
        index = 0
        used = 0

        while True:
            if self.random_sample is None or used == self.sample_usages:
                self.random_sample = numpy.random.randn(self.sample_size)
                used = 0

            sample = self.random_sample[index]
            index += 1
            if index == self.sample_size:
                index = 0
                used += 1

            yield sample
