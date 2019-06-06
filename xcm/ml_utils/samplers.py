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

        self.index = 0
        self.used = 0

    def get(self):
        if self.random_sample is None or self.used == self.sample_usages:
            self.random_sample = numpy.random.randn(self.sample_size)
            self.used = 0

        sample = self.random_sample[self.index]
        self.index += 1
        if self.index == self.sample_size:
            self.index = 0
            self.used += 1

        return sample
