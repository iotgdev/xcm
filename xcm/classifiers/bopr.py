"""
Implementation of the BOPR training algorithm.

Bayesian Online Probit Regression is a linear model which supports bayesian beliefs over the model weights.
"""
from __future__ import unicode_literals


try:
    from itertools import izip
except ImportError:  # 2to3
    izip = zip

from math import sqrt, exp, erf

import numpy

from xcm.core.base_classes import XCMClassifier
from xcm.utilities.sampler import Sampler

DEFAULT_HASH_SIZE = 1000000
DEFAULT_INITIAL_VARIANCE = 5
MIN_X = -6.
MAX_X = 6.
SQRT2PI = sqrt(2 * numpy.pi)
SQRT2 = sqrt(2)


def is_array(val):
    """Check if an object is a numpy n-dimentional array"""
    return isinstance(val, numpy.ndarray)


def probit(x):
    """
    The standard probit function (cumulative distribution of a standard normal function)

    :param float x: the input for the probit function
    """
    return (1 + erf(x / SQRT2)) / 2.0


def gaus_probit_ratio(x):
    """
    This function returns the ratio of the gaussian and probit functions.
    For small inputs the limit of the ratio is returned in order to avoid
    numerical instabilities.

    :type x: float
    :rtype: float
    :return: the ratio between the gaussian and probit functions
    """
    if x < MIN_X:
        return -x
    elif x > MAX_X:
        return 0.
    else:
        return exp(-x ** 2 / 2.) / SQRT2PI / probit(x)


def weighting(x):
    """
    This function is defined in terms of v(x). Extra care is needed for low or high values
    of x due to numerical instabilities, in these cases will return the asymptotic
    value

    :type x: float
    :rtype: float
    :return: v(x)*(v(x) + x)
    """
    if x < MIN_X:
        return 1.
    elif x > MAX_X:
        return 0.
    else:
        vx = gaus_probit_ratio(x)
        return vx * (vx + x)


class BOPRClassifier(XCMClassifier):
    """
    A bayesian online probit regression implementation as seen on

    Grapael, Candela, Borchert, Herbrich
    "Web-Scale Bayesian Click-Through Rate Prediction for Sponsored Search
    Advertising in Microsoft's Bing Search Engine"
    """

    def __init__(self, weights=None, variances=None, beta=0.1):
        """
        Initialise the BOPR fields.

        :type weights: numpy.array
        :type variances: numpy.array
        :type beta: float
        """
        self.n_features = weights.shape[0] if is_array(weights) else DEFAULT_HASH_SIZE
        self.beta = beta

        self.initial_variance = variances if is_array(variances) else \
            numpy.ones(self.n_features) * DEFAULT_INITIAL_VARIANCE
        self.initial_weights = weights if is_array(weights) else numpy.zeros(self.n_features)

        self.weights = self.initial_weights.copy()
        self.variance = self.initial_variance.copy()

        self.sampler = Sampler(10000, 10000)

    def forget(self, eps):
        """
        This method implements a partial reset of the learned values (mu and sigma) back to
        the initialization values. Repeated execution of this method without observing new data
        eventually takes the classifier back to the original state. This process is faster for
        bigger values if eps.

        :param float eps: a float in the range (0, 1) measuring the strength of the reset,
        0 for no reset and 1 for a full reset
        """
        if not (0 <= eps <= 1):
            raise RuntimeError('Invalid forget quantity, must be between 0 and 1')

        noisy_variance = numpy.multiply(self.variance, self.initial_variance) / (
            (1 - eps) * self.initial_variance + eps * self.variance)
        noisy_weights = numpy.multiply(noisy_variance, (1 - eps) * numpy.divide(
            self.weights, self.variance) + eps * numpy.divide(self.initial_weights, self.initial_variance))

        self.weights = noisy_weights
        self.variance = noisy_variance

    def predict(self, hash_vector, exploration=0.0):
        """
        Returns the class probability for the given sample

        :param list[int] hash_vector: a list of nonzero indices in the binary representation of a sample
        :param float exploration: a quantity to
        :rtype: float
        :return: the class probability for the sample
        """
        hash_vector = numpy.unique(hash_vector)

        nonzero_variance_sum = numpy.take(self.variance, hash_vector).sum()

        std_deviation = sqrt(self.beta ** 2 + nonzero_variance_sum)
        nonzero_weights_sum = numpy.take(self.weights, hash_vector).sum()

        if exploration > 0:
            nonzero_weights_sum += self.sampler.get() * std_deviation * min(exploration, 1.0)

        return probit(nonzero_weights_sum / std_deviation)

    def partial_fit(self, hash_vectors, labels):
        """
        Iterates through a list of samples performs the learning procedure

        :param list[list[int]] hash_vectors: a list of hash vectors from the samples
        :param list[int] labels: the labels corresponding to the samples
        """
        if len(hash_vectors) != len(labels):
            raise RuntimeError(
                'Number of data points %s does not match the number of labels %s' % (len(hash_vectors), len(labels)))

        for label, nonzero_indices in izip(labels, hash_vectors):

            nonzero_indices = numpy.unique(nonzero_indices)  # removes collision issues
            trial_score = 1 if label else -1

            nonzero_variance = numpy.take(self.variance, nonzero_indices)
            nonzero_weights = numpy.take(self.weights, nonzero_indices)

            nonzero_variance_sum = nonzero_variance.sum()
            std_deviation = sqrt(self.beta ** 2 + nonzero_variance_sum)
            score = trial_score * nonzero_weights.sum() / std_deviation

            new_weights = nonzero_weights + trial_score * nonzero_variance / std_deviation * gaus_probit_ratio(score)
            new_variance = numpy.multiply(
                nonzero_variance, 1 - nonzero_variance / std_deviation ** 2 * weighting(score))

            self.weights[nonzero_indices] = new_weights
            self.variance[nonzero_indices] = new_variance
