"""
Defines function for evaluating RO curves and AUROC scores
"""
from __future__ import unicode_literals

import warnings
from collections import defaultdict

import numpy as np

from xcm.canonical.xcm_time import xcmd_dt

warnings.simplefilter('always', DeprecationWarning)
warnings.warn('This module has been deprecated from the XCM package', DeprecationWarning)


def roc_curve(labels, scores):  # todo: remove from XCM
    """
    Given a list of labels and the corresponding scores, returns the points of the
    receiver operating characteristic (ROC) curve for the ranking defined by the scores

    :param np.array labels: the labels of the data
    :param np.array scores: the scores defining the ranking under testing
    :rtype: tuple(np.array, np.array, np.array)
    """
    if len(scores.shape) != 1:
        raise ValueError('Wrong shape for predicted scores array {}'.format(scores.shape))

    sorted_indices = np.argsort(scores)

    scores = np.array(scores)[sorted_indices]
    labels = np.array(labels)[sorted_indices]

    n_samples = len(scores)
    n_clk = sum(labels)
    n_imp = n_samples - n_clk

    fpr = [1]
    tpr = [1]
    thresholds = [scores[0]]

    current_fpr = 1
    current_tpr = 1
    current_label = labels[0]

    tpr_dec = 1/float(n_clk)
    fpr_dec = 1/float(n_imp)

    for index, label in enumerate(labels):
        if label != current_label:
            current_label = label
            fpr.append(current_fpr)
            tpr.append(current_tpr)
            thresholds.append(scores[index])

        if label == 1:
            current_tpr -= tpr_dec
        else:
            current_fpr -= fpr_dec

    return np.array(fpr), np.array(tpr), np.array(thresholds)


def auc(fpr, tpr):  # todo: remove from XCM
    """
    Given a receiver operating characteristic (ROC) curve returns the value of the area under
    the ROC (AUROC)

    :param np.array fpr: x coordinates of the ROC
    :param np.array tpr: y coordinates of the ROC
    :rtype: float
    """
    bins = -(fpr[1:] - fpr[:-1])
    heights = (tpr[1:] + tpr[:-1]) / 2
    return np.dot(bins, heights)


class ModelEvaluator(object):  # todo: remove from XCM

    def evaluate_models(self, models, log_reader, filter_function, label_function, start_iod, end_iod):
        """
        Runs AUC scoring using the holdout data on the baseline and current models.

        :param list[machine_learning.models.xcm.xcm_training_model.XCMTrainingModel] models: list of models to evaluate
        :param BeeswaxWinLogReader log_reader: place to get the data for evaluation
        :param callable filter_function: a filter function to apply (lambda x: x % 20 recommended)
        :param callable label_function: a function returning a label (lambda x: 1 if x['clicks'] else 0)
        :type start_iod: int
        :type end_iod: int
        """

        # First, create the scores for each model, organised by day:
        scores = defaultdict(list)
        labels = defaultdict(list)

        for iod in range(start_iod, end_iod + 1):
            for auction in log_reader.get_ml_features_dict(xcmd_dt(iod)):
                if filter_function and filter_function(auction):
                    for model in models:
                        scores[model.mnemonic, iod].append(model.predict(auction, {})[0])
                    labels[iod].append(label_function(auction))

        # We score the baseline and updated model...
        for model in models:
            # ... on two different time windows
            for offsets in [(0, -2), (1, -1)]:
                start_iod += offsets[0]
                end_iod += offsets[1]

                model.auc_scores["XCM Day {0} - {1}".format(start_iod, end_iod)] = \
                    self._get_auc_score(model.mnemonic, start_iod, end_iod, scores, labels)

        # TODO: think about return types here

    @staticmethod
    def _get_auc_score(mnemonic, start_iod, end_iod, scores_dict, labels_dict):
        """
        Get the AUC score for a range of days given scores and labels for those days.

        :type mnemonic: str
        :type start_iod: int
        :type end_iod: int
        :param dict[tuple, list[float]] scores_dict: a mapping from IO days to lists of classification scores
        :param dict[int, list[int|float]] labels_dict: a mapping from IO days to lists of auction labels (0/1)
        :rtype: int
        """
        scores = []
        labels = []
        for iod in range(start_iod, end_iod + 1):
            if (mnemonic, iod) in scores_dict:
                scores.extend(scores_dict[mnemonic, iod])
                labels.extend(labels_dict[iod])

        # We check that we have both 0 and 1 labels otherwise the
        # AUC score is not defined
        if labels and labels.count(0) and labels.count(1):
            fpr, tpr, th = roc_curve(labels, np.array(scores))
            return auc(fpr, tpr)
        else:
            return 0
