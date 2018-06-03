import torch
from torchlite.torch.metrics import Metric


def naive_loss(distances, target):
    """

    :param distances: 1d Tensor shape: (num_examples, )
    :param target: 1d Tensor shape: (num_examples, )
    """

    diff = torch.abs(distances - target)
    return torch.sum(diff)


class TripletLoss:

    def __init__(self, alpha=0.4):
        self.alpha = alpha

    def __call__(self, distances, target):
        """

        :param distances: [ dist(base_1, pos_1), dist(base_1, neg_1), dist(base_2, pos_2), ... ]
        :param target: not used
        :return:
        """
        pairs = distances.view(-1, 2)
        positives = pairs[:, 0]
        negatives = pairs[:, 1]
        errors = positives - negatives + self.alpha
        return torch.sum(errors[errors > 0])


class TripletAccuracy(Metric):

    def __init__(self, alpha=0.4):
        self.alpha = alpha

    @property
    def get_name(self):
        return "triplet_accuracy"

    def __call__(self, distances, target):
        """

        :param distances: [ dist(base_1, pos_1), dist(base_1, neg_1), dist(base_2, pos_2), ... ]
        :param target: not used
        :return:
        """
        total = distances.shape[0] / 2
        pairs = distances.view(-1, 2)
        positives = pairs[:, 0]
        negatives = pairs[:, 1]
        errors = positives - negatives + self.alpha
        true_pred = (errors <= 0).int().sum()
        return true_pred / total


class DistAccuracy(Metric):

    @property
    def get_name(self):
        return "dist_accuracy"

    def __call__(self, y_pred, y_true):
        """

        :param y_pred: Distance between objects.
        :param y_true: ???
        :return:
        """

        positive = torch.sum((y_pred * y_true >= 0).int()).data.item()
        total = y_true.shape[0]
        return positive / total
