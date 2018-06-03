import torch
from torchlite.torch.metrics import Metric


def naive_loss(distances, target):
    """

    :param distances: 1d Tensor shape: (num_examples, )
    :param target: 1d Tensor shape: (num_examples, )
    """

    diff = torch.abs(distances - target)
    return torch.sum(diff)


class DistAccuracy(Metric):

    def __init__(self, alpha=0.4):
        self.alpha = alpha

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
