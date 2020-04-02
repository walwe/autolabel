from pathlib import Path
from typing import Dict, List, Union, Optional

import numpy as np
import torch
import torchvision.models as models
from torch.nn import Softmax

from ..image import ImageListDataset
from .resnetlabels import LabelDecoder
from .base import Classifier


class _ResnetClassifier(Classifier):

    NAME = None
    _MODEL_CLS = None

    def __init__(self, device: Optional[torch.device] = None):
        """
        :param device: pytorch device, e.g. torch.device('cpu').
        """
        self._device = device
        self._model = self.__class__._MODEL_CLS(True, False)
        self._model.eval()  # No training today
        self._model.to(self.device)

    @property
    def device(self) -> torch.device:
        """
        Device used for computation
        :return: torch.device
        """
        if self._device:
            return self._device
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def predict(self, dataset: ImageListDataset, decode: bool = True, top=5) -> Dict[Path, List[Union[str, int]]]:
        """
        Predict given dataset
        :param dataset: Dataset
        :param decode: If True (default) string labels are returned instead of class number
        :param top: Number of results being returned
        :return: Dictionary of labels
        """
        preds = []
        for d in dataset.loader():
            d = d.to(self.device)
            preds.append(
                Softmax(dim=1)(self._model(d)).data.cpu().numpy()
            )
            del d
        if not decode:
            return dict(zip(dataset.paths, np.concatenate(preds)[-top:]))
        return dict(zip(dataset.paths, self.decode(np.concatenate(preds), top)))

    def decode(self, preds, top=3):
        """
        Decode given image label id to string
        :param preds: List of label ids
        :param top: How many result to return
        :return:
        """
        return LabelDecoder.decode(preds, top=top)


class Resnet18Classifier(_ResnetClassifier):
    NAME = 'resnet18'
    _MODEL_CLS = models.resnet18


class Resnet34Classifier(_ResnetClassifier):
    NAME = 'resnet34'
    _MODEL_CLS = models.resnet34


class Resnet50Classifier(_ResnetClassifier):
    NAME = 'resnet50'
    _MODEL_CLS = models.resnet50


class Resnet101Classifier(_ResnetClassifier):
    NAME = 'resnet101'
    _MODEL_CLS = models.resnet101


class Resnet152Classifier(_ResnetClassifier):
    NAME = 'resnet152'
    _MODEL_CLS = models.resnet152


classifiers = _ResnetClassifier.__subclasses__()
