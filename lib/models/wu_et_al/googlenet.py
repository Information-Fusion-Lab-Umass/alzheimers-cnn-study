from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor

from lib.models.model import Model


class GoogleNet(Model):
    def __init__(self, output_dim: int = 3):
        super().__init__()
        """
        Due to the limited data set in this study, this technique was employed to learn the appropriate salient 
        features for MR-based imaging classification, where all CNN layers except for the last were fine-tuned with a 
        learning rate using 1/10 of the default learning rate. The last fully-connected layer was randomly initialized 
        and freshly trained, in order to accommodate the new object categories in this study. Its learning rate was set 
        to 1/100 of the default value.
        """
        self.model = models.googlenet(pretrained=True)
        self.model.fc = nn.Linear(1024, output_dim)

    def forward(self, images: Tensor) -> Tensor:
        return self.model(images)

    def classification_loss(self, images: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        pred = self.forward(images)
        loss = F.cross_entropy(pred, labels)

        return loss, pred
