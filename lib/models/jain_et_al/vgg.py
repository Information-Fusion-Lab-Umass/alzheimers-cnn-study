from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor

from lib.models.model import Model


class VGG(Model):
    def __init__(self, output_dim: int = 3):
        super().__init__()
        self.model = models.vgg16(pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=256, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=output_dim, bias=True)
        )
        #self.model.fc = nn.Linear(1024, output_dim)

    def forward(self, images: Tensor) -> Tensor:
        return self.model(images)

    def classification_loss(self, images: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        pred = self.forward(images)
        loss = F.cross_entropy(pred, labels)
        return loss, pred
