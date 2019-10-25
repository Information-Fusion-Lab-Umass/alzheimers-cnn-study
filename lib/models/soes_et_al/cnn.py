from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor

from lib.models.model import Model


class CNN(Model):
    def __init__(self, output_dim: int = 3):
        super().__init__()
        num_classes = kwargs.get("num_classes", 3)
        num_channels = kwargs.get("num_channels", 1)
        cnn_dropout = kwargs.get("cnn_dropout", 0.1)
        class_dropout = kwargs.get("class_dropout", 0.4)

        # 3^3 x 32 filters
        self.conv1 = nn.Conv3d(num_channels, 32, kernel_size=3, stride=2)
        self.pool1 = nn.MaxPool3d(2)

        # 3^3 x 64 filters
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool3d(3, padding=1)

        # 3^3 x 128 filters
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3)
        self.pool3 = nn.MaxPool3d(4, padding=1)

        self.cnn_dropout = nn.Dropout(cnn_dropout)
        self.classification_dropout = nn.Dropout(class_dropout)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        l1 = self.pool1(self.conv1(images))
        l2 = self.pool2(self.conv2(l1))
        l2 = self.cnn_dropout(l2)
        l3 = self.pool3(self.conv3(l2))
        flattened = l3.view(len(l3), -1)
        dropped = self.classification_dropout(flattened)
        return self.fc(dropped)

    def classification_loss(self, images: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        pred = self.forward(images)
        loss = F.cross_entropy(pred, labels)
        return loss, pred
