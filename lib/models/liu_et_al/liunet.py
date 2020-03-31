from typing import Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor

#from lib.models.model import Model


class LiuNet(nn.Module):
    def __init__(self, expansion: int = 8, output_dim: int = 3):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv3d(in_channels=1,out_channels=4*expansion,kernel_size=1),
            nn.InstanceNorm3d(4*expansion),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(in_channels=4*expansion,out_channels=32*expansion,kernel_size=3, dilation=2),
            nn.InstanceNorm3d(32*expansion),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv3d(in_channels=32*expansion,out_channels=64*expansion,kernel_size=5, padding=2, dilation=2),
            nn.InstanceNorm3d(64*expansion),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=5, stride=2)
        )
        self.block4 = nn.Sequential(
            nn.Conv3d(in_channels=64*expansion,out_channels=64*expansion,kernel_size=3, padding=1, dilation=2),
            nn.InstanceNorm3d(64*expansion),
            nn.ReLU(inplace=True)
        )
        self.img_feature_encoder = nn.Linear(64*125*expansion, 1024)
        self.pe = torch.zeros(240, 512)
        position = torch.arange(0, 240).unsqueeze(1).float()
        div_term = torch.exp((torch.arange(0, 512, 2).float() *
                             -(math.log(10000.0) / 512)).float())
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.age_encoder = nn.Sequential(
            nn.Linear(512,512),
            nn.LayerNorm(512),
            nn.Linear(512,1024)
        )
        self.classifier = nn.Linear(2*1024,3)
        self.dropout = nn.Dropout(p=0.1)
        """
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, mean=0., std=0.01)

        for m in self.modules():
            m.apply(init_weights)
        """

    def forward(self, images: Tensor) -> Tensor:
        images, age = images
        b1 = self.block1(images)
        b2 = self.block2(b1)
        b3 = self.block3(b2)
        b4 = self.block4(b3)
        flattened = b4.view(len(images), -1)
        encoded_img = self.img_feature_encoder(flattened)
        age = self.pe[age.long(),:].cuda()
        print(age.size())
        print(encoded_img.size())
        encoded_age = self.age_encoder(age)
        print(encoded_age.size())
        encoded_img_age = torch.cat([encoded_img, encoded_age], 1)
        return self.classifier(self.dropout(encoded_img_age))

    def classification_loss(self, images: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        pred = self.forward(images)
        loss = F.cross_entropy(pred, labels)
        return loss, pred

model = LiuNet()
