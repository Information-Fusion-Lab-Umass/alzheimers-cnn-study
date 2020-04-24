import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor
import torch

class JainNet(nn.Module):
    def __init__(self, output_dim: int = 3):
        super().__init__()
        self.model = models.vgg16(pretrained=True)
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=256, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=output_dim, bias=True))
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, images):
        return self.model(images)

    def compute_logit_loss(self, input, targets):
        images, targets = input[0].to(self.device), targets.to(self.device)
        outputs = self.forward(images)
        loss = self.criterion(outputs, targets)
        return outputs, loss
