import torch
import torch.nn as nn
import torch.nn.functional as F

from pdb import set_trace

class TwoD(nn.Module):
    '''
    Super deep network with many convolution layers. MUST RUN ON M40 GPU!
    '''
    def __init__(self, **kwargs):
        super().__init__()
        num_classes = kwargs.get("num_classes", 3)
        num_channels = kwargs.get("num_channels", 1)
        
        self.conv1 = nn.Conv2d(num_channels, 6, 3)
        self.conv2 = nn.Conv2d(6, 9, 3)
        self.conv3 = nn.Conv2d(9, 12, 3)
        self.conv4 = nn.Conv2d(12, 16, 3)

        self.pool1 = nn.MaxPool2d(2, 2)

        self.fwd1 = nn.Linear(11664, 3024)
        self.fwd2 = nn.Linear(3024, 666)
        self.fwd3 = nn.Linear(666,96)
        self.fwd4 = nn.Linear(96, 16)
        self.fwd5 = nn.Linear(16, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool1(x)
        x = F.relu(self.conv4(x))
        x = self.pool1(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fwd1(x))
        x = F.relu(self.fwd2(x))
        x = F.relu(self.fwd3(x))
        x = F.relu(self.fwd4(x))
        x = self.fwd5(x)
        return x

    def loss(self, x, y):
        return F.cross_entropy(x, y)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
