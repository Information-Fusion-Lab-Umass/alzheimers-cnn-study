import torch
import torch.nn as nn
import torch.nn.functional as F


class Classify(nn.Module):
    '''
    A basic CNN with a three-layer encoder.
    '''
    def __init__(self, **kwargs):
        super().__init__()
        num_kernels = kwargs.get("num_kernels", [16, 32, 64])

        self.cn1 = nn.Conv3d(1, num_kernels[0],
                             kernel_size=7, stride=3, padding=0)
        nn.init.kaiming_normal_(self.cn1.weight)
        self.bn1 = nn.BatchNorm3d(num_kernels[0])

        self.cn2 = nn.Conv3d(num_kernels[0], num_kernels[1],
                             kernel_size=3, stride=3, padding=0)
        nn.init.kaiming_normal_(self.cn2.weight)
        self.pool2 = nn.MaxPool3d(2,1)
        self.bn2 = nn.BatchNorm3d(num_kernels[1])

        self.cn3 = nn.Conv3d(num_kernels[1], num_kernels[2],
                             kernel_size=3, stride=3, padding=1)
        nn.init.kaiming_normal_(self.cn3.weight)
        self.bn3 = nn.BatchNorm3d(num_kernels[2])

        self.cn4 = nn.Conv3d(num_kernels[2], 2*num_kernels[2],
                             kernel_size=3, stride=3, padding=1)
        nn.init.kaiming_normal_(self.cn4.weight)
        self.pool4 = nn.MaxPool3d(2,2)
        self.bn4 = nn.BatchNorm3d(2*num_kernels[2])

        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 3)

    def forward(self, x):
        l1 = F.relu(self.bn1(self.cn1(x)))
        l2 = self.pool2(F.relu(self.bn2(self.cn2(l1))))
        l3 = F.relu(self.bn3(self.cn3(l2)))
        l4 = self.bn4(self.cn4(l3))
        #print(l4.shape)
        l4 = self.pool4(F.relu(l4))
        #print(l4.shape)
        l4 = l4.view(-1, 128)
        #print(l4.shape)
        l5 = F.relu(self.fc1(l4))
        l6 = F.relu(self.fc2(l5))
        l7 = self.fc3(l6)
        return l7

    def loss(self, x, y):
        return F.cross_entropy(x, y)
