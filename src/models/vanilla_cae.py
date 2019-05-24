import torch
import torch.nn as nn
import torch.nn.functional as F

from pdb import set_trace

class VanillaCAE(nn.Module):
    '''
    A basic CNN with a three-layer encoder and three-layer decoder.
    '''
    def __init__(self, **kwargs):
        super().__init__()
        num_kernels = kwargs.get("num_kernels", [16, 32, 64])

        self.encoder = VanillaEncoder(num_kernels)
        self.decoder = VanillaDecoder(num_kernels)

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

    def loss(self, x, y):
        return F.mse_loss(x, y)

class VanillaEncoder(nn.Module):
    '''
    The encoder portion of the VanillaCAE model.
    '''
    def __init__(self, num_kernels):
        super().__init__()

        self.cn1 = nn.Conv3d(1, num_kernels[0],
                             kernel_size=7,
                             stride=3,
                             padding=0)
        nn.init.kaiming_normal_(self.cn1.weight)
        self.bn1 = nn.BatchNorm3d(num_kernels[0])

        self.cn2 = nn.Conv3d(num_kernels[0], num_kernels[1],
                             kernel_size=3,
                             stride=3,
                             padding=0)
        nn.init.kaiming_normal_(self.cn2.weight)
        self.bn2 = nn.BatchNorm3d(num_kernels[1])

        self.cn3 = nn.Conv3d(num_kernels[1], num_kernels[2],
                             kernel_size=3,
                             stride=3,
                             padding=1)
        nn.init.kaiming_normal_(self.cn3.weight)

    def forward(self, x):
        l1 = F.relu(self.bn1(self.cn1(x)))
        l2 = F.relu(self.bn2(self.cn2(l1)))
        return self.cn3(l2)

class VanillaDecoder(nn.Module):
    '''
    The decoder portion of the VanillaCAE model.
    '''
    def __init__(self, num_kernels):
        super().__init__()

        self.cn1 = nn.ConvTranspose3d(num_kernels[2], num_kernels[1],
                                      kernel_size=3,
                                      stride=3,
                                      padding=1)
        nn.init.kaiming_normal_(self.cn1.weight)
        self.bn1 = nn.BatchNorm3d(num_kernels[1])

        self.cn2 = nn.ConvTranspose3d(num_kernels[1], num_kernels[0],
                                      kernel_size=3,
                                      stride=3,
                                      padding=0)
        nn.init.kaiming_normal_(self.cn2.weight)
        self.bn2 = nn.BatchNorm3d(num_kernels[0])

        self.cn3 = nn.ConvTranspose3d(num_kernels[0], 1,
                                      kernel_size=7,
                                      stride=3,
                                      padding=0)
        nn.init.kaiming_normal_(self.cn3.weight)

    def forward(self, x):
        l1 = F.relu(self.bn1(self.cn1(x)))
        l2 = F.relu(self.bn2(self.cn2(l1)))
        return torch.sigmoid(self.cn3(l2))
