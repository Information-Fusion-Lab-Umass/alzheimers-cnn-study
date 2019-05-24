import torch
import torch.nn as nn
import torch.nn.functional as F

class AE_CNN(nn.Module):
    """
    Sparse Autoencoder + Convolution Neural Network Classifier
    """
    def __init__(self, n_filters=150, dropout=0, n_classes=2):
        super(AE_CNN, self).__init__()
        self.downsample = nn.MaxPool3d(2, 2)
        self.encode = nn.Conv3d(1, n_filters, 5)
        self.decode = nn.ConvTranspose3d(n_filters, 1, 5)
        self.pool = nn.MaxPool3d(5, 5)
        self.fc1 = nn.Linear(n_filters * 11 * 13 * 11, 800)
        self.fc2 = nn.Linear(800, n_classes)
        self.n_filters = n_filters
        self.dropout = nn.Dropout(p=dropout)

    def reconstruct(self, x):
        #x = self.downsample(x)
        h = F.relu(self.encode(x))
        out = F.relu(self.decode(h))
        return out, h, x

    def forward(self, train=False):
        d = self.downsample(x)
        h = F.relu(self.encode(d))
        h = self.pool(h)
        h = h.view(-1, self.n_filters * 11 * 13 * 11)
        h = self.dropout(h)
        h = F.relu(self.fc1(h))
        out = self.fc2(h)
        return out

    def reconstruction_loss(self, x, h, out):
        nn.MSELoss(x, out) + torch.abs(h).sum()

    def loss(self, x, y):
        return F.cross_entropy(x, y)
