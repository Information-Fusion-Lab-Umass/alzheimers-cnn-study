import torch
import torch.nn as nn
import torch.nn.functional as F

class Soes3D(nn.Module):
    ''' Replicating Esmaeilzadeh (2018) Miccai Paper
        
        End-To-End Alzheimer's Disease Diagnosis 
        and Biomarker Identification

        - Simple 3D architecture

    '''
     def __init__(self, **kwargs):
        super().__init__()

        num_classes = kwargs.get("num_classes", 3)
        num_channels = kwargs.get("num_channels", 1)
        cnn_dropout = kwards.get("cnn_dropout", 0.1)
        class_dropout = kwargs.get("class_dropout", 0.1)
        
        # 3^3 x 32 filters
        self.conv1 = nn.Conv3d(num_channels, 32, kernel_size=3, stride=1,
                                padding=1)
        self.pool1 = nn.MaxPool3d(2, stride=2)

        # 3^3 x 64 filters
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=2,
                                padding=1)
        self.pool2 = nn.MaxPool3d(3, padding=1)

        # 3^3 x 128 filters
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=2,
                                padding=1)
        self.pool3 = nn.MaxPool3d(4, padding=1)

        self.cnn_dropout = nn.Dropout(cnn_dropout)
        self.classification_dropout = nn.Dropout(class_dropout)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x, reconstruct=False):
        l1 = self.pool1(self.conv1(x))
        l2 = self.pool2(self.conv2(x))
        l2 = self.cnn_dropout(l2)
        l3 = self.pool3(self.conv3(x))
        flattened = l3.view(len(x), -1)
        dropped = self.classification_dropout(flattened)
        return self.classify(dropped)

    def loss(self, pred, target):
        return F.cross_entropy(pred, target)
