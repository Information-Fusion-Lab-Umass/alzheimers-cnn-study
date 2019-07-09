import torch
import torch.nn as nn
import torch.nn.functional as F

from pdb import set_trace

class Wang3D(nn.Module):
    ''' Replicating Wang et al. (2018) ICMLA paper. 

    MUST RUN ON M40 GPU!

    Notes:
        - Growth rate for the # of new feature-volumes increased at each layer
            |- 24 is the best, as reported in fig 3.
        - Two dense blocks
            |- Total depth of ~20 layers, as reported in fig 4. 
        - Reduction for the 1x1x1 convolutions
            |- Used the degree of compression theta=0.7, as reported in fig 5.
    '''
    def __init__(self, **kwargs):
        super().__init__()

        num_classes = kwargs.get("num_classes", 3)
        num_channels = kwargs.get("num_channels", 1)
        num_blocks = kwargs.get("num_blocks", [1, 1, 1, 1, 1])
        class_dropout = kwargs.get("class_dropout", 0.1)
        cnn_dropout = kwargs.get("cnn_dropout", 0.1)
        self.sparsity = kwargs.get("sparsity", 0.0)

        # input 145, output 14
        self.conv1 = nn.Conv3d(num_channels, 24, kernel_size=3, stride=2,
                                padding=0)

        #num_layers, num_input_features, bn_size, growth_rate, drop_rate

        # input 143, output 143
        self.block1 = _DenseBlock(8, 24, 4, 24, cnn_dropout-0.05)
        # input 143, output 71
        self.conv2 = nn.Conv3d(216, 151, kernel_size=3, stride=2, padding=0)
        self.pool2 = nn.MaxPool3d(2, stride=2)

        # input 71, output 71
        self.block2 = _DenseBlock(9, 151, 4, 24, cnn_dropout)
        # input 71, output 35
        self.conv3 = nn.Conv3d(367, 257, kernel_size=3, stride=2, padding=0)

        self.classification_dropout = nn.Dropout(class_dropout)

        classification_layers = [
            #nn.Linear(12960, 1000),
            nn.Linear(3084,1000),
            nn.Linear(1000, 100),
            nn.Linear(100, num_classes)
        ]

        self.classify = nn.Sequential(*classification_layers)

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, mean=0., std=0.01)

        for m in self.modules():
            m.apply(init_weights)

    def forward(self, x, reconstruct=False):
        l1 = self.conv1(x)
        l2 = self.pool2(self.conv2(self.block1(l1)))
        l3 = self.pool2(self.conv3(self.block2(l2)))

        flattened = l3.view(len(x), -1)
        dropped = self.classification_dropout(flattened)
        return self.classify(dropped)

    def loss(self, pred, target):
        return F.cross_entropy(pred, target)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate,
                                bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, growth_rate,
                                           kernel_size=3, stride=1,
                                           padding=1,bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return torch.cat([x, new_features], 1)
