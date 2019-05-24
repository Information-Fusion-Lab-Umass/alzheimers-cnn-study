import torch
import torch.nn as nn
import torch.nn.functional as F

from pdb import set_trace

class DeepMRI(nn.Module):
    '''
    Super deep network with many convolution layers. MUST RUN ON M40 GPU!
    '''
    def __init__(self, **kwargs):
        super().__init__()
        num_classes = kwargs.get("num_classes", 3)
        num_channels = kwargs.get("num_channels", 1)

        # input 145x145x145, output 143x143x143
        self.block1_1 = ResidualBlock(num_channels, 16, dropout=0.0)
        # input 143x143x143, output 143x143x143
        self.block1_2 = ResidualBlock(16, 16, dropout=0.5)
        # input 143x143x143, output 143x143x143
        self.block1_3 = ResidualBlock(16, 16, dropout=0.5)
        # input 143x143x143, output 143x143x143
        self.block1_4 = ResidualBlock(16, 16, dropout=0.5)

        # input 143x143x143, output 71x71x71
        self.mp1 = nn.MaxPool3d(2, stride=2)

        # input 71x71x71, output 69x69x69
        self.block2_1 = ResidualBlock(16, 32, dropout=0.5)
        # input 69x69x69, output 69x69x69
        self.block2_2 = ResidualBlock(32, 32, dropout=0.5)
        # input 69x69x69, output 69x69x69
        self.block2_3 = ResidualBlock(32, 32, dropout=0.5)
        # input 69x69x69, output 69x69x69
        self.block2_4 = ResidualBlock(32, 32, dropout=0.5)

        # input 69x69x69, output 34x34x34
        self.mp2 = nn.MaxPool3d(2, stride=2)

        # input 34x34x34, output 32x32x32
        self.block3_1 = ResidualBlock(32, 64, dropout=0.5)
        # input 32x32x32, output 32x32x32
        self.block3_2 = ResidualBlock(64, 64, dropout=0.5)
        # input 32x32x32, output 32x32x32
        self.block3_3 = ResidualBlock(64, 64, dropout=0.5)
        # input 32x32x32, output 32x32x32
        self.block3_4 = ResidualBlock(64, 64, dropout=0.5)

        # input 32x32x32, output 16x16x16
        self.mp3 = nn.MaxPool3d(2, stride=2)

        # input 16x16x16, output 14x14x14
        self.block4_1 = ResidualBlock(64, 128, dropout=0.5)
        # input 14x14x14, output 14x14x14
        self.block4_2 = ResidualBlock(128, 128, dropout=0.5)
        # input 14x14x14, output 14x14x14
        self.block4_3 = ResidualBlock(128, 128, dropout=0.5)
        # input 14x14x14, output 14x14x14
        self.block4_4 = ResidualBlock(128, 128, dropout=0.5)

        # input 14x14x14, output 7x7x7
        self.mp4 = nn.MaxPool3d(2, stride=2)

        # input 7x7x7, output 5x5x5
        self.conv1 = ConvolutionBlock(128, 256, kernel_size=3, stride=1,
                        padding=0, batch_norm=True, max_pool=False, relu=True)
        # input 5x5x5, output 3x3x3
        self.conv2 = ConvolutionBlock(256, 256, kernel_size=3, stride=1,
                        padding=0, batch_norm=True, max_pool=False, relu=True)

        # input 3x3x3, output 1x1x1
        self.global_pool = nn.AvgPool3d(3)

        self.class_dropout = nn.Dropout(p=0.5)

        classification_layers = [
            nn.Linear(1*1*1*256, 128),
            nn.ReLU(True),
            nn.Linear(128, 16),
            nn.ReLU(True),
            nn.Linear(16, num_classes)
        ]

        self.classify = nn.Sequential(*classification_layers)

    def forward(self, x):
        b1_1, res1_1 = self.block1_1(x)
        b1_2, res1_2 = self.block1_2(b1_1, res1_1)
        b1_3, _ = self.block1_3(b1_2, res1_2)
        b1_4, _ = self.block1_4(b1_3, res1_1)
        g1 = self.mp1(b1_4)

        b2_1, res2_1 = self.block2_1(g1)
        b2_2, res2_2 = self.block2_2(b2_1, res2_1)
        b2_3, _ = self.block2_3(b2_2, res2_2)
        b2_4, _ = self.block2_4(b2_3, res2_1)
        g2 = self.mp2(b2_4)

        b3_1, res3_1 = self.block3_1(g2)
        b3_2, res3_2 = self.block3_2(b3_1, res3_1)
        b3_3, _ = self.block3_3(b3_2, res3_2)
        b3_4, _ = self.block3_4(b3_3, res3_1)
        g3 = self.mp3(b3_4)

        b4_1, res4_1 = self.block4_1(g3)
        b4_2, res4_2 = self.block4_2(b4_1, res4_1)
        b4_3, _ = self.block4_3(b4_2, res4_2)
        b4_4, _ = self.block4_4(b4_3, res4_1)
        g4 = self.mp4(b4_4)

        conv1 = self.conv1(g4)
        conv2 = self.conv2(conv1)

        pooled = self.global_pool(conv2)

        final_conv = self.class_dropout(pooled)

        hidden = final_conv.view(len(x), -1)
        scores = self.classify(hidden)

        return scores

    def reconstruct(self, x):
        raise NotImplementedError("Pretraining is not implemented for this model. Set pretrain num_epochs to 0.")

    def loss(self, x, y):
        return F.cross_entropy(x, y)

    def reconstruction_loss(self, x, y, hidden=None):
        loss = F.mse_loss(x, y)

        # sparsity constraint
        if hidden is not None:
            loss += torch.sum(torch.abs(hidden))

        return loss

class ConvolutionBlock(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__()
        conv_params = {
            "kernel_size": kwargs.get("kernel_size", 3),
            "stride": kwargs.get("conv_stride", 1),
            "padding": kwargs.get("padding", 0)
        }
        batch_norm = kwargs.get("batch_norm", True)
        max_pool = kwargs.get("max_pool", True)
        max_pool_kernel = kwargs.get("pool_kernel", 2)
        max_pool_stride = kwargs.get("pool_stride", 2)
        relu = kwargs.get("relu", True)

        layers = []

        conv = nn.Conv3d(input_dim, output_dim, **conv_params)

        layers.append(conv)

        if max_pool:
            layers.append(nn.MaxPool3d(max_pool_kernel, stride=max_pool_stride))

        if relu:
            layers.append(nn.ReLU(True))
            nn.init.kaiming_normal_(conv.weight)
        else:
            nn.init.xavier_normal_(conv.weight)

        if batch_norm:
            layers.append(nn.BatchNorm3d(output_dim))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__()
        dropout = kwargs.get("dropout", 0.0)
        bottleneck = kwargs.get("bottleneck", True)

        hidden_dim = output_dim // 2 if bottleneck else output_dim

        self.dropout = nn.Dropout(dropout)

        self.conv1 = ConvolutionBlock(input_dim, output_dim, kernel_size=3,
                        conv_stride=1, max_pool=False, relu=True)
        self.conv2 = ConvolutionBlock(output_dim, hidden_dim,
                        kernel_size=3, conv_stride=1, max_pool=False, relu=True)
        self.conv3 = ConvolutionBlock(hidden_dim, output_dim,
                        kernel_size=3, conv_stride=1, max_pool=False,
                        relu=False)

    def forward(self, x, prev_state=None):
        dropped = self.dropout(x)
        l1 = self.conv1(dropped)
        l3 = self.conv3(self.conv2(l1))

        prev_state = prev_state if prev_state is not None else l1

        amount_to_pad = (prev_state.shape[-1] - l3.shape[-1]) // 2
        padding = (amount_to_pad, ) * 6

        output = F.pad(l3, padding, value=0) + prev_state
        F.relu(output, inplace=True)

        return output, l1
