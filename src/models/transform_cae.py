import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce

class SpatialTransformConvAutoEnc(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.input_transform = InputSpatialTransformer()

        num_features = [16, 32, 64, 128]

        encoder_layers = [
            # input 256x256x256, output 127x127x127
            ConvolutionBlock(1, num_features[0], kernel_size=3,
                             max_pool=False),
            # input 127x127x127, output 63x63x63
            ConvolutionBlock(num_features[0], num_features[1],
                             kernel_size=3, conv_stride=2, max_pool=False),
            # input 63x63x63, output 30x30x30
            ConvolutionBlock(num_features[1], num_features[2],
                             kernel_size=3, conv_stride=2, max_pool=False),
            # input 30x30x30, output 14x14x14
            ConvolutionBlock(num_features[2], num_features[3],
                             kernel_size=3, conv_stride=2, max_pool=False,
                             relu=False)
        ]

        decoder_layers = [
            # input 14x14x14, output 30x30x30
            DeconvolutionBlock(num_features[3], num_features[2],
                               kernel_size=3, stride=2, output_padding=1),
            # input 30x30x30, output 63x63x63
            DeconvolutionBlock(num_features[2], num_features[1],
                               kernel_size=3, stride=2, output_padding=1),
            # input 63x63x63, output 127x127x127
            DeconvolutionBlock(num_features[1], num_features[0],
                               kernel_size=3, stride=2),
            # input 127x127x127, output 256x256x256
            DeconvolutionBlock(num_features[0], 1,
                               kernel_size=3, stride=2, padding=1,
                               output_padding=1, relu=False)
        ]

        self.encode = nn.Sequential(*encoder_layers)
        self.decode = nn.Sequential(*decoder_layers)

    def forward(self, x):
        transformed_x = self.input_transform(x)
        encoded_x = self.encode(transformed_x)
        decoded_x = self.decode(encoded_x)

        return torch.sigmoid(transformed_x * decoded_x)

    def loss(self, x, y):
        return F.mse_loss(x, y)

class InputSpatialTransformer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.loc_input_dim = 112
        loc_hidden_dim = kwargs.get("loc_hidden_dim", 32)

        localization_layers = [
            # input 256x256x256, output 123x123x123
            ConvolutionBlock(1, 4, kernel_size=11, conv_stride=2,
                             max_pool=False),
            # input 123x123x123, output 57x57x57
            ConvolutionBlock(4, 6, kernel_size=11, conv_stride=2,
                             max_pool=False),
            # input 57x57x57, output 27x27x27
            ConvolutionBlock(6, 8, kernel_size=5, conv_stride=2,
                             max_pool=False),
            # input 27x27x27, output 13x13x13
            ConvolutionBlock(8, 10, kernel_size=3, conv_stride=2,
                             max_pool=False),
            # input 13x13x13, output 6x6x6
            ConvolutionBlock(10, 12, kernel_size=3, conv_stride=2,
                             max_pool=False),
            # input 6x6x6, output 2x2x2
            ConvolutionBlock(12, 14, kernel_size=3, conv_stride=2,
                             max_pool=False, relu=False)
        ]

        self.localize = nn.Sequential(*localization_layers)
        self.fc_loc = nn.Sequential(
            # input 14*2*2*2
            nn.Linear(self.loc_input_dim, loc_hidden_dim),
            nn.ReLU(True),
            nn.Linear(loc_hidden_dim, 4 * 3)
        )

        identity = torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                                dtype=torch.float)

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(identity)

    def forward(self, x):
        localized = self.localize(x)
        localized = localized.view(-1, self.loc_input_dim)
        theta = self.fc_loc(localized)
        theta = theta.view(-1, 4, 3)

        grid = F.affine_grid(theta, x.size())
        return F.grid_sample(x, grid)

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
        max_pool_stride = kwargs.get("pool_stride", 2)
        relu = kwargs.get("relu", True)

        layers = []

        conv = nn.Conv3d(input_dim, output_dim, **conv_params)

        layers.append(conv)

        if max_pool:
            layers.append(nn.MaxPool3d(max_pool_stride))

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

class DeconvolutionBlock(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__()
        conv_params = {
            "kernel_size": kwargs.get("kernel_size", 3),
            "stride": kwargs.get("stride", 1),
            "padding": kwargs.get("padding", 0),
            "output_padding": kwargs.get("output_padding", 0)
        }
        batch_norm = kwargs.get("batch_norm", True)
        relu = kwargs.get("relu", True)

        conv = nn.ConvTranspose3d(input_dim, output_dim, **conv_params)

        layers = []
        layers.append(conv)

        if relu:
            nn.init.kaiming_normal_(conv.weight)
            layers.append(nn.ReLU(True))
        else:
            nn.init.xavier_normal_(conv.weight)

        if batch_norm:
            layers.append(nn.BatchNorm3d(output_dim))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
