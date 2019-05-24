import torch
import torch.nn as nn
import torch.nn.functional as F

from pdb import set_trace

class DeepAutoencMRI(nn.Module):
    '''Super deep autoencoder network with pretraining routine.

    MUST RUN ON M40 GPU!
        - One channel (Starting with 64 channels)
            - Classification-only (classification batch size):
                |_ 2 images per GPU with num_blocks [1,1,1,1,1]
        - Three channels
            - Classification-only (classification batch size):
                |_ 4 images per GPU with num_blocks [1,1,1,1,1]
                |_ 3 images per GPU with num_blocks [2,2,2,2,2]
            - Reconstruction-only (pre-training batch size):
                |_ 2 images per GPU with num_blocks [1,1,1,1,1]
                |_ 3 images per GPU with num_blocks [1,1,0,0,0]
            - Classification with frozen weights
                |_ (SGD) 10 images per GPU with num_blocks [1,1,1,1,1]
                |_ (ADAM) 8 images per GPU with num_blocks [1,1,1,1,1]

    Notes:
        - pre-training: 15-20 epochs leads to convergence,
        - training: 210 epochs?

        - 7-layer network
            - FreeSurfer: 2 image / GPU
            - CLINICA: 5 image / GPU
    '''
    def __init__(self, **kwargs):
        super().__init__()

        num_classes = kwargs.get("num_classes", 3)
        num_channels = kwargs.get("num_channels", 1)
        num_blocks = kwargs.get("num_blocks", [1, 1, 1, 1, 1])
        class_dropout = kwargs.get("class_dropout", 0.0)
        cnn_dropout = kwargs.get("cnn_dropout", 0.0)
        self.sparsity = kwargs.get("sparsity", 0.0)

        # input 145, output 143
        self.conv1 = nn.Conv3d(num_channels, 32, kernel_size=3, stride=1,
                                padding=0)

        # input 143, output 143
        self.block1 = ResidualStack(32, num_blocks=num_blocks[0],
                                    bottleneck=True, dropout=cnn_dropout)
        # input 143, output 71
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=0)

        # input 71, output 71
        self.block2 = ResidualStack(64, num_blocks=num_blocks[1],
                                    bottleneck=True, dropout=cnn_dropout)
        # input 71, output 35
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=0)

        # input 35, output 35
        self.block3 = ResidualStack(128, num_blocks=num_blocks[2],
                                    bottleneck=True, dropout=cnn_dropout)
        # input 35, output 17
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=0)

        # input 17, output 17
        self.block4 = ResidualStack(256, num_blocks=num_blocks[3],
                                    bottleneck=True, dropout=cnn_dropout)
        # input 17, output 8
        self.conv5 = nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=0)

        # input 8, output 8
        self.block5 = ResidualStack(512, num_blocks=num_blocks[4],
                                    bottleneck=True, dropout=cnn_dropout)

        # input 8, output 4
        self.conv6 = nn.Conv3d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm3d(512)

        # input 4, output 1
        self.conv7 = nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=0)
        self.bn7 = nn.BatchNorm3d(512)

        self.classification_dropout = nn.Dropout(class_dropout)

        classification_layers = [
            # clinica
            # nn.Linear(2*2*2*512, num_classes)
            # freesurfer
            nn.Linear(4*4*4*512, num_classes)
        ]

        self.classify = nn.Sequential(*classification_layers)

        # decoder batch norm layers
        self.back_bn1 = nn.BatchNorm3d(512)
        self.back_bn2 = nn.BatchNorm3d(512)
        self.back_bn3 = nn.BatchNorm3d(512)

        self.back_bn4 = nn.BatchNorm3d(256)
        self.back_bn5 = nn.BatchNorm3d(256)

        self.back_bn6 = nn.BatchNorm3d(128)
        self.back_bn7 = nn.BatchNorm3d(128)

        self.back_bn8 = nn.BatchNorm3d(64)
        self.back_bn9 = nn.BatchNorm3d(64)

        self.back_bn10 = nn.BatchNorm3d(32)
        self.back_bn11 = nn.BatchNorm3d(32)

    def forward(self, x, reconstruct=False):
        l1 = self.conv1(x)
        l2 = self.conv2(self.block1(l1))
        l3 = self.conv3(self.block2(l2))
        l4 = self.conv4(self.block3(l3))
        l5 = self.conv5(self.block4(l4))
        l6 = F.relu(self.bn6(self.conv6(self.block5(l5))))
        hidden = F.relu(self.bn7(self.conv7(l6)))

        if reconstruct:
            weight_flip = (0, 1, 3, 2, 4)

            conv7, conv6 = self.conv7, self.conv6

            # -> l6q
            deconv2 = F.conv_transpose3d(hidden,
                        conv7.weight.flip(*weight_flip), stride=1)
            deconv2 = F.relu(self.back_bn1(deconv2))

            deconv3 = F.conv_transpose3d(deconv2,
                        conv6.weight.flip(*weight_flip), stride=2, padding=1,
                        output_padding=1)
            deconv3 = F.relu(self.back_bn2(deconv3))

            # -> l5
            deconv4 = F.relu(self.back_bn3(self.block5.backward(deconv3)))

            conv5 = self.conv5
            deconv5 = F.conv_transpose3d(deconv4,
                        conv5.weight.flip(*weight_flip), stride=2)
            deconv5 = F.relu(self.back_bn4(deconv5))

            # -> l4
            deconv6 = F.relu(self.back_bn5(self.block4.backward(deconv5)))

            conv4 = self.conv4
            deconv7 = F.conv_transpose3d(deconv6,
                        conv4.weight.flip(*weight_flip), stride=2)
            deconv7 = F.relu(self.back_bn6(deconv7))

            # -> l3
            deconv8 = F.relu(self.back_bn7(self.block3.backward(deconv7)))

            conv3 = self.conv3
            deconv9 = F.conv_transpose3d(deconv8,
                        conv3.weight.flip(*weight_flip), stride=2)
            deconv9 = F.relu(self.back_bn8(deconv9))

            # -> l2
            deconv10 = F.relu(self.back_bn9(self.block2.backward(deconv9)))

            conv2 = self.conv2
            deconv11 = F.conv_transpose3d(deconv10,
                        conv2.weight.flip(*weight_flip), stride=2)
            deconv11 = F.relu(self.back_bn10(deconv11))

            deconv12 = F.relu(self.back_bn11(self.block1.backward(deconv11)))

            conv1 = self.conv1
            deconv13 = F.conv_transpose3d(deconv12,
                        conv1.weight.flip(*weight_flip), stride=1)

            return torch.sigmoid(deconv13), hidden
        else:
            flattened = hidden.view(len(x), -1)
            dropped = self.classification_dropout(flattened)

            return self.classify(dropped)

    def loss(self, pred, target):
        return F.cross_entropy(pred, target)

    def reconstruction_loss(self, pred, target, hidden_state=None):
        loss = F.mse_loss(pred, target)

        if hidden_state is not None:
            loss += self.sparsity * torch.sum(torch.abs(hidden_state))

        return loss

    def freeze(self):
        '''Freeze the weights of the convolution layers
        '''
        self.block1.freeze()
        for params in self.conv1.parameters():
            params.requires_grad = False

        self.block2.freeze()
        for params in self.conv2.parameters():
            params.requires_grad = False

        self.block3.freeze()
        for params in self.conv3.parameters():
            params.requires_grad = False

        self.block4.freeze()
        for params in self.conv4.parameters():
            params.requires_grad = False

        self.block5.freeze()
        for params in self.conv5.parameters():
            params.requires_grad = False

        for params in self.conv6.parameters():
            params.requires_grad = False
        for params in self.bn6.parameters():
            params.requires_grad = False

        for params in self.conv7.parameters():
            params.requires_grad = False
        for params in self.bn7.parameters():
            params.requires_grad = False

class ResidualStack(nn.Module):
    '''A stack of residual blocks.
    '''
    def __init__(self, num_chan, num_blocks, **kwargs):
        super().__init__()

        layers = [ResidualBlock(num_chan, **kwargs) for _ in range(num_blocks)]
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        result = x

        for layer in self.blocks.children():
            result = layer.forward(result, x)

        return result

    def backward(self, x):
        result = x

        for layer in reversed(list(self.blocks.children())):
            result = layer.backward(result)

        return result

    def freeze(self):
        '''Freeze the weights of the residual block so that no gradient is calculated.
        '''
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        '''Unfreeze the weights of the residual block so that gradient is calculated.
        '''
        for param in self.parameters():
            param.requires_grad = True


class ResidualBlock(nn.Module):
    '''Three-layer residual block.

    Follow "bottleneck" building block from https://arxiv.org/abs/1512.03385
    Follows "pre-activation" setup from https://arxiv.org/abs/1603.05027
    '''
    def __init__(self, num_chan, **kwargs):
        super().__init__()

        dropout = kwargs.get("dropout", 0.0)
        bottleneck = kwargs.get("bottleneck", True)

        hidden_chan = num_chan // 2 if bottleneck else num_chan

        self.dropout = nn.Dropout(dropout)

        self.bn1 = nn.BatchNorm3d(num_chan)
        self.conv1 = nn.Conv3d(num_chan, hidden_chan, kernel_size=1,
                                        stride=1, padding=0)
        nn.init.kaiming_normal_(self.conv1.weight)

        self.bn2 = nn.BatchNorm3d(hidden_chan)
        self.conv2 = nn.Conv3d(hidden_chan, hidden_chan, kernel_size=3,
                                        stride=1, padding=0)
        nn.init.kaiming_normal_(self.conv2.weight)

        self.bn3 = nn.BatchNorm3d(hidden_chan)
        self.conv3 = nn.Conv3d(hidden_chan, num_chan, kernel_size=1,
                                        stride=1, padding=0)
        nn.init.kaiming_normal_(self.conv3.weight)

        self.bn3_back = nn.BatchNorm3d(hidden_chan)
        self.bn2_back = nn.BatchNorm3d(hidden_chan)

    def forward(self, x, prev_state=None):
        dropped = self.dropout(x)

        prev_state = prev_state if prev_state is not None else x

        l1 = self.conv1(F.relu(self.bn1(x)))
        l2 = self.conv2(F.relu(self.bn2(l1)))
        l3 = self.conv3(F.relu(self.bn3(l2)))

        amount_to_pad = (l1.shape[-1] - l3.shape[-1]) // 2
        # times six because there are six sides to pad
        padding = (amount_to_pad, ) * 6

        return F.pad(l3, padding, value=0) + prev_state

    def backward(self, hidden):
        weight_flip = (0, 1, 3, 2, 4)

        # reverse the padding
        shrunk = hidden[:, :, 1:-1, 1:-1, 1:-1]

        l3 = F.conv_transpose3d(shrunk, self.conv3.weight.flip(*weight_flip),
                                stride=1)
        F.relu(self.bn3_back(l3), inplace=True)

        l2 = F.conv_transpose3d(l3, self.conv2.weight.flip(*weight_flip),
                                stride=1)
        F.relu(self.bn2_back(l2), inplace=True)

        l1 = F.conv_transpose3d(l2, self.conv1.weight.flip(*weight_flip),
                                stride=1)

        return l1 + hidden
