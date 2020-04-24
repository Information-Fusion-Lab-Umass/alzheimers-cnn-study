# -*- coding: utf-8 -*-
"""
@author: Sheng
"""
import torch
import torch.nn as nn
from torch import cat
import torch.nn.init as init
import math
import sys
sys.path.append('Utils')
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor


class LiuNet(nn.Module):

    def __init__(self, in_channel=1,feat_dim=1024,expansion = 8, type_name='conv3x3x3', norm_type = 'Instance'):
        super(LiuNet, self).__init__()
        
        self.conv = nn.Sequential()

        self.conv.add_module('conv0_s1',nn.Conv3d(in_channel, 4*expansion, kernel_size=1))
        self.conv.add_module('lrn0_s1',nn.InstanceNorm3d(4*expansion))
        self.conv.add_module('relu0_s1',nn.ReLU(inplace=True))
        self.conv.add_module('pool0_s1',nn.MaxPool3d(kernel_size=3, stride=2))

        self.conv.add_module('conv1_s1',nn.Conv3d(4*expansion, 32*expansion, kernel_size=3,padding=0, dilation=2))
        self.conv.add_module('lrn1_s1',nn.InstanceNorm3d(32*expansion))
        self.conv.add_module('relu1_s1',nn.ReLU(inplace=True))
        self.conv.add_module('pool1_s1',nn.MaxPool3d(kernel_size=3, stride=2))

        self.conv.add_module('conv2_s1',nn.Conv3d(32*expansion, 64*expansion, kernel_size=5, padding=2, dilation=2))
        self.conv.add_module('lrn2_s1',nn.InstanceNorm3d(64*expansion))
        self.conv.add_module('relu2_s1',nn.ReLU(inplace=True))
        self.conv.add_module('pool2_s1',nn.MaxPool3d(kernel_size=5, stride=2))

        self.conv.add_module('conv3_s1',nn.Conv3d(64*expansion, 64*expansion, kernel_size=3, padding=1, dilation=2))
        self.conv.add_module('lrn3_s1',nn.InstanceNorm3d(64*expansion))
        self.conv.add_module('relu3_s1',nn.ReLU(inplace=True))
        self.conv.add_module('pool2_s1',nn.MaxPool3d(kernel_size=5, stride=2))
        
        self.fc = nn.Sequential()
        self.fc.add_module('fc',nn.Linear(64*expansion*5*5*5, feat_dim))
        
        pe = torch.zeros(240, 512)
        position = torch.arange(0, 240).unsqueeze(1).float()
        div_term = torch.exp((torch.arange(0, 512, 2).float() *
                             -(math.log(10000.0) / 512)).float())
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        self.age_encoder = nn.Sequential()
        self.age_encoder.add_module('fc6_s1',nn.Linear(512,512))
        self.age_encoder.add_module('lrn0_s1',nn.LayerNorm(512))
        self.age_encoder.add_module('fc6_s3',nn.Linear(512, feat_dim))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(1024,512),
            nn.Linear(512,3)
        )

        for m in self.modules():
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, mean=0., std=0.01)
                m.bias.data.fill_(0.0)

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load(self,checkpoint):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(checkpoint)['state_dict']
        pretrained_dict = {k[6:]: v for k, v in list(pretrained_dict.items()) if k[6:] in model_dict and 'conv3_s1' not in k and 'fc6' not in k and 'fc7' not in k and 'fc' not in k}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print([k for k, v in list(pretrained_dict.items())])
        return pretrained_dict.keys()

    def freeze(self, pretrained_dict_keys):
        for name, param in self.named_parameters():
            if name in pretrained_dict_keys:
                param.requires_grad = False
                
    def save(self,checkpoint):
        torch.save(self.state_dict(), checkpoint)
    
    def forward(self, x, age_id):
        z = self.conv(x)
        z = self.fc(z.view(x.shape[0],-1))
        if age_id is not None:
            y = torch.autograd.Variable(self.pe[age_id,:],
                         requires_grad=False)
            y = self.age_encoder(y)
            z += y
        return self.classifier(z)
 
    def compute_logit_loss(self, input, targets):
        x, age_id, targets = input[0].to(self.device), input[1].to(self.device), targets.to(self.device)
        outputs = self.forward(x, age_id)
        loss = self.criterion(outputs, targets)
        return outputs, loss
