import torch
import torch.nn as nn

class NetWork(nn.Module):

    def __init__(self, in_channel=1,feat_dim=1024,expansion = 4, type_name='conv3x3x3', norm_type = 'Instance'):
        super(NetWork, self).__init__()
        

        self.conv = nn.Sequential()

        self.conv.add_module('conv0_s1',nn.Conv3d(in_channel, 4*expansion, kernel_size=1))

        if norm_type == 'Instance':
           self.conv.add_module('lrn0_s1',nn.InstanceNorm3d(4*expansion))
        else:
           self.conv.add_module('lrn0_s1',nn.BatchNorm3d(4*expansion))
        self.conv.add_module('relu0_s1',nn.ReLU(inplace=True))
        self.conv.add_module('pool0_s1',nn.MaxPool3d(kernel_size=3, stride=2))

        self.conv.add_module('conv1_s1',nn.Conv3d(4*expansion, 32*expansion, kernel_size=3,padding=0, dilation=2))
        
        if norm_type == 'Instance':
            self.conv.add_module('lrn1_s1',nn.InstanceNorm3d(32*expansion))
        else:
            self.conv.add_module('lrn1_s1',nn.BatchNorm3d(32*expansion))
        self.conv.add_module('relu1_s1',nn.ReLU(inplace=True))
        self.conv.add_module('pool1_s1',nn.MaxPool3d(kernel_size=3, stride=2))


        
        
        self.conv.add_module('conv2_s1',nn.Conv3d(32*expansion, 64*expansion, kernel_size=5, padding=2, dilation=2))
        
        if norm_type == 'Instance':
            self.conv.add_module('lrn2_s1',nn.InstanceNorm3d(64*expansion))
        else:
            self.conv.add_module('lrn2_s1',nn.BatchNorm3d(64*expansion))
        self.conv.add_module('relu2_s1',nn.ReLU(inplace=True))
        self.conv.add_module('pool2_s1',nn.MaxPool3d(kernel_size=5, stride=2))

        
        self.conv.add_module('conv3_s1',nn.Conv3d(64*expansion, 64*expansion, kernel_size=3, padding=1, dilation=2))
        
        if norm_type == 'Instance':
            self.conv.add_module('lrn3_s1',nn.InstanceNorm3d(64*expansion))
        else:
            self.conv.add_module('lrn2_s1',nn.BatchNorm3d(64*expansion))
        self.conv.add_module('relu3_s1',nn.ReLU(inplace=True))
        #self.conv.add_module('pool2_s1',nn.MaxPool3d(kernel_size=5, stride=2))
        #self.fu = nn.MaxPool3d(kernel_size=5, stride=2)
        self.fc6 = nn.Sequential()
        self.fc6.add_module('fc6_s1',nn.Linear(64*expansion*5*5*5, feat_dim))
        #self.age_encoder = AgeEncoding(512,0.1,feat_dim)

        images = torch.randn(10,1,96,96,96)
        um = self.conv(images)
        print(um.size())
        #print(self.fu(um).size())
stuff = NetWork()
