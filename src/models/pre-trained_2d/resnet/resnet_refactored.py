import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn

class Pretrained(torch.nn.Module):
    def __init__(self, 
                model_name = 'resnet',
                pretrain = True,
                num_classes = 3,
                optim_name = "ADAM",
                lr = 0.01):

        super(Pretrained, self).__init__()
        """
        Pre-trained model wrapper class for 2D image classification problems. 
        Grabs all layers except the last layer from pretrained model and adds 
        a fresh fully connected layer to the end.
        """
        # Load pre-trained model
        if(model_name == 'resnet'):
            ptmodel = models.resnet18(pretrained=pretrain)
            self.conv = nn.Sequential(*list(ptmodel.children())[:-1])
            num_ftrs = ptmodel.fc.in_features
            self.fc = nn.Linear(512, num_classes)
        #elif(model_name == 'densenet'):
        
        # Load optimizer
        if(optim_name == "ADAM"):
           self.optimizer = optim.Adam(self.parameters(), lr=lr) 
        
        # Define loss, which is always Cross Entropy.
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1,x.size(1))
        x = self.fc(x)
        return x

                



