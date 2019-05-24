import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import resnet_refactored

torch.backends.cudnn.benchmark=True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Routine():
    """
    Class which can run neural network training, validation, and testing routines
    """
    def __init__(self,
                model,
                epochs,
                checkpt_load=True,
                checkpt_loadpath='',
                checkpt_save=True,
                checkpt_savepath='',
                exp_id='',
                verbose=True):

        self.model = model
        self.epochs = epochs
        self.checkpt_loadpath = checkpt_loadpath
        self.checkpt_savepath = checkpt_savepath
        self.exp_id = exp_id
        self.verbose = verbose

    def train(self, trainloader, valloader):
        total = trainloader.dataset.__len__()

        for epoch in range(self.epochs):
            epoch_loss = 0.0 
            epoch_correct = 0

            for data in trainloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                # boilerplate torch training
                self.model.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.model.criterion(outputs,labels)
                loss.backward()
                self.model.optimizer.step()

                # aggregate loss and accuracy for current epoch
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data,1) # for three clf: [0.2, 0.1, 0.7] --> 2
                epoch_correct += (predicted == labels).sum().item()
            
            # calculate metrics from current epoch
            epoch_acc = 100*epoch_correct/total
            epoch_loss = epoch_loss/total

            if(self.verbose):
                print('[epoch %d] Train Loss: %.8f, Train Accuracy: %.8f' % (epoch + 1, epoch_loss, epoch_acc))

            self.validate(valloader)

            if(epoch % 10 == 9):
                torch.save({
                            'epoch': epoch, # REMINDER: what if it's loaded from a checkpoint? Need to consider offset
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.model.optimizer.state_dict(),
                            'loss': loss
                            }, './checkpoints/routine_test')
                            

       
    def validate(self, valloader):
        self.model.eval()

        total = valloader.dataset.__len__()
        print(total)
        correct = 0
        val_loss = 0
        val_acc = 0

        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device),labels.to(device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data,1)
                batch_loss = self.model.criterion(outputs, labels)
                val_loss += batch_loss.item()
                correct += (predicted == labels).sum().item()
        print(correct)
        val_acc = 100*correct/total

        if(self.verbose):
            print('Validation Accuracy: %.3f , Validation Loss: %.6f' % (val_acc, val_loss/total))

        self.model.train()
            
def main():
    """
    For testing purposes only
    """
    transform = transforms.Compose([transforms.ToTensor()])

    cifartrain = torchvision.datasets.CIFAR10(root='../data', train=True, 
                                            download=True , transform=transform)

    trainset, valset = torch.utils.data.random_split(cifartrain, [40000,10000])


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, 
                                                shuffle=True, num_workers=2)

    valloader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=True, num_workers=2)

    model = resnet_refactored.Pretrained(num_classes=10)
    routine = Routine(model=model,epochs=1)

    routine.train(trainloader,valloader)

if __name__=="__main__": 
    main()

    #def validate(self, val_set):

    #def test(self, test_set):

