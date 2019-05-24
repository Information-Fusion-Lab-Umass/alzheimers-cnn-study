########################################################
# Branched off of models/pre-trained_2d/resnet/resnet.py
# This is a simple implementation of pretrained resnet 
# on 2D slices.
# Didn't want to delete anything from above file while
# building this code so created new file here.
# Implemented dataset loader etc and got resnet to work
# on that.  3/16/2019
#######################################################

import os
import cv2
import torch
import pickle
import random
import numpy as np
import nibabel as nib
import torch.nn as nn
from utils.transforms import NaNToNum, RangeNormalization
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
from  torch.nn.modules.upsampling import Upsample
import pandas as pd
from sklearn.utils import shuffle

torch.backends.cudnn.benchmark=True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_split, val_split, test_split = 0.6, 0.2, 0.2

lrate = 0.001
num_epochs = 10
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs,3)
#model.avgpool = nn.AdaptiveAvgPool2d(1)

#send to GPU
model.to(device)

#define loss function + optimizer
optimizer = optim.Adam(model.parameters(),lr=lrate)
criterion = nn.CrossEntropyLoss()

class ADNIDataset2D(Dataset):
    def __init__(self, mode="train", data_ver="presliced"):
        self.mode = mode
        self.data_ver = data_ver
        trans = [T.ToTensor()]
        self.transform = T.Compose(trans)
        trans2 = [NaNToNum(), RangeNormalization()]
        self.transform2 = T.Compose(trans2)
        if data_ver == "presliced":
            self.subsample_path = "/mnt/nfs/work1/mfiterau/ADNI_data/slice_all_no_seg/coronal_skullstrip"
            self.list_of_subjectdir = os.listdir(self.subsample_path)
            if self.mode == "train":
                self.list_of_subjectdir = self.list_of_subjectdir[:int(train_split*1033)]
            elif self.mode == "val":
                self.list_of_subjectdir = self.list_of_subjectdir[int(train_split*1033):int((train_split+val_split)*1033)]
            elif self.mode == "test":
                self.list_of_subjectdir = self.list_of_subjectdir[-int(test_split*1033):]
        elif data_ver == "liveslice":
            with open('outputs/normalized_mapping.pickle', 'rb') as f:
                data = pickle.load(f)
            data = data[data.misc.notnull()] 
            data_AD = data[data["label"] == "AD"]
            data_MCI = data[data["label"] == "LMCI"]
            data_CN = data[data["label"] == "CN"]
            #data_AD = shuffle(data_AD)
            #data_MCI = shuffle(data_MCI)
            #data_CN = shuffle(data_CN)
            if self.mode == "train":
                data_AD = data_AD[:int(586*train_split)]
                data_MCI = data_MCI[:int(586*train_split)] 
                data_CN = data_CN[:int(586*train_split)]
            elif self.mode == "val":
                data_AD = data_AD[int(586*train_split):int(586*(train_split+val_split))]
                data_MCI = data_MCI[int(586*train_split):int(586*(train_split+val_split))]
                data_CN = data_CN[int(586*(train_split)):int(586*(train_split+val_split))]
            elif self.mode == "test":
                data_AD = data_AD[int(586*(train_split+val_split)):int(586*(train_split+val_split+test_split))]
                data_MCI = data_MCI[int(586*(train_split+val_split)):int(586*(train_split+val_split+test_split))]
                data_CN = data_CN[int(586*(train_split+val_split)):int(586*(train_split+val_split+test_split))]
            self.data = pd.concat([data_AD, data_MCI, data_CN],ignore_index=True)
        self.cmap = plt.get_cmap('viridis') 
  
    def __len__(self):
        if self.data_ver=="presliced":
            return len(self.list_of_subjectdir)
        elif self.data_ver=="liveslice":
            if self.mode == "train":
                return int((self.data).shape[0])
            elif self.mode == "val":
                return int((self.data).shape[0])
            elif self.mode == "test":
                return int((self.data).shape[0])
    
    def __getitem__(self, idx):
        if self.data_ver=="presliced":
            img, label = self._get_item_presliced_helper()
        elif self.data_ver=="liveslice":
            img, label = self._get_item_live_slice_helper()
        return img, label

    def _get_item_presliced_helper(self):
        list_of_subjectdir = self.list_of_subjectdir
        subject_path = random.choice(list_of_subjectdir)
        img = cv2.imread(self.subsample_path+"/"+subject_path+"/normalized_seg_33.tiff")
        while img is None:
            subject_path = random.choice(list_of_subjectdir)
            img = cv2.imread(self.subsample_path+"/"+subject_path+"/normalized_seg_33.tiff")
        img = img[:,:,[2,1,0]]
        # Just crop out the white paths, its consistently the same place..
        img = img[50:-40, 25:-35, :]
        img = cv2.resize(img, (256, 256))
        img = self.transform(img)
        label = subject_path.strip(".tiff").split("_")[-1]
        label = 2 if (label=="AD") else (1 if (label=="MCI") else 0)
        return img, label
    
    def _get_item_live_slice_helper(self):
        df = self.data
        idx = random.randint(0, len(df.index)-1)
        data_path = df.ix[idx, "misc"]
        img = self._slice_2D_from_3D(data_path)
        labell = df.ix[idx, "label"]
        label = 2 if (labell=="AD") else (1 if (labell=="LMCI") else 0)
        return img, label

    def _slice_2D_from_3D(self, data_path):
        mri = nib.load(data_path).get_fdata().squeeze() 
        mask_icv = nib.load('mask_ICV.nii').get_fdata().squeeze()
        mri = torch.tensor(mask_icv * mri)
        mri = torch.tensor(mri).permute([0,2,1])
        mri = mri.flip(1) 
        mri = mri.flip(2)
        slice1 = mri[:, :, 67]
        slice1 = slice1.transpose(-2, -1)
        slice1 = self.transform2(slice1)
        slice1 = slice1.numpy()
        #slice1 = cv2.resize(slice1, (224, 224))
        slice1 = self.cmap(slice1)
        slice1 = slice1[:, :, :3]
        slice1 = self.transform(slice1)
        #slice1 = slice1[:, :, ::-1]
        return slice1 

data_slice_ver = "liveslice"
train_dataset = ADNIDataset2D(mode="train", data_ver=data_slice_ver)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=False)
val_dataset = ADNIDataset2D(mode="val", data_ver=data_slice_ver)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
test_dataset = ADNIDataset2D(mode="test", data_ver=data_slice_ver)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

train_loss_vals = []
val_loss_vals = []
train_acc_vals = []
val_acc_vals = []
epochs = []

for epoch in range(num_epochs):
    running_loss = 0.0
    total_train = 0
    train_correct = 0

    for num_iter, (x,y) in enumerate(train_loader):
        x, y = x.to(device, dtype=torch.float), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_train += y.size(0)

        _, pred = torch.max(pred.data,1)
        train_correct += (pred == y).sum().item()

    train_acc = float(train_correct)/total_train
    print("Train Accuracy", epoch, train_correct, total_train, str(train_acc)+"%")

    running_loss_val = 0.0
    total_val = 0
    val_correct = 0
    for num_iter, (x, y) in enumerate(val_loader):
        x, y = x.to(device, dtype=torch.float), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)

        running_loss_val += loss.item()
        total_val += y.size(0)

        _, pred = torch.max(pred.data,1)
        val_correct += (pred == y).sum().item()
        
    val_acc = float(val_correct)/total_val
    print("Val Accuracy", epoch, val_correct, total_val, str(val_acc)+"%")

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, './checkpoints/resnet_lr_0.001_cifar_epoch_{}34.test'.format(epoch))

    train_loss_vals.append(running_loss / total_train)
    val_loss_vals.append(running_loss_val / total_val)
    val_acc_vals.append(val_acc) 
    train_acc_vals.append(train_acc)
    epochs.append(epoch)


f = plt.figure(figsize=(10,10))

# loss graph
f.add_subplot(2,1,1)
plt.plot(epochs,train_loss_vals,label='train loss')
plt.plot(epochs,val_loss_vals,label='val loss')
plt.title('Loss and Accuracy over Epochs')
plt.legend()
plt.ylabel('loss')

# accuracy graph
f.add_subplot(2,1,2)
plt.plot(epochs,train_acc_vals,label='train acc')
plt.plot(epochs,val_acc_vals,label='val_acc')
plt.ylabel('acc')
plt.legend()

plt.savefig('./figures/resnet_CIFAR_train_val' + str(66) + '.png')


total_test = 0
test_correct = 0
for num_iter, (x, y) in enumerate(test_loader):
    x, y = x.to(device, dtype=torch.float), y.to(device)
    pred = model(x)
    total_test += y.size(0)
    _, pred = torch.max(pred.data,1)
    test_correct += (pred == y).sum().item()
print("Test Accuracy", epoch, test_correct, total_test, str(float(test_correct)/total_test)+"%")
