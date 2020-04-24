import os, torch, pdb
import numpy as np
import json
from PIL import Image
from PIL import ImageFile
import torch.utils.data as data
import random
import collections
from numpy import random as nprandom
import pickle
import glob
import re
import numpy as np
import pandas as pd
from random import shuffle
import random
import math
import nibabel as nib
import scipy
from sklearn import preprocessing
from scipy.ndimage.interpolation import *
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2

class ADNI_dataset(data.Dataset):

    def __init__(self, method, data_lookup_file, mode = 'Train', split_ratios = [], n_label = 3):
        self.LABEL_MAPPING = preprocessing.LabelEncoder()
        if n_label == 3:
            self.LABEL_MAPPING.fit(["CN", "MCI", "AD"])
        elif n_label == 2:
            self.LABEL_MAPPING.fit(["CN", "AD"])
        self.MAPPING = pd.io.parsers.read_csv(data_lookup_file, sep=',')
        if mode == "Train":
            self.MAPPING = self.MAPPING[:int(self.MAPPING.shape[0]*split_ratios[0])]
        elif mode == "Val":
            self.MAPPING = self.MAPPING[int(self.MAPPING.shape[0]*split_ratios[0]):\
                               int(self.MAPPING.shape[0]*(split_ratios[0]+split_ratios[1]))]
        else:
            self.MAPPING = self.MAPPING[int(self.MAPPING.shape[0]*split_ratios[-1]):]
        self.method = method
        self.mode = mode
        
    def __len__(self):
        return self.MAPPING.shape[0]

    def __getitem__(self, idx):
        label = self.LABEL_MAPPING.transform([self.MAPPING.iloc[idx][2]])[0]
        try:
            MRI_path = self.MAPPING.iloc[idx][3]
            image = self.load_MRI(MRI_path) 
            image[np.isnan(image)] = 0.0
            image = (image - image.min())/(image.max() - image.min() + 1e-6)
            if self.method == "jain":
                return self.preprocess_Jain(image, label)
            elif self.method == "liu":
                age = float(self.MAPPING.iloc[idx][4])
                return self.preprocess_Liu(image, age, label)
        except Exception as e:
            print(f"Failed to load #{idx}: {MRI_path}")
            print(f"Errors encountered: {e}")

    def load_MRI(self, image_path):
        if ".gz" in image_path or ".nii" in image_path:
            return nib.load(image_path).get_data().squeeze()
        return cv2.imread(image_path)

    def preprocess_Jain(self, image, label):
        return image.transpose((2, 0, 1)).astype(np.float32), label

    def preprocess_Liu(self, image, age, label):
        if self.mode == 'Train':
            image = self.augment_image(image)
            image = self.randomCrop(image,96,96,96)
        else:
            image = self.centerCrop(image,96,96,96)
        image = np.expand_dims(image,axis=0)
        age = list(np.arange(0.0,120.0,0.5)).index(age)
        return image.astype(np.float32), age, label

    def centerCrop(self, img, length, width, height):
        x = img.shape[0]//2 - length//2
        y = img.shape[1]//2 - width//2
        z = img.shape[2]//2 - height//2
        return img[x:x+length, y:y+width, z:z+height]

    def randomCrop(self, img, length, width, height):
        x = random.randint(0, img.shape[0] - length)
        y = random.randint(0, img.shape[1] - width)
        z = random.randint(0, img.shape[2] - height )
        return img[x:x+length, y:y+width, z:z+height]

    def augment_image(self, image):
        sigma = np.random.uniform(0.0,1.0,1)[0]
        return scipy.ndimage.filters.gaussian_filter(image, sigma, truncate=8)
