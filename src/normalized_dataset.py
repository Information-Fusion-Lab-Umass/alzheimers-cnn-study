import os
import yaml
import torch
import pickle
import traceback
import pandas as pd
import numpy as np
import nibabel as nib
import torchvision.transforms as T

from pdb import set_trace
from PIL import Image
from torch.utils.data import Dataset
from utils.transforms import RangeNormalization, NaNToNum, PadToSameDim
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
import random

class NormalizedDataset(Dataset):
    ''' Data Generator
    '''
    VALID_MODES = ["train", "valid", "test", "all"]
    VALID_TASKS = ["pretrain", "classify"]

    def __init__(self, **kwargs):
        self.config = kwargs.get("config", {
            "image_col": [ "misc" ],
            "label_col": "label",
            "label_path": "outputs/normalized_mapping.pickle"
        })
        
        # number of dimensions in the image, 2D vs 3D
        self.num_dim = kwargs.get("num_dim", self.config["data"]["num_dim"])
        # if 2D, which view and index at which to slice
        self.slice_view = kwargs.get("slice_view",
                                      self.config["data"]["slice_view"])
        self.slice_idx = kwargs.get("slice_num",
                                    self.config["data"]["slice_num"])
        
        # limit for the size of the dataset, for debugging purposes
        self.limit = kwargs.get("limit", -1)
        self.verbose = kwargs.get("verbose", self.config["verbose"])

        transforms = kwargs.get("transforms", [
            T.ToTensor()
        ])
        self.transforms = T.Compose(transforms)

        # name of the image and label column in the dataframe
        self.image_col = self.config["image_col"]
        self.label_col = self.config["label_col"]

        # skull-stripping mask
        mask_path = self.config.get("mask_path", None)
        if mask_path is not None:
            self.brain_mask = nib.load(mask_path) \
                                .get_fdata() \
                                .squeeze()
        else:
            self.brain_mask = None

        mapping_path = kwargs.get("mapping_path",
                                  self.config["label_path"])
        mode = kwargs.get("mode", "all")
        self.mode = mode
        task = kwargs.get("task", "classify")
        valid_split = kwargs.get("valid_split", 0.2)
        test_split = kwargs.get("test_split", 0.0)

        df, label_encoder = self._get_data(mapping_path)
        input_encoder = kwargs.get("label_encoder", None)
        self.label_encoder = label_encoder if input_encoder is None \
                                           else input_encoder

        self.class_balancing_live = False
        if "class_balancing_live" in self.config:
           if self.config["class_balancing_live"] == True:
               self.class_balancing_live = True

        self.customized_split_idx = -1
        if "customized_split_idx" in self.config["data"]:
            self.customized_split_idx = self.config["data"]["customized_split_idx"]

        self.features_to_pickles = kwargs.get("features_to_pckl", False)
        self.cross_val_fold_num = kwargs.get("cross_val_fold", 0)
        self.dataframe = self._split_data(df, valid_split, test_split, mode, task)

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        if self.class_balancing_live == True and self.mode == "train":
            label = random.choice(["CN", "MCI", "AD"])
            data = self.dataframe[self.dataframe["label"] == label].sample(n=1)
            idx = data.drop(["index"], axis=1).index[0] 
        image_paths = self._get_img_paths(idx)
        image = self._load_imgs(image_paths)

        label = self.dataframe[self.label_col].iloc[idx]
        encoded_label = self.label_encoder.transform([label])[0]
        
        if self.features_to_pickles == True:
            patient_id = self.dataframe["patient_id"].iloc[idx]
            visit_code = self.dataframe["visit_code"].iloc[idx]
            encoded_label = (patient_id, visit_code)

        if self.verbose:
            print("Fetched image (label: {}/{}) from {}."
                    .format(label, encoded_label, image_paths))

        if image is None:
            return None, None
        else:
            if self.num_dim == 2:
                return self.transforms(image), encoded_label
            elif self.num_dim == 3:
                # transforms for 3D happens in load_data
                return image, encoded_label

    # ==============================
    # Helper Methods
    # ==============================
    def _get_img_paths(self, idx):
        image_paths = []

        for channel in range(len(self.image_col)):
            image_path = self.dataframe[self.image_col[channel]].iloc[idx]
            image_paths.append(image_path)

        return image_paths

    def _load_imgs(self, paths, **kwargs):
        images = []

        #slice_num = self.slice_idx[0]  ##
        for idx in range(len(paths)):
            try:
                image = nib.load(paths[idx]) \
                           .get_fdata() \
                           .squeeze()

                if self.brain_mask is not None:
                    image *= self.brain_mask

                # extract slices for 2D classification
                if self.num_dim == 2:
                    view = self.slice_view
                    #slice_idx_randomized = random.randint(slice_num-5,slice_num+6)  ##
                    #slice_idx = [slice_idx_randomized]  ##
                    slices = [
                        np.copy(self._get_slice(image, view, idx)[None, :, :])
                            for idx in self.slice_idx ]
                    if len(slices) > 1:
                        image = np.concatenate(slices, axis=0)
                    else:
                        image = slices[0]
                    images.append(image)
                elif self.num_dim == 3:
                    # NOTE: Transforms on 3D images must be performed here. 2D image transforms are performed in __getitem__
                    #image = image[:,30:110,:]
                    images.append(self.transforms(image))
            except Exception as e:
                print("Failed to load #{}: {}".format(idx, paths[idx]))
                print("Errors encountered: {}".format(e))
                print(traceback.format_exc())
                return None

        if len(images) == 3:
            if self.num_dim == 3:
                stacked_image = torch.stack(images)
            elif self.num_dim == 2:
                stacked_image = np.stack(images)
        elif len(images) == 1:
            stacked_image = images[0]
            if self.num_dim == 3:
                stacked_image = stacked_image.unsqueeze(0)
        else:
            raise Exception("Invalid number of images in the images array, expected 1 or 3, got {}.".format(len(images)))

        if self.num_dim == 2:
            # PIL only takes valid images (0,1) or (0, 255)
            stacked_image = NaNToNum()(stacked_image)
            # Get pixel values to between 0 and 255
            stacked_image = np.uint8(RangeNormalization()(stacked_image) * 255)

            if stacked_image.shape[0] == 1:
                stacked_image = np.repeat(stacked_image, 3, axis=0)
                # transpose to (W,H,C) for PIL
                stacked_image = stacked_image.transpose((1,2,0))

        return stacked_image

    def _get_slice(self, mri, view, idx):
        if view == "coronal":
            image = mri[:, idx, :]
        elif view == "axial":
            image = mri[:, :, idx]
        elif view == "sagital":
            image = mri[idx, :, :]
        else:
            raise Exception("Unrecognized slice view: {}" \
                                .format(view))
        return image

    def _split_data(self, df, valid_split, test_split, mode, task):
        if task == "classify":
            if self.customized_split_idx != -1:
                if mode == "train":
                    df = df[:1400]
                elif mode == "valid":
                    df = df[1400:-1]
                else:
                    df = df[-1:]
                return df
       
    def _get_data(self, mapping_path):
        """
        Note: This function is called in repeated times for train, val, test, etc.
        """
        if not os.path.exists(mapping_path):
            raise Exception("Failed to create dataset, \"{}\" does not exist! \
                Run \"utils/normalized_mapping.py\" script to generate mapping."
                .format(mapping_path))

        with open(mapping_path, "rb") as file:
            df = pickle.load(file)
        
        # setup labels encoder
        labels = df[self.label_col].unique()
        encoder = LabelEncoder()
        encoder.fit(labels)
       
        df = df[df["DX"] != "MCI"]
        #df = df.groupby('PTID').first()  
        print(df.shape)
        return df, encoder
