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
    '''CLINICA-normalized dataset for classification task.
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
        # if 2D, which view
        self.slice_view = kwargs.get("slice_view",
                                      self.config["data"]["slice_view"])
        # the index at which to slice
        self.slice_idx = kwargs.get("slice_num",
                                    self.config["data"]["slice_num"])
        if not isinstance(self.slice_idx, list):
            raise Exception("Expected a list for slice_num, but instead got {}".format(self.slice_idx))
        if not (len(self.slice_idx) == 1 or len(self.slice_idx) == 3):
            raise Exception("Expected either 1 or three slices for slice_num, got {} instead.".format(len(self.slice_idx)))

        # limit for the size of the dataset, for debugging purposes
        self.limit = kwargs.get("limit", -1)
        self.verbose = kwargs.get("verbose", self.config["verbose"])

        if "apply_cmap" in self.config["data"]:
            self.apply_cmap = self.config["data"]["apply_cmap"]

        transforms = kwargs.get("transforms", [
            T.ToTensor()
        ])
        self.transforms = T.Compose(transforms)

        # name of the image column in the dataframe
        self.image_col = self.config["image_col"]
        # name of the label column in the dataframe
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
        task = kwargs.get("task", "classify")
        valid_split = kwargs.get("valid_split", 0.2)
        test_split = kwargs.get("test_split", 0.0)

        df, label_encoder = self._get_data(mapping_path)
        input_encoder = kwargs.get("label_encoder", None)
        self.label_encoder = label_encoder if input_encoder is None \
                                           else input_encoder

        self.dataframe = self._split_data(df, valid_split, test_split, mode,
                                          task)

    def __len__(self):
        return len(self.dataframe.index)

    def __getitem__(self, idx):
        image_paths = self._get_img_paths(idx)
        image = self._load_imgs(image_paths)

        label = self.dataframe[self.label_col].iloc[idx]
        encoded_label = self.label_encoder.transform([label])[0]

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

            # apply color map
            if self.apply_cmap:
                stacked_image = plt \
                    .get_cmap("viridis")(stacked_image.squeeze())[:,:,:3]

                stacked_image = np.uint8(RangeNormalization() \
                                    (stacked_image) * 255)

                # Rotate to upright
                stacked_image = np.rot90(stacked_image)
            elif stacked_image.shape[0] == 1:
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
        if mode not in self.VALID_MODES:
            raise Exception("Invalid mode: {}. Valid options are {}"
                                .format(mode, self.VALID_MODES))

        if task not in self.VALID_TASKS:
            raise Exception("Invalid task: {}. Valid options are {}"
                                .format(task, self.VALID_TASKS))

        if not 0.0 <= valid_split <= 1.0:
            raise Exception("Invalid validation split percentage: {}"
                                .format(valid_split))

        if not 0.0 <= test_split <= 1.0:
            raise Exception("Invalid test split percentage: {}"
                                .format(test_split))

        if (valid_split + test_split) >= 1.0:
            raise Exception("valid_split + test_split ({}) is greater than or equal to 1.0".format(valid_split + test_split))

        ad = df[df[self.label_col] == "AD"]
        mci = df[df[self.label_col] == "MCI"]
        cn = df[df[self.label_col] == "CN"]

        ad = shuffle(ad)
        mci = shuffle(mci)
        cn = shuffle(cn)

        size = min(len(ad.index), len(mci.index), len(cn.index)) \
                if self.limit == -1 else self.limit

        if task == "classify":
            ad = self._split_dataframe(ad[:size], valid_split, test_split, mode)
            mci = self._split_dataframe(mci[:size], valid_split, test_split,
                                        mode)
            cn = self._split_dataframe(cn[:size], valid_split, test_split, mode)

            print("Class distribution for {} {}: {} AD, {} MCI, {} CN"
                    .format(task, mode, len(ad.index), len(mci.index),
                            len(cn.index)))
        elif task == "pretrain":
            ad = self._split_dataframe(ad[size:], valid_split, test_split, mode)
            mci = self._split_dataframe(mci[size:], valid_split, test_split,
                                        mode)
            cn = self._split_dataframe(cn[size:], valid_split, test_split, mode)

            if self.limit != -1:
                ad = ad[:self.limit]
                mci = mci[:self.limit]
                cn = cn[:self.limit]

            print("Class distribution for {} {}: {} AD, {} MCI, {} CN"
                    .format(task, mode, len(ad.index), len(mci.index),
                            len(cn.index)))

        return pd.concat([ad, mci, cn])

    def _split_dataframe(self, df, valid_split, test_split, mode):
        train_split = 1 - valid_split - test_split
        num_train = int(len(df.index) * train_split)
        num_valid = int(len(df.index) * valid_split)
        num_test = int(len(df.index) * test_split)

        if mode == "train":
            return df[:num_train].reset_index(drop=True)
        elif mode == "valid":
            start = num_train
            end = start + num_valid
            return df[start:end].reset_index(drop=True)
        elif mode == "test":
            start = num_train + num_valid
            end = start + num_test
            return df[start:end].reset_index(drop=True)
        else:
            return df[:]

    def _get_data(self, mapping_path):
        if not os.path.exists(mapping_path):
            raise Exception("Failed to create dataset, \"{}\" does not exist! Run \"utils/normalized_mapping.py\" script to generate mapping."
                .format(mapping_path))

        with open(mapping_path, "rb") as file:
            df = pickle.load(file)

        # filter out rows with empty label
        df = df[df[self.label_col].notnull()].reset_index()
        # filter out rows with empty image path
        for i in range(len(self.image_col)):
            df = df[df[self.image_col[i]].notnull()].reset_index(drop=True)

        # change LMCI and EMCI to MCI
        target = (df[self.label_col] == "LMCI") | \
                 (df[self.label_col] == "EMCI")
        df.loc[target, self.label_col] = "MCI"

        # setup labels encoder
        labels = df[self.label_col].unique()
        encoder = LabelEncoder()
        encoder.fit(labels)

        df = df.sample(frac=1)

        return df, encoder

if __name__ == "__main__":
    with open("config/2d_classify.yaml") as file:
        config = yaml.load(file)
    dataset = NormalizedDataset(
        num_dim=2,
        slice_view="coronal",
        slice_num=[ 80 ],
        mode="train",
        task="classify",
        valid_split=0.1,
        test_split=0.1,
        limit=-1,
        config=config,
        transforms=[
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            NaNToNum(),
            RangeNormalization()
        ]
    )
    image = dataset[0]

    # with open("config/3d_classify.yaml") as file:
    #     config = yaml.load(file)
    # dataset = NormalizedDataset(
    #     num_dim=3,
    #     slice_view="coronal",
    #     slice_num=[ 80 ],
    #     mode="train",
    #     task="classify",
    #     valid_split=0.1,
    #     test_split=0.1,
    #     limit=-1,
    #     config=config,
    #     transforms=[
    #         T.ToTensor(),
    #         PadToSameDim(),
    #         NaNToNum(),
    #         RangeNormalization()
    #     ]
    # )
    # image = dataset[0]
