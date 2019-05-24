import os
import pickle
import numpy as np
import nibabel as nib
import multiprocessing as mp
import torchvision
import torchvision.transforms as T
import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.loader import invalid_collate
from utils.transforms import OrientFSImage, PadPreprocImage, RangeNormalization, MeanStdNormalization
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

from pdb import set_trace

class ADNIAutoEncDataset(Dataset):
    '''
    ADNI dataset for training auto-encoder. This dataset relies on a mapping file, which is generated with the mapping.py script in utils/. The mapping.py script makes certain assumptions about the location of all of the data files.

    Args:
        **kwargs:
            transforms (List): A list of torchvision.transforms functions, see https://pytorch.org/docs/stable/torchvision/transforms.html for more information.

            mode (string): "train" for training split, "valid" for validation split, and "all" for the whole set, defaults to "all".

            valid_split (float): Percentage of data reserved for validation, ignored if mode is set to "train" or "all, defaults to 0.2.

            limit (int): Size limit for the dataset, used for debugging. This is the total number of images to include for both the training set and validation set. For example, limit=10 with a valid_split of 0.2 will assign 8 images to the training set, 2 images to the validation set. Setting to -1 means to include the whole set. Defaults to -1.
    '''
    def __init__(self, **kwargs):
        config = kwargs.get("config", {})
        mapping_path = kwargs.get("mapping_path", config["label_path"])
        preproc_transforms = kwargs.get("preproc_transforms",
                                        [ T.ToTensor(),
                                          PadPreprocImage(),
                                          MeanStdNormalization() ])
        postproc_transforms = kwargs.get("postproc_transforms",
                                         [ T.ToTensor(),
                                           OrientFSImage(),
                                           MeanStdNormalization() ])
        mode = kwargs.get("mode", "all")

        valid_split = kwargs.get("valid_split", 0.2)
        test_split = kwargs.get("test_split", 0.0)

        self.limit = kwargs.get("limit", -1)
        # column name for the image paths
        self.image_col = config["image_col"][0]
        # column name for the label paths
        self.label_col = config["label_col"]

        self._check_mapping_file(mapping_path)
        self._check_valid_mode(mode)
        self._check_valid_split(valid_split, test_split)

        self.preproc_transforms = T.Compose(preproc_transforms)
        self.postproc_transforms = T.Compose(postproc_transforms)

        with open(mapping_path, "rb") as file:
            self.df = pickle.load(file)

        # filter out the rows where label is empty.
        self.df = self.df[self.df[self.label_col].notnull()].reset_index()
        # split data according to the mode that is set
        self.df = self._split_data(self.df, valid_split, test_split, mode)

    def __len__(self):
        if self.limit == -1:
            return len(self.df.index)
        elif 0 <= self.limit:
            return min(self.limit, len(self.df.index))
        else:
            raise Exception("Invalid dataset size limit set: {}, set size: {}"
                            .format(self.limit, len(self.df.index)))

    def __getitem__(self, idx):
        preproc_path, postproc_path = self._get_paths(idx)

        try:
            preproc_img = nib.load(preproc_path) \
                             .get_fdata() \
                             .squeeze()
            postproc_img = nib.load(postproc_path) \
                              .get_fdata() \
                              .squeeze()

            if (np.isnan(preproc_img).sum() > 0) or \
               (np.isnan(postproc_img).sum() > 0):
                raise Exception("Corrupted image {}.".format(idx))
        except Exception as e:
            print("Failed to load #{}, skipping.".format(idx))
            return None, None

        preproc_img = self.preproc_transforms(preproc_img)
        postproc_img = self.postproc_transforms(postproc_img)

        # Add a "channel" dimension
        preproc_img = preproc_img.unsqueeze(0)
        postproc_img = postproc_img.unsqueeze(0)
        return preproc_img, postproc_img

    def process_images(self, fn, **kwargs):
        '''
        Takes a function and performs it on all images. This method uses DataLoader to parallelize the operation.

        Args:
            fn (function): A function to be applied to each individual images and has a signature of fn(preproc, postproc).
            **kwargs:
                max_proc (int): The maximum number of processes to spawn for DataLoader workers, defaults to half the number of cpu cores.

        Returns:
            dict: A dictionary of outputs from the fn.
        '''
        max_proc = kwargs.get("max_proc", mp.cpu_count() // 2)
        num_workers = max(max_proc, 1)
        loader = DataLoader(self,
                            batch_size=1,
                            num_workers=max_proc,
                            collate_fn=invalid_collate)
        outputs = []

        print("Running process_images with {} DataLoader workers"
                .format(num_workers))
        for idx, images in enumerate(loader):
            if len(images) == 0:
                continue
            preproc, postproc = images

            for i in range(len(preproc)):
                if preproc[i] is None or postproc[i] is None:
                    continue

                output = fn(preproc[i], postproc[i])

                if output is not None:
                    outputs.append(output)
        print("Done!")

        return outputs

    def _get_paths(self, idx):
        '''
        Returns the file paths for the given index.

        Args:
            idx (int): Index of the paths
        Returns:
            tuple: A pair of strings containing the preprocess and post-processed image paths.
        '''
        postproc_img_col = self.image_col
        preproc_path = self.df["preproc_path"].iloc[idx]
        postproc_path = self.df[postproc_img_col].iloc[idx]

        return preproc_path, postproc_path

    def _get_dims(self, preproc, postproc):
        '''
        A function for process_images, returns the shapes of preproc and postproc images.

        Args:
            preproc (torch.Tensor): Preprocessed image
            postproc (torch.Tensor): Postprocessed image

        Returns:
            tuple: Two tuples of shapes for preproc and postproc
        '''
        return (tuple(preproc.shape), tuple(postproc.shape))

    def _check_mapping_file(self, mapping_path):
        if not os.path.exists(mapping_path):
            raise Exception("Failed to create dataset, \"{}\" does not exist! Run \"utils/mapping.py\" script to generate mapping."
                .format(mapping_path))

    def _check_valid_mode(self, mode):
        if mode not in ["train", "valid", "test", "all"]:
            raise Exception("Invalid mode: {}".format(mode))

    def _check_valid_split(self, valid_split, test_split):
        if not 0.0 <= valid_split <= 1.0:
            raise Exception("Invalid validation split percentage: {}"
                                .format(valid_split))

        if not 0.0 <= test_split <= 1.0:
            raise Exception("Invalid test split percentage: {}"
                                .format(test_split))

        if (valid_split + test_split) > 1.0:
            raise Exception("valid_split + test_split ({}) is greater than 1.0".format(valid_split + test_split))

    def _split_data(self, df, valid_split, test_split, mode):
        train_split = 1 - valid_split - test_split
        num_train = int(len(df.index) * train_split)
        num_valid = int(len(df.index) * valid_split)
        num_test = int(len(df.index) * test_split)

        if mode == "train":
            return df[:num_train].reset_index()
        elif mode == "valid":
            start = num_train
            end = start + num_valid
            return df[start:end].reset_index()
        elif mode == "test":
            start = num_train + num_valid
            end = start + num_test
            return df[start:end].reset_index()
        else:
            return df[:]

    def _get_label_encoder(self, labels):
        encoder = LabelEncoder()
        encoder.fit(labels)

        return encoder

class ADNIClassDataset(ADNIAutoEncDataset):
    '''
    ADNI dataset for training classification. This dataset relies on a mapping file, which is generated with the mapping.py script in utils/. The mapping.py script makes certain assumptions about the location of all of the data files.

    Args:
        **kwargs:
            transforms (List): A list of torchvision.transforms functions, see https://pytorch.org/docs/stable/torchvision/transforms.html for more information.

            mode (string): "train" for training split, "valid" for validation split, and "all" for the whole set, defaults to "all".

            valid_split (float): Percentage of data reserved for validation, ignored if mode is set to "train" or "all, defaults to 0.2.
    '''
    def __init__(self, **kwargs):
        # This is to ensure the entire set is loaded by the parent
        mode = kwargs.get("mode", "all")
        kwargs["mode"] = "all"
        super().__init__(**kwargs)

        valid_split = kwargs.get("valid_split", 0.2)
        test_split = kwargs.get("test_split", 0.0)
        self.task = kwargs.get("task", "classify")

        if self.task not in ["classify", "pretrain"]:
            raise Exception("Task {} not recognized.".format(self.task))

        self.df = self._filter_data()

        # Change LMCI and EMCI to MCI
        target = (self.df[self.label_col] == "LMCI") | \
                 (self.df[self.label_col]  == "EMCI")
        self.df.loc[target, self.label_col] = "MCI"

        # Setup labels
        not_null = self.df[self.label_col].notnull()
        labels = self.df[self.label_col][not_null].unique()
        self.label_encoder = self._get_label_encoder(labels)

        # Split data

        # AD = 568, MCI = 1787, and CN = 959
        self.ad = shuffle(self.df[ self.df[self.label_col] == "AD" ])
        self.mci = shuffle(self.df[ self.df[self.label_col] == "MCI" ])
        self.cn = shuffle(self.df[ self.df[self.label_col] == "CN" ])

        if self.limit == -1:
            set_size = 550
        else:
            set_size = self.limit

        if self.task == "classify":
            ad = self._split_data(self.ad[:set_size],
                                valid_split,
                                test_split,
                                mode)
            mci = self._split_data(self.mci[:set_size],
                                valid_split,
                                test_split,
                                mode)
            cn = self._split_data(self.cn[:set_size],
                                valid_split,
                                test_split,
                                mode)
            print("Class distribution for {}/{}: AD - {}, MCI - {}, CN - {}"
                    .format(mode, self.task, len(ad.index), len(mci.index), len(cn.index)))
            self.df = pd.concat([ad, mci, cn])
        elif self.task == "pretrain":
            self.df = pd.concat([
                self.ad[set_size:],
                self.mci[set_size:],
                self.cn[set_size:]
            ])

    def __getitem__(self, idx):
        postproc_path = self.df.iloc[idx][self.image_col]
        label = self._get_label(idx)

        try:
            postproc_img = nib.load(postproc_path) \
                              .get_fdata() \
                              .squeeze()

            if (np.isnan(postproc_img).sum() > 0):
                raise Exception("Corrupted image {}.".format(idx))
        except Exception as e:
            print("Failed to load #{}, skipping.".format(idx))
            return None, None

        postproc_img = self.postproc_transforms(postproc_img)

        # Add a "channel" dimension
        postproc_img = postproc_img.unsqueeze(0)

        return postproc_img[:, 30:-30, 30:-30, 30:-30], label

    def _get_label(self, idx):
        '''
        Returns the patient label, eg AD/MCI/NC, for the given index

        Args:
            idx (int): Index of the patient info
        Return:
            string: AD/MCI/NC
        '''
        label = self.df[self.label_col].iloc[idx]

        return self.label_encoder.transform([label])[0]

    def _filter_data(self):
        '''
        Filter out the records that don't have postproc_path or labels

        Returns:
            pandas.DataFrame: All of the rows that contain postproc_path and labels.
        '''
        has_labels = self.df[self.label_col].notnull()
        has_paths = self.df[self.image_col].notnull()

        return self.df[has_labels & has_paths]

class ADNIAeCnnDataset(ADNIClassDataset):
    def __getitem__(self, idx):
        postproc_path = self.df.iloc[idx][self.image_col]
        label = self._get_label(idx)
        try:
            postproc_img = nib.load(postproc_path) \
                              .get_fdata() \
                              .squeeze()
            if (np.isnan(postproc_img).sum() > 0):
                raise Exception("Corrupted image {}.".format(idx))
        except Exception as e:
            print("Failed to load #{}, skipping.".format(idx))
            return None, None
        #postproc_img = self.postproc_transforms(postproc_img)
        # Add a "channel" dimension
        #postproc_img = postproc_img.unsqeeze(0)
        #patch_samples = getRandomPatches(postproc_img)
        #patch_dict = {"patch": patch_samples}
        return np.ones((5,5,5)), label

    def customToTensor(pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic)
            img = torch.unsqueeze(img,0)
            # backward compatibility
            return img.float()

    def getRandomPatches(image_array):
        patches = []
        mean_ax = np.ndarray.mean(image_array, axis = NON_AX)
        mean_cor = np.ndarray.mean(image_array, axis = NON_COR)
        mean_sag = np.ndarray.mean(image_array, axis = NON_SAG)

        first_ax = int(round(list(mean_ax).index(filter(lambda x: x>0, mean_ax)[0])))
        last_ax = int(round(list(mean_ax).index(filter(lambda x: x>0, mean_ax)[-1])))
        first_cor = int(round(list(mean_cor).index(filter(lambda x: x>0, mean_cor)[0])))
        last_cor = int(round(list(mean_cor).index(filter(lambda x: x>0, mean_cor)[-1])))
        first_sag = int(round(list(mean_sag).index(filter(lambda x: x>0, mean_sag)[0])))
        last_sag = int(round(list(mean_sag).index(filter(lambda x: x>0, mean_sag)[-1])))

        first_ax = first_ax + 20
        last_ax = last_ax - 5

        ax_samples = [random.randint(first_ax - 3, last_ax - 3) for r in xrange(10000)]
        cor_samples = [random.randint(first_cor - 3, last_cor - 3) for r in xrange(10000)]
        sag_samples = [random.randint(first_sag - 3, last_sag - 3) for r in xrange(10000)]

        for i in range(1): #000):
            ax_i = ax_samples[i]
            cor_i = cor_samples[i]
            sag_i = sag_samples[i]
            patch = image_array[ax_i-3:ax_i+4, cor_i-3:cor_i+4, sag_i-3:sag_i+4]
            while (np.ndarray.sum(patch) == 0):
                ax_ni = random.randint(first_ax - 3, last_ax - 4)
                cor_ni = random.randint(first_cor - 3, last_cor - 4)
                sag_ni = random.randint(first_sag - 3, last_sag - 4)
                patch = image_array[ax_ni-3:ax_ni+4, cor_ni-3:cor_ni+4, sag_ni-3:sag_ni+4]
            patch = customToTensor(patch)
            patches.append(patch)
        return patches

if __name__ == "__main__":
    # dataset = ADNIAutoEncDataset()

    # Get the unique shapes for preprocessed and post-processed images
    # shapes = dataset.process_images(dataset._get_dims)
    # preproc_shapes = set(map(lambda x: x[0], shapes))
    # postproc_shapes = set(map(lambda x: x[1], shapes))
    # preproc_shapes -> [(256, 170, 256), (146, 256, 256), (124, 256, 256), (166, 256, 256), (160, 192, 192), (192, 160, 192), (170, 256, 256), (180, 256, 256), (184, 256, 256), (162, 256, 256)]
    # (166, 256, 256) = 694
    # (180, 256, 256) = 264
    # (160, 192, 192) = 621
    # (170, 256, 256) = 46
    # (184, 256, 256) = 9
    # (256, 170, 256) = 1
    # (146, 256, 256) = 1
    # (192, 160, 192) = 1
    # (124, 256, 256) = 1
    # (162, 256, 256) = 2
    # postproc_shapes -> [(256, 256, 256)]

    # Figure out the mean and std of post-processed images
    # dataset = ADNIClassDataset(mode="all")
    pass
