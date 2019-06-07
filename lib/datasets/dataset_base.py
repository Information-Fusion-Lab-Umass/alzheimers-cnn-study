import os
import pickle
import traceback
import nibabel as nib

import torchvision.transforms as T
from torch.utils.data import Dataset

from .dataset_splitters import AllDataSplitter

class DatasetBase(Dataset):
    LABEL_MAPPING = ["CN", "MCI", "AD"]
    VALID_SPLIT = ["train", "valid", "test", "all"]
    VALID_TASKS = ["pretrain", "classify"]
    SPLIT_METHOD = [
        "split_all_data",
        "split_balanced_data"
    ]

    def __init__(self, config, logger, **kwargs):
        super().__init__()
        self.config = config
        self.logger = logger
        self.image_columns = config.image_columns

        shuffle = kwargs.get("shuffle", False)
        mapping_path = config.mapping_path

        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Mapping file \"{mapping_path}\" does not exist! Run \"util/mapping.py\" script to generate mapping.")

        self.dataframe = self._load_data(mapping_path)

        self.split_config = {
            "valid_split": config.validation_split,
            "test_split": config.testing_split
        }

        if os.path.exists(config.brain_mask_path):
            self.brain_mask = nib.load(config.brain_mask_path) \
                                 .get_fdata() \
                                 .squeeze()
        else:
            self.brain_mask = None

        if shuffle:
            self.dataframe = self.dataframe.sample(frac=1)

        self.data_idx = []

    def __len__(self):
        if len(self.data_idx) == 0:
            self.logger.warn("Len of the dataset is 0, make sure to call load_split() methods before iterating through the dataset.")

        return self.data_idx

    def __getitem__(self, idx):
        return None

    def process_item(self, idx):
        paths = [ self.dataframe.iloc[idx][col] for col in self.image_columns ]
        images = []
        label = self.dataframe.iloc[idx][self.config.label_column]

        for path in paths:
            try:
                image = nib.load(path) \
                           .get_fdata() \
                           .squeeze()

                if self.brain_mask is not None:
                    image *= self.brain_mask

            except Exception as e:
                print(f"Failed to load #{idx}: {paths[idx]}")
                print(f"Errors encountered: {e}")
                print(traceback.format_exc())
                return None

            image.append(images)

        return images, label

    def encode_label(self, label):
        """Transforms a string label into integer.
        """
        return self.LABEL_MAPPING.index(label)

    def decode_label(self, num):
        """Transforms an integer label into string.
        """
        return self.LABEL_MAPPING[num]

    def _load_data(self, path):
        """Load data from the mapping file.
        """
        with open(path, "rb") as file:
            df = pickle.load(file)

        # filter out rows with empty image path
        for i in range(len(self.image_col)):
            df = df[df[self.image_col[i]].notnull()].reset_index(drop=True)

        # change LMCI and EMCI to MCI
        target = (df[self.label_col] == "LMCI") | \
                 (df[self.label_col] == "EMCI")
        df.loc[target, self.label_col] = "MCI"

        return df

    def load_split(self, split, method="split_all_data"):
        assert split in VALID_SPLIT, \
            f"Invalid split argument: {split}, valid options are: {VALID_SPLIT}"

        if split == "all":
            self.data_idx = list(range(self.dataframe))

        # In the format of [ (index, label), (index, label), ... ]
        reduced_dataset = list(self.dataframe["DX"].to_dict().items())

        valid_ratio = self.split_config["valid_split"]
        test_ratio = self.split_config["test_split"]
        train_ratio = 1.0 - valid_ratio - test_ratio
        split_ratio = [ train_ratio, valid_ratio, test_ratio ]

        if method == "split_all_data":
            split = AllDataSplitter(reduced_dataset, split_ratio)
            train_split, valid_split, test_split = split()
        else:
            raise Exception(f"Split method {method} not recognized or supported.")

        self.logger.debug(f"Splitting dataset for \"{split}\" split and method \"{method}\". Resulting split size: training - {len(train_split)}, validation {len(valid_split)}, testing - {len(test_split)}")

        if split == "train":
            self.data_idx = train_split
        elif split == "valid":
            self.data_idx = valid_split
        elif split == "test":
            self.data_idx = test_split
